"""Replaces Some of Home Assistant's helpers/llm.py code to allow us to choose the correct prompt."""
from langfuse.decorators import langfuse_context, observe

from homeassistant.components.conversation import (
    ChatLog,
    ConversationInput,
    ConverseError,
    SystemContent,
    trace,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import intent, llm

from . import CustomConversationConfigEntry
from .api import CustomLLMAPI
from .const import DOMAIN, LLM_API_ID, LOGGER
from .prompt_manager import PromptContext, PromptManager


@observe(name="cc_update_llm_data", capture_input=False)
async def async_update_llm_data(
    hass: HomeAssistant,
    user_input: ConversationInput,
    config_entry: CustomConversationConfigEntry,
    chat_log: ChatLog,
    prompt_manager: PromptManager,
    llm_api_names: list[str] | None = None,
):
    """Process the incoming message for the LLM.

    Overrides the session's async_process_llm_message method
    to allow us to implement prompt management and support multiple APIs.
    """

    llm_context = llm.LLMContext(
        platform=DOMAIN,
        context=user_input.context,
        language=user_input.language,
        assistant="conversation", # Todo: Confirm
        device_id=user_input.device_id,
    )

    user_name: str | None = None

    if (
        user_input.context
        and user_input.context.user_id
        and (
            user := await hass.auth.async_get_user(user_input.context.user_id)
            )
    ):
        user_name = user.name

    # Collect all API instances and their tools
    api_instances: list[llm.APIInstance] = []
    all_tools: list[llm.Tool] = []
    api_prompts: list[str] = []

    if llm_api_names:
        for api_name in llm_api_names:
            try:
                if api_name == LLM_API_ID:
                    LOGGER.debug("Using Custom LLM API for request")
                    api_instance = CustomLLMAPI(
                        hass,
                        user_name,
                        conversation_config_entry=config_entry,
                    )
                    if (
                        langfuse_client := hass.data.get(DOMAIN,{})
                        .get(config_entry.entry_id,{})
                        .get("langfuse_client")
                    ):
                        LOGGER.debug("Setting langfuse client for Custom LLM API")
                        api_instance.set_langfuse_client(langfuse_client)
                    llm_api = await api_instance.async_get_api_instance(llm_context)
                else:
                    LOGGER.debug("Using LLM API with ID %s", api_name)
                    llm_api = await llm.async_get_api(
                        hass,
                        api_name,
                        llm_context,
                    )

                api_instances.append(llm_api)
                all_tools.extend(llm_api.tools)

                # Collect API prompts
                api_prompt = await llm_api.api_prompt
                if isinstance(api_prompt, tuple):
                    _, prompt_text = api_prompt
                else:
                    prompt_text = api_prompt
                api_prompts.append(prompt_text)

            except HomeAssistantError as err:
                LOGGER.error(
                    "Error getting LLM API %s for %s: %s",
                    api_name,
                    DOMAIN,
                    err,
                )
                # Continue with other APIs instead of failing completely
                continue

    # Build combined prompt
    prompt_object = None
    try:
        prompt_context = PromptContext(
            hass=hass,
            ha_name=hass.config.location_name,
            user_name=user_name,
        )

        # Check if we have CustomLLMAPI (it provides its own complete prompt)
        has_custom_api = any(
            isinstance(api.api, CustomLLMAPI) for api in api_instances
        )

        if has_custom_api and len(api_instances) == 1:
            # Single CustomLLMAPI - use its prompt directly
            llm_api = api_instances[0]
            prompt = await llm_api.api_prompt
            if isinstance(prompt, tuple):
                LOGGER.debug("Retrieved Langfuse Prompt")
                prompt_object, prompt = prompt
            LOGGER.debug("LLM API prompt: %s", prompt)
        elif not api_instances:
            # No API is enabled - just get the base prompt
            prompt = await prompt_manager.async_get_base_prompt(
                prompt_context,
                config_entry,
            )
            if isinstance(prompt, tuple):
                LOGGER.debug("Retrieved Basic Langfuse Prompt")
                prompt_object, prompt = prompt
            LOGGER.debug("Base prompt: %s", prompt)
        else:
            # Multiple APIs or external APIs - combine base prompt with all API prompts
            base_prompt = await prompt_manager.async_get_base_prompt(
                prompt_context,
                config_entry,
            )
            if isinstance(base_prompt, tuple):
                prompt_object, base_prompt_text = base_prompt
            else:
                base_prompt_text = base_prompt

            prompt_parts = [base_prompt_text]
            prompt_parts.extend(api_prompts)
            prompt = "\n".join(prompt_parts)
            LOGGER.debug("Combined prompt from %d APIs: %s", len(api_prompts), prompt)

    except TemplateError as err:
        LOGGER.error("Error rendering prompt: %s", err)
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_error(
            intent.IntentResponseErrorCode.UNKNOWN,
            "Sorry, I had a problem with my template",
        )
        raise ConverseError(
            "Error rendering prompt",
            conversation_id=chat_log.conversation_id,
            response=intent_response,
        ) from err

    extra_system_prompt = (
        # Take new system prompt if one was given
        user_input.extra_system_prompt or chat_log.extra_system_prompt
    )

    if extra_system_prompt:
        LOGGER.debug("Using extra system prompt: %s", extra_system_prompt)
        prompt += "\n" + extra_system_prompt
        langfuse_context.update_current_trace(tags=["extra_system_prompt"])

    # Create a combined API instance with all tools if we have multiple APIs
    if api_instances:
        if len(api_instances) == 1:
            # Single API - use it directly
            chat_log.llm_api = api_instances[0]
        else:
            # Multiple APIs - create a synthetic APIInstance with combined tools
            LOGGER.debug(
                "Combining tools from %d APIs: %d total tools",
                len(api_instances),
                len(all_tools)
            )
            chat_log.llm_api = llm.APIInstance(
                api=api_instances[0].api,  # Use first API for reference
                api_prompt=prompt,
                llm_context=llm_context,
                tools=all_tools,  # Combined tools from all APIs
                custom_serializer=llm.selector_serializer,
            )
    else:
        chat_log.llm_api = None

    chat_log.extra_system_prompt = extra_system_prompt
    chat_log.content[0] = SystemContent(content=prompt)
    trace.async_conversation_trace_append(
        trace.ConversationTraceEventType.AGENT_DETAIL,
        {
            "messages": chat_log.content,
            "tools": chat_log.llm_api.tools if chat_log.llm_api else None,
        }
    )
    return prompt_object
