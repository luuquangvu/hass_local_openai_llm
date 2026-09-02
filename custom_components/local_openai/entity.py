"""Base entity for Local OpenAI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import openai

if TYPE_CHECKING:
    import voluptuous as vol
else:
    try:
        import probatio as vol
    except ImportError:
        import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_MODEL, CONF_PROMPT
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import Entity
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)
from openai.types.responses import ToolChoiceTypesParam
from openai.types.responses.response_output_item import ImageGenerationCall
from openai.types.responses.tool_param import ImageGeneration
from openai.types.shared_params import ResponseFormatJSONSchema

from . import LocalAiConfigEntry
from .const import DOMAIN, LOGGER, MAX_TOOL_ITERATIONS, LocalAiConfigKey
from .helpers import (
    _convert_completion_messages_to_response_input,
    _convert_content_to_chat_message,
    _format_structured_output,
    _format_tool,
    _latex_to_text,
    _strip_emojis,
    _strip_emphasis_markers,
    _transform_stream,
)
from .prompt import format_custom_prompt


class LocalAiEntity(Entity):
    """Base entity for Local OpenAI."""

    _attr_has_entity_name = True

    def __init__(self, entry: LocalAiConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="Local OpenAI",
            model=subentry.data.get(CONF_MODEL, "Local"),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def model(self) -> str:
        """Return the model name."""
        return str(self.subentry.data.get(CONF_MODEL, "Local"))

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
        user_input: conversation.ConversationInput | None = None,
        force_image: bool = False,
    ) -> None:
        """Generate an answer for the chat log."""
        options = self.subentry.data
        strip_emojis = bool(options.get(LocalAiConfigKey.STRIP_EMOJIS))
        strip_emphasis = bool(options.get(LocalAiConfigKey.STRIP_EMPHASIS))
        strip_latex = bool(options.get(LocalAiConfigKey.STRIP_LATEX))
        if structure is not None:
            strip_emphasis = False
            strip_latex = False
        raw_temp = options.get(LocalAiConfigKey.TEMPERATURE, 1)
        temperature = float(raw_temp) if isinstance(raw_temp, int | float) else 1.0
        parallel_tool_calls = bool(options.get(LocalAiConfigKey.PARALLEL_TOOL_CALLS, True))

        tools: list[ChatCompletionFunctionToolParam] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in sorted(chat_log.llm_api.tools, key=lambda tool: tool.name)
            ]

        messages: list[ChatCompletionMessageParam] = [
            m
            for content in chat_log.content
            if (m := await _convert_content_to_chat_message(content, self.model)) is not None
        ]

        if options.get(LocalAiConfigKey.MANUAL_PROMPTING, False) and user_input:
            prompt = format_custom_prompt(
                self.hass, str(options.get(CONF_PROMPT, "")), user_input, tools
            ).strip()
            new_system_message = ChatCompletionSystemMessageParam(role="system", content=prompt)
            found = False
            for i, msg in enumerate(messages):
                if msg["role"] == "system":
                    messages[i] = new_system_message
                    found = True
                    break

            if not found:
                messages.insert(0, new_system_message)

        if force_image:
            await self._async_handle_image_response(
                chat_log,
                messages,
                strip_emojis,
                strip_emphasis,
                strip_latex,
                temperature,
            )
            return

        client = self.entry.runtime_data

        for _iteration in range(MAX_TOOL_ITERATIONS):
            LOGGER.debug("Sending chat request to API for model: %s", self.model)
            try:
                if structure:
                    response_format = ResponseFormatJSONSchema(
                        type="json_schema",
                        json_schema=_format_structured_output(
                            structure_name or "response", structure, chat_log.llm_api
                        ),
                    )
                    if tools:
                        result_stream = await client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            tools=tools,
                            temperature=temperature,
                            parallel_tool_calls=parallel_tool_calls,
                            prompt_cache_key=chat_log.conversation_id,
                            response_format=response_format,
                            stream=True,
                        )
                    else:
                        result_stream = await client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=temperature,
                            parallel_tool_calls=parallel_tool_calls,
                            prompt_cache_key=chat_log.conversation_id,
                            response_format=response_format,
                            stream=True,
                        )
                elif tools:
                    result_stream = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        parallel_tool_calls=parallel_tool_calls,
                        prompt_cache_key=chat_log.conversation_id,
                        stream=True,
                    )
                else:
                    result_stream = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        parallel_tool_calls=parallel_tool_calls,
                        prompt_cache_key=chat_log.conversation_id,
                        stream=True,
                    )
            except openai.OpenAIError as err:
                LOGGER.error("Error requesting response from API: %s", err)
                raise HomeAssistantError(f"Error talking to API: {err}") from err

            try:
                messages.extend(
                    [
                        msg
                        async for content in chat_log.async_add_delta_content_stream(
                            self.entity_id,
                            _transform_stream(
                                result_stream, strip_emojis, strip_emphasis, strip_latex
                            ),
                        )
                        if (msg := await _convert_content_to_chat_message(content, self.model))
                        is not None
                    ]
                )
            except Exception as err:
                LOGGER.exception("Error handling API response: %s", err)
                break

            if not chat_log.unresponded_tool_results:
                break

    async def _async_handle_image_response(
        self,
        chat_log: conversation.ChatLog,
        messages: list[ChatCompletionMessageParam],
        strip_emojis: bool,
        strip_emphasis: bool,
        strip_latex: bool,
        temperature: float,
    ) -> None:
        """Generate an image response using the Responses API."""
        response_input = _convert_completion_messages_to_response_input(messages)

        client = self.entry.runtime_data

        LOGGER.debug("Sending image generation request to API for model: %s", self.model)
        try:
            response = await client.responses.create(
                model=self.model,
                input=response_input,
                prompt_cache_key=chat_log.conversation_id,
                temperature=temperature,
                stream=False,
                store=True,
                tool_choice=ToolChoiceTypesParam(type="image_generation"),
                tools=[
                    ImageGeneration(
                        type="image_generation",
                        model=self.model,
                        output_format="png",
                    )
                ],
            )
        except openai.OpenAIError as err:
            LOGGER.error("Error requesting image response from API: %s", err)
            raise HomeAssistantError(f"Error talking to API: {err}") from err

        LOGGER.debug("Received image response from API: %s", response)
        raw_text_output = getattr(response, "output_text", None)
        text_output: str | None = str(raw_text_output).strip() if raw_text_output else None

        LOGGER.debug("Extracted text_output before filtering: %s", text_output)
        if strip_emojis and text_output:
            text_output = await _strip_emojis(text_output)
        if strip_latex and text_output:
            text_output = await _latex_to_text(text_output)
        if strip_emphasis and text_output:
            text_output = await _strip_emphasis_markers(text_output)

        if text_output:
            text_output = text_output.strip() or None

        LOGGER.debug("Final text_output after filtering: %s", text_output)

        image_call: ImageGenerationCall | None = None
        for item in response.output or ():
            if isinstance(item, ImageGenerationCall):
                if image_call is None or image_call.result is None:
                    image_call = item
                else:
                    item.result = None

        if image_call is None and text_output is None:
            raise HomeAssistantError("No image response returned from API")

        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id,
                content=text_output,
                native=image_call,
            )
        )
