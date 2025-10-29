"""Base entity for Open Router."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
import json
import base64
import mimetypes
from typing import TYPE_CHECKING, Any, Literal, cast

import demoji
import asyncio
import openai
from openai._streaming import AsyncStream
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
)

from openai.types.chat.chat_completion_message_function_tool_call_param import Function
from openai.types.responses.response_output_item import ImageGenerationCall
from openai.types.shared_params import FunctionDefinition, ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_MODEL, CONF_PROMPT
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity

from . import LocalAiConfigEntry
from .const import (
    DOMAIN,
    LOGGER,
    CONF_STRIP_EMOJIS,
    CONF_STRIP_EMPHASIS,
    CONF_MANUAL_PROMPTING,
    CONF_MAX_MESSAGE_HISTORY,
    CONF_TEMPERATURE,
    GEMINI_MODELS,
)
from .prompt import format_custom_prompt

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

AUDIO_MIME_TYPE_MAP: dict[str, Literal["mp3", "wav"]] = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mpeg3": "mp3",
    "audio/x-mpeg-3": "mp3",
    "audio/x-mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/vnd.wave": "wav",
}


def _should_strip_emphasis(inner: str) -> bool:
    """Return True if the emphasis markers should be removed."""
    trimmed = inner.strip()
    if not trimmed:
        return True
    if inner != trimmed:
        return False
    return True


def _consume_emphasis(buffer: str, flush: bool = False) -> tuple[str, str]:
    """Strip emphasis markers from the buffer and return remaining text."""
    output_parts: list[str] = []
    idx = 0
    length = len(buffer)

    while idx < length:
        start = buffer.find("**", idx)
        if start == -1:
            output_parts.append(buffer[idx:])
            return "".join(output_parts), ""

        output_parts.append(buffer[idx:start])
        end = buffer.find("**", start + 2)

        if end == -1:
            if flush:
                output_parts.append(buffer[start:])
                return "".join(output_parts), ""
            return "".join(output_parts), buffer[start:]

        inner = buffer[start + 2 : end]
        if _should_strip_emphasis(inner):
            output_parts.append(inner)
        else:
            output_parts.append(buffer[start : end + 2])

        idx = end + 2

    return "".join(output_parts), ""


def _strip_markdown_emphasis(text: str) -> str:
    """Remove Markdown bold markers while avoiding math/operator usage."""
    cleaned, _ = _consume_emphasis(text, flush=True)
    return cleaned


def _is_gemini_model(model: str | None) -> bool:
    """Return True if the model is identified as a Gemini model."""
    if not model:
        return False
    model_name = model.lower()
    return any(identifier in model_name for identifier in GEMINI_MODELS)


def _attachment_supported(mime_type: str, model: str | None) -> bool:
    """Validate whether the attachment MIME type is supported for the active model."""
    mime_type = mime_type.lower()

    if mime_type.startswith("image/") or mime_type == "application/pdf":
        return True

    if _is_gemini_model(model) and mime_type.startswith(("audio/", "video/", "text/")):
        return True

    return False


def _adjust_schema(schema: dict[str, Any]) -> None:
    """Adjust the schema to be compatible with OpenRouter API."""
    if schema["type"] == "object":
        if "properties" not in schema:
            return

        if "required" not in schema:
            schema["required"] = []

        # Ensure all properties are required
        for prop, prop_info in schema["properties"].items():
            _adjust_schema(prop_info)
            if prop not in schema["required"]:
                prop_info["type"] = [prop_info["type"], "null"]
                schema["required"].append(prop)

    elif schema["type"] == "array":
        if "items" not in schema:
            return

        _adjust_schema(schema["items"])


def _format_structured_output(
    name: str, schema: vol.Schema, llm_api: llm.APIInstance | None
) -> JSONSchema:
    """Format the schema to be compatible with OpenRouter API."""

    result: JSONSchema = {
        "name": name,
        "strict": True,
    }
    result_schema = convert(
        schema,
        custom_serializer=(
            llm_api.custom_serializer if llm_api else llm.selector_serializer
        ),
    )

    _adjust_schema(result_schema)

    result["schema"] = result_schema
    return result


def _format_tool(
    tool: llm.Tool,
    custom_serializer: Callable[[Any], Any] | None,
) -> ChatCompletionFunctionToolParam:
    """Format tool specification."""
    tool_spec = FunctionDefinition(
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
    )
    tool_spec["description"] = (
        tool.description if tool.description.strip() else "A callable function"
    )
    return ChatCompletionFunctionToolParam(type="function", function=tool_spec)


def _convert_completion_content_part_to_response_input(
    part: ChatCompletionContentPartParam,
) -> dict[str, Any]:
    """Convert a chat completion content part into responses API format."""
    if part["type"] == "text":
        return {"type": "input_text", "text": part["text"]}
    if part["type"] == "image_url":
        image_entry: dict[str, Any] = {
            "type": "input_image",
            "image_url": part["image_url"]["url"],
        }
        detail = part["image_url"].get("detail")
        if detail:
            image_entry["detail"] = detail
        return image_entry
    return {"type": "input_text", "text": ""}


def _convert_completion_messages_to_response_input(
    messages: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    """Convert chat completion style messages into responses API format."""
    response_messages: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            response_messages.append(
                {
                    "type": "message",
                    "role": "developer",
                    "content": message.get("content") or "",
                }
            )
            continue

        if role == "user":
            content = message.get("content")
            if isinstance(content, list):
                response_messages.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            _convert_completion_content_part_to_response_input(part)
                            for part in content
                        ],
                    }
                )
            else:
                response_messages.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": content or "",
                    }
                )
            continue

        if role == "assistant":
            response_messages.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": message.get("content") or "",
                }
            )
            tool_calls = message.get("tool_calls")
            if tool_calls:
                for tool_call in tool_calls:
                    response_messages.append(
                        {
                            "type": "function_call",
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"],
                            "call_id": tool_call["id"],
                        }
                    )
            continue

        if role == "tool":
            response_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message.get("content") or "",
                }
            )

    return response_messages


def b64_file(file_path):
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


async def _convert_content_to_chat_message(
    content: conversation.Content,
    model: str | None = None,
) -> ChatCompletionMessageParam | None:
    """Convert any native chat message for this agent to the native format."""
    if isinstance(content, conversation.ToolResultContent):
        return ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=content.tool_call_id,
            content=json.dumps(content.tool_result),
        )

    role: Literal["user", "assistant", "system"] = content.role
    if role == "system" and content.content:
        return ChatCompletionSystemMessageParam(role="system", content=content.content)

    if role == "user":
        content_parts: list[ChatCompletionContentPartParam] = []
        attachments = getattr(content, "attachments", None) or ()

        if attachments:
            loop = asyncio.get_running_loop()
            for attachment in attachments:
                raw_mime_type = (
                    attachment.mime_type
                    or mimetypes.guess_type(str(attachment.path))[0]
                    or "application/octet-stream"
                )

                if not _attachment_supported(raw_mime_type, model):
                    raise HomeAssistantError(
                        translation_domain=DOMAIN,
                        translation_key="unsupported_attachment_type",
                    )

                base64_file = await loop.run_in_executor(
                    None, b64_file, attachment.path
                )
                mime_type = raw_mime_type.lower()

                if mime_type.startswith("image/"):
                    content_parts.append(
                        ChatCompletionContentPartImageParam(
                            type="image_url",
                            image_url={
                                "url": f"data:{raw_mime_type};base64,{base64_file}",
                                "detail": "auto",
                            },
                        )
                    )
                    continue

                if (audio_format := AUDIO_MIME_TYPE_MAP.get(mime_type)) is not None:
                    content_parts.append(
                        ChatCompletionContentPartInputAudioParam(
                            type="input_audio",
                            input_audio={"format": audio_format, "data": base64_file},
                        )
                    )
                    continue

                content_parts.append(
                    cast(
                        ChatCompletionContentPartParam,
                        {
                            "type": "file",
                            "file": {
                                "file_data": base64_file,
                                "filename": attachment.path.name,
                            },
                        },
                    )
                )

        if content.content:
            content_parts.append(
                ChatCompletionContentPartTextParam(type="text", text=content.content)
            )

        if content_parts:
            return ChatCompletionUserMessageParam(role="user", content=content_parts)
        return None

    if role == "assistant":
        param = ChatCompletionAssistantMessageParam(
            role="assistant",
            content=content.content,
        )
        if isinstance(content, conversation.AssistantContent) and content.tool_calls:
            param["tool_calls"] = [
                ChatCompletionMessageFunctionToolCallParam(
                    type="function",
                    id=tool_call.id,
                    function=Function(
                        arguments=json.dumps(tool_call.tool_args),
                        name=tool_call.tool_name,
                    ),
                )
                for tool_call in content.tool_calls
            ]
        return param
    LOGGER.warning("Could not convert message to Completions API: %s", content)
    return None


def _decode_tool_arguments(arguments: str) -> Any:
    """Decode tool call arguments."""
    try:
        return json.loads(arguments)
    except json.JSONDecodeError as err:
        raise HomeAssistantError(f"Unexpected tool argument response: {err}") from err


async def _transform_stream(
    stream: AsyncStream[ChatCompletionChunk],
    strip_emojis: bool,
    strip_emphasis: bool,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]:
    """Transform a streaming OpenAI response to ChatLog format."""

    new_msg = True
    pending_think = ""
    in_think = False
    seen_visible = False
    loop = asyncio.get_running_loop()
    current_tool_call: dict | None = None
    pending_emphasis: str = ""

    async for event in stream:
        chunk: conversation.AssistantContentDeltaDict = {}

        if not event.choices:
            continue

        choice = event.choices[0]
        delta = choice.delta

        if new_msg:
            chunk["role"] = delta.role
            new_msg = False
            pending_emphasis = ""

        if choice.finish_reason and current_tool_call:
            chunk["tool_calls"] = [
                llm.ToolInput(
                    tool_name=current_tool_call["name"],
                    tool_args=json.loads(current_tool_call["args"])
                    if current_tool_call["args"]
                    else {},
                )
            ]
            current_tool_call = None

        if (tool_calls := delta.tool_calls) is not None and tool_calls:
            tool_call = tool_calls[0]
            if current_tool_call is None:
                current_tool_call = {
                    "name": tool_call.function.name,
                    "args": tool_call.function.arguments or "",
                }
            else:
                current_tool_call["args"] += tool_call.function.arguments

        content_segments: list[str] = []

        if (content := delta.content) is not None:
            if strip_emojis:
                content = await loop.run_in_executor(None, demoji.replace, content, "")
            if strip_emphasis and content:
                pending_emphasis += content
                content, pending_emphasis = _consume_emphasis(
                    pending_emphasis, flush=False
                )
            elif not strip_emphasis:
                pending_emphasis = ""
            else:
                content = ""

            if content:
                content_segments.append(content)

        if strip_emphasis and choice.finish_reason and pending_emphasis:
            flushed, pending_emphasis = _consume_emphasis(pending_emphasis, flush=True)
            if flushed:
                content_segments.append(flushed)

        combined_output = ""
        for segment in content_segments:
            if segment == "<think>":
                in_think = True
                pending_think = ""

            if in_think:
                if segment == "</think>":
                    in_think = False
                    if pending_think.strip():
                        LOGGER.debug(f"LLM Thought: {pending_think}")
                    pending_think = ""
                elif segment != "<think>":
                    pending_think = pending_think + segment
            elif segment.strip():
                seen_visible = True

            combined_output += segment

        if seen_visible and combined_output:
            chunk["content"] = combined_output

        if seen_visible or chunk.get("tool_calls") or chunk.get("role"):
            yield chunk


class LocalAiEntity(Entity):
    """Base entity for Open Router."""

    _attr_has_entity_name = True

    def __init__(self, entry: LocalAiConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self.model = subentry.data[CONF_MODEL]
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

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
        strip_emojis = bool(options.get(CONF_STRIP_EMOJIS))
        strip_emphasis = bool(options.get(CONF_STRIP_EMPHASIS))
        max_message_history = options.get(CONF_MAX_MESSAGE_HISTORY, 0)
        temperature = options.get(CONF_TEMPERATURE, 0.6)

        tools: list[ChatCompletionFunctionToolParam] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        messages = self._trim_history(
            [
                m
                for content in chat_log.content
                if (m := await _convert_content_to_chat_message(content, self.model))
            ],
            max_message_history,
        )

        # Full manual prompting - wipe out the HASS-compiled prompt, allow the user to take FULL CONTROL here
        # Additional variables for tools and devices are exposed to the jinja prompt
        if options.get(CONF_MANUAL_PROMPTING, False) and user_input:
            prompt = format_custom_prompt(
                self.hass, options.get(CONF_PROMPT), user_input, tools
            )
            messages[0] = ChatCompletionSystemMessageParam(
                role="system", content=prompt
            )

        if force_image:
            await self._async_handle_image_response(
                chat_log,
                messages,
                strip_emojis,
                strip_emphasis,
                temperature,
            )
            return

        model_args = {
            "model": self.model,
            "user": chat_log.conversation_id,
            "temperature": temperature,
            "messages": messages,
        }

        if tools:
            model_args["tools"] = tools

        if structure:
            if TYPE_CHECKING:
                assert structure_name is not None
            model_args["response_format"] = ResponseFormatJSONSchema(
                type="json_schema",
                json_schema=_format_structured_output(
                    structure_name, structure, chat_log.llm_api
                ),
            )

        client = self.entry.runtime_data

        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                result_stream = await client.chat.completions.create(
                    **model_args, stream=True
                )
            except openai.OpenAIError as err:
                LOGGER.error("Error requesting response from API: %s", err)
                raise HomeAssistantError("Error talking to API") from err

            try:
                model_args["messages"].extend(
                    [
                        msg
                        async for content in chat_log.async_add_delta_content_stream(
                            self.entity_id,
                            _transform_stream(
                                result_stream, strip_emojis, strip_emphasis
                            ),
                        )
                        if (
                            msg := await _convert_content_to_chat_message(
                                content, self.model
                            )
                        )
                    ]
                )
            except Exception as err:  # pylint: disable=broad-except
                LOGGER.error("Error handling API response: %s", err)

            if not chat_log.unresponded_tool_results:
                break

    async def _async_handle_image_response(
        self,
        chat_log: conversation.ChatLog,
        messages: list[ChatCompletionMessageParam],
        strip_emojis: bool,
        strip_emphasis: bool,
        temperature: float,
    ) -> None:
        """Generate an image response using the Responses API."""
        response_input = _convert_completion_messages_to_response_input(messages)

        model_args: dict[str, Any] = {
            "model": self.model,
            "input": response_input,
            "user": chat_log.conversation_id,
            "temperature": temperature,
            "stream": False,
            "store": True,
            "tool_choice": {"type": "image_generation"},
            "tools": [
                {
                    "type": "image_generation",
                    "model": self.model,
                    "output_format": "png",
                }
            ],
        }

        client = self.entry.runtime_data

        try:
            response = await client.responses.create(**model_args)
        except openai.OpenAIError as err:
            LOGGER.error("Error requesting image response from API: %s", err)
            raise HomeAssistantError("Error talking to API") from err

        text_output = getattr(response, "output_text", None)

        if (not text_output) and getattr(response, "output", None):
            text_parts: list[str] = []
            for item in response.output or ():
                content = getattr(item, "content", None)
                if not content:
                    continue
                for part in content or []:
                    if getattr(part, "type", None) == "output_text":
                        text_parts.append(getattr(part, "text", ""))
            if text_parts:
                text_output = "".join(text_parts)

        if text_output:
            text_output = text_output.strip()
        if strip_emojis and text_output:
            text_output = demoji.replace(text_output, "")
        if strip_emphasis and text_output:
            text_output = _strip_markdown_emphasis(text_output)
        if text_output == "":
            text_output = None

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

    @staticmethod
    def _trim_history(messages: list, max_messages: int) -> list:
        """Trims excess messages from a single history.

        This sets the max history to allow a configurable size history may take
        up in the context window.

        Logic borrowed from the Ollama integration with thanks
        """
        if max_messages < 1:
            # Keep all messages
            return messages

        # Ignore the in progress user message
        num_previous_rounds = sum(m["role"] == "assistant" for m in messages) - 1
        if num_previous_rounds >= max_messages:
            # Trim history but keep system prompt (first message).
            # Every other message should be an assistant message, so keep 2x
            # message objects. Also keep the last in progress user message
            num_keep = 2 * max_messages + 1
            drop_index = len(messages) - num_keep
            messages = [
                messages[0],
                *messages[int(drop_index) :],
            ]

        return messages
