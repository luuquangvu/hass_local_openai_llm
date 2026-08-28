"""Base entity for Local OpenAI."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import re
import unicodedata
from collections.abc import AsyncGenerator, Callable
from typing import Literal

import demoji
import openai
import orjson
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_MODEL, CONF_PROMPT
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import llm
from homeassistant.helpers.entity import Entity
from openai import AsyncStream
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_param import File, FileFile
from openai.types.chat.chat_completion_message_function_tool_call_param import Function
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseFunctionToolCallParam,
    ResponseInputContentParam,
    ResponseInputImageParam,
    ResponseInputItemParam,
    ResponseInputTextParam,
    ToolChoiceTypesParam,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.response_output_item import ImageGenerationCall
from openai.types.responses.tool_param import ImageGeneration
from openai.types.shared_params import FunctionDefinition, ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema
from pylatexenc.latex2text import LatexNodes2Text
from voluptuous_openapi import convert

from . import LocalAiConfigEntry
from .const import (
    CURRENCY_PATTERN,
    DOMAIN,
    GEMINI_MIME_TYPES_SUPPORTED,
    LATEX_MATH_SPAN,
    LOGGER,
    MAX_TOOL_ITERATIONS,
    LocalAiConfigKey,
)
from .prompt import format_custom_prompt


class AssistantMessageWithReasoning(ChatCompletionAssistantMessageParam, total=False):
    """Chat completion assistant message parameter with reasoning content."""

    reasoning_content: str


async def _strip_emojis(text: str) -> str:
    """Strip emojis from text."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, demoji.replace, text, "")


def _sync_latex_to_text(text: str) -> str:
    """Synchronous helper to convert LaTeX math spans to text."""
    converter = LatexNodes2Text(keep_comments=True, keep_braced_groups=True)

    def replace(match: re.Match[str]) -> str:
        span = match.group(0)
        return converter.latex_to_text(span)

    return LATEX_MATH_SPAN.sub(replace, text)


async def _latex_to_text(text: str) -> str:
    """Convert LaTeX to text asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_latex_to_text, text)


def _is_punctuation(char: str) -> bool:
    """Return True if the character is a punctuation mark."""
    return bool(char) and unicodedata.category(char).startswith("P")


def _is_word_character(char: str) -> bool:
    """Return True if the character can be part of a word."""
    return bool(char) and (char.isalnum() or char == "_")


def _should_strip_emphasis(inner: str, previous: str, following: str) -> bool:
    """Return True if the emphasis markers should be removed."""
    trimmed = inner.strip()
    if not trimmed:
        return False

    if inner != trimmed:
        leading_ws = inner[: len(inner) - len(inner.lstrip())]
        trailing_ws = inner[len(inner.rstrip()) :]
        if leading_ws or trailing_ws:
            return False

    return True


def _sync_strip_emphasis_markers(text: str) -> str:
    """Synchronous helper to strip markdown emphasis markers."""
    length = len(text)
    out: list[str] = []
    i = 0

    while i < length:
        ch = text[i]
        if ch in ("*", "_"):
            marker = ch
            marker_len = 1
            if i + 1 < length and text[i + 1] == marker:
                marker_len = 2

            start = i
            idx = i + marker_len
            prev_char = text[start - 1] if start > 0 else ""
            char_after_open = text[idx] if idx < length else ""

            if not char_after_open or char_after_open.isspace():
                out.append(text[start:idx])
                i = idx
                continue

            if marker == "_" and _is_word_character(prev_char):
                out.append(text[start:idx])
                i = idx
                continue

            search_marker = marker * marker_len
            closing_idx = text.find(search_marker, idx)
            if closing_idx != -1:
                inner = text[idx:closing_idx]
                char_before_close = text[closing_idx - 1] if closing_idx > idx else ""
                follow_idx = closing_idx + marker_len
                following_char = text[follow_idx] if follow_idx < length else ""

                if (
                    not char_before_close.isspace()
                    and not (marker == "_" and _is_word_character(following_char))
                    and _should_strip_emphasis(inner, prev_char, following_char)
                ):
                    out.append(inner)
                    i = follow_idx
                    continue

                out.append(text[start:idx])
                i = idx
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def _consume_emphasis(buffer: str, flush: bool = False) -> tuple[str, str]:
    """Strip emphasis markers from buffer and return (ready_text, pending_buffer)."""
    if flush or not buffer:
        return _sync_strip_emphasis_markers(buffer), ""

    for marker in ("**", "__"):
        if buffer.count(marker) % 2 != 0:
            last_idx = buffer.rfind(marker)
            return _sync_strip_emphasis_markers(buffer[:last_idx]), buffer[last_idx:]

    for marker, double_marker in (("*", "**"), ("_", "__")):
        single_count = buffer.count(marker) - 2 * buffer.count(double_marker)
        if single_count % 2 != 0:
            last_idx = buffer.rfind(marker)
            return _sync_strip_emphasis_markers(buffer[:last_idx]), buffer[last_idx:]

    return _sync_strip_emphasis_markers(buffer), ""


async def _strip_emphasis_markers(text: str) -> str:
    """Strip emphasis markers from text asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_strip_emphasis_markers, text)


def _consume_dollar_latex(buffer: str) -> tuple[str, str] | None:
    """Check for an unclosed single dollar LaTeX delimiter."""
    if buffer.count("$") % 2 == 0:
        return None

    last_dollar_idx = buffer.rfind("$")
    following_text = buffer[last_dollar_idx:]
    if CURRENCY_PATTERN.match(following_text):
        return buffer, ""

    if following_no_dollar := buffer[last_dollar_idx + 1 :]:
        return (
            (buffer, "")
            if following_no_dollar[0].isspace()
            else (buffer[:last_dollar_idx], buffer[last_dollar_idx:])
        )
    return buffer[:last_dollar_idx], buffer[last_dollar_idx:]


def _consume_latex(buffer: str, flush: bool = False) -> tuple[str, str]:
    """Check the buffer for incomplete LaTeX patterns.

    Returns (safe_to_process, keep_in_buffer).
    """
    if flush or not buffer:
        return buffer, ""

    if buffer.count("$$") % 2 != 0:
        last_double = buffer.rfind("$$")
        return buffer[:last_double], buffer[last_double:]

    if (result := _consume_dollar_latex(buffer)) is not None:
        return result

    if match := re.search(r"(\\[a-zA-Z]*)$", buffer):
        start_index = match.start(1)
        return buffer[:start_index], buffer[start_index:]

    last_backslash = buffer.rfind("\\")
    if last_backslash != -1:
        tail = buffer[last_backslash:]
        if tail.count("{") > tail.count("}"):
            return buffer[:last_backslash], buffer[last_backslash:]
        if tail in ("\\", "\\[", "\\("):
            return buffer[:last_backslash], buffer[last_backslash:]

    return buffer, ""


def _attachment_supported(mime_type: str) -> bool:
    """Validate whether the attachment MIME type is supported for the active model."""
    return mime_type.lower() in GEMINI_MIME_TYPES_SUPPORTED if mime_type else False


def _adjust_schema(schema: dict[str, object]) -> None:
    """Adjust the schema to be compatible with OpenRouter API."""
    if schema.get("type") == "object":
        if "properties" not in schema:
            return

        if "required" not in schema:
            schema["required"] = []

        properties = schema.get("properties")
        if isinstance(properties, dict):
            required = schema.get("required")
            if isinstance(required, list):
                for prop, prop_info in properties.items():
                    if isinstance(prop_info, dict):
                        _adjust_schema(prop_info)
                        if prop not in required:
                            prop_type = prop_info.get("type")
                            if isinstance(prop_type, list):
                                if "null" not in prop_type:
                                    prop_type.append("null")
                            elif prop_type:
                                prop_info["type"] = [prop_type, "null"]
                            required.append(prop)

    elif schema.get("type") == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            _adjust_schema(items)


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
        custom_serializer=(llm_api.custom_serializer if llm_api else llm.selector_serializer),
    )

    _adjust_schema(result_schema)

    result["schema"] = result_schema
    return result


def _format_tool(
    tool: llm.Tool,
    custom_serializer: Callable[[object], object] | None,
) -> ChatCompletionFunctionToolParam:
    """Format tool specification."""
    tool_spec = FunctionDefinition(
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
    )
    tool_spec["description"] = (
        tool.description.strip()
        if (tool.description and tool.description.strip())
        else "A callable function"
    )
    return ChatCompletionFunctionToolParam(type="function", function=tool_spec)


def _convert_completion_content_part_to_response_input(
    part: ChatCompletionContentPartParam | ChatCompletionContentPartRefusalParam,
) -> ResponseInputContentParam:
    """Convert a chat completion content part into responses API format."""
    if part["type"] == "text":
        return ResponseInputTextParam(type="input_text", text=part["text"])
    if part["type"] == "image_url":
        image_url = part["image_url"]["url"]
        detail = part["image_url"].get("detail")
        image_detail: Literal["auto", "low", "high", "original"] = (
            detail if detail in ("auto", "low", "high", "original") else "auto"
        )
        return ResponseInputImageParam(
            type="input_image",
            image_url=image_url,
            detail=image_detail,
        )
    return ResponseInputTextParam(type="input_text", text="")


def _convert_completion_messages_to_response_input(
    messages: list[ChatCompletionMessageParam],
) -> list[ResponseInputItemParam]:
    """Convert chat completion style messages into responses API format."""
    response_messages: list[ResponseInputItemParam] = []
    for message in messages:
        if message["role"] == "system":
            raw_content = message.get("content") or ""
            content_str = raw_content if isinstance(raw_content, str) else str(raw_content)
            response_messages.append(
                EasyInputMessageParam(
                    type="message",
                    role="developer",
                    content=content_str,
                )
            )
            continue

        if message["role"] == "user":
            content = message.get("content")
            if isinstance(content, list):
                response_messages.append(
                    EasyInputMessageParam(
                        type="message",
                        role="user",
                        content=[
                            _convert_completion_content_part_to_response_input(part)
                            for part in content
                        ],
                    )
                )
            else:
                response_messages.append(
                    EasyInputMessageParam(
                        type="message",
                        role="user",
                        content=str(content) if content is not None else "",
                    )
                )
            continue

        if message["role"] == "assistant":
            raw_content = message.get("content") or ""
            content_str = raw_content if isinstance(raw_content, str) else str(raw_content)
            response_messages.append(
                EasyInputMessageParam(
                    type="message",
                    role="assistant",
                    content=content_str,
                )
            )
            if tool_calls := message.get("tool_calls"):
                for tool_call in tool_calls:
                    if tool_call["type"] == "function":
                        fn = tool_call["function"]
                        response_messages.append(
                            ResponseFunctionToolCallParam(
                                type="function_call",
                                name=fn["name"],
                                arguments=fn["arguments"],
                                call_id=tool_call["id"],
                            )
                        )
            continue

        if message["role"] == "tool":
            response_messages.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=message["tool_call_id"],
                    output=str(message.get("content") or ""),
                )
            )

    return response_messages


def b64_file(file_path):
    """Retrieve the base64 encoded file contents."""
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def _stringify_keys(obj: object) -> object:
    """Recursively convert dictionary keys to strings."""
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    return [_stringify_keys(v) for v in obj] if isinstance(obj, list) else obj


async def _convert_content_to_chat_message(
    content: object,
    model: str,
) -> ChatCompletionMessageParam | None:
    """Convert any ChatLog content to ChatCompletion message format."""
    if isinstance(content, conversation.ToolResultContent):
        return ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=content.tool_call_id,
            content=orjson.dumps(_stringify_keys(content.tool_result)).decode("utf-8"),
        )

    if isinstance(content, conversation.SystemContent):
        return ChatCompletionSystemMessageParam(
            role="system",
            content=str(content.content) if content.content is not None else "",
        )

    if isinstance(content, conversation.UserContent):
        content_parts: list[ChatCompletionContentPartParam] = []

        if content.attachments:
            for attachment in content.attachments:
                mime_type = (
                    attachment.mime_type
                    or mimetypes.guess_type(str(attachment.path))[0]
                    or "application/octet-stream"
                )

                if not _attachment_supported(mime_type):
                    LOGGER.debug(
                        "Unsupported attachment type '%s' for model '%s'",
                        mime_type,
                        model,
                    )
                    raise HomeAssistantError(
                        translation_domain=DOMAIN,
                        translation_key="unsupported_attachment_type",
                    )

                base64_file = await asyncio.to_thread(b64_file, attachment.path)
                if not base64_file:
                    continue

                if mime_type.startswith("image/"):
                    content_parts.append(
                        ChatCompletionContentPartImageParam(
                            type="image_url",
                            image_url={
                                "url": f"data:{mime_type};base64,{base64_file}",
                            },
                        )
                    )
                    continue

                if mime_type.startswith("audio/"):
                    ext = mimetypes.guess_extension(mime_type)
                    audio_fmt: Literal["wav", "mp3"] | None = None
                    if ext in (".wav", ".mp3"):
                        audio_fmt = "wav" if ext == ".wav" else "mp3"
                    if audio_fmt is not None:
                        content_parts.append(
                            ChatCompletionContentPartInputAudioParam(
                                type="input_audio",
                                input_audio={"format": audio_fmt, "data": base64_file},
                            )
                        )
                        continue

                content_parts.append(
                    File(
                        type="file",
                        file=FileFile(
                            file_data=base64_file,
                            filename=attachment.path.name,
                        ),
                    )
                )

        if content.content:
            content_parts.append(
                ChatCompletionContentPartTextParam(type="text", text=str(content.content))
            )

        if content_parts:
            return ChatCompletionUserMessageParam(role="user", content=content_parts)
        return None

    if isinstance(content, conversation.AssistantContent):
        param = AssistantMessageWithReasoning(
            role="assistant",
            content=str(content.content) if content.content is not None else "",
        )
        if (thinking := getattr(content, "thinking_content", None)) is not None:
            param["reasoning_content"] = str(thinking)

        if content.tool_calls:
            param["tool_calls"] = [
                ChatCompletionMessageFunctionToolCallParam(
                    type="function",
                    id=tool_call.id,
                    function=Function(
                        arguments=orjson.dumps(_stringify_keys(tool_call.tool_args)).decode(
                            "utf-8"
                        ),
                        name=tool_call.tool_name,
                    ),
                )
                for tool_call in content.tool_calls
            ]
        return param

    return None


def _decode_tool_arguments(arguments: str) -> object:
    """Decode tool call arguments."""
    try:
        return orjson.loads(arguments)
    except orjson.JSONDecodeError as err:
        LOGGER.error("Unexpected tool argument response: %s", err)
        raise HomeAssistantError(f"Unexpected tool argument response: {err}") from err


async def _transform_stream(
    stream: AsyncStream[ChatCompletionChunk],
    strip_emojis: bool,
    strip_emphasis: bool,
    strip_latex: bool,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform a streaming OpenAI response to ChatLog format."""
    pending_think = ""
    in_think = False
    seen_visible = False
    pending_tool_calls: list[dict] = []
    pending_emphasis: str = ""
    pending_latex: str = ""

    async for event in stream:
        chunk: conversation.AssistantContentDeltaDict = {}

        if not event.choices:
            continue

        choice = event.choices[0]
        delta = choice.delta

        if (reasoning := getattr(delta, "reasoning_content", None)) is not None:
            chunk["thinking_content"] = reasoning

        if choice.finish_reason and pending_tool_calls:
            chunk["tool_calls"] = [
                llm.ToolInput(
                    tool_name=tool_call["name"],
                    tool_args=orjson.loads(tool_call["args"]) if tool_call["args"] else {},
                )
                for tool_call in pending_tool_calls
                if tool_call["name"]
            ]
            pending_tool_calls = []

        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                index = tool_call.index
                while len(pending_tool_calls) <= index:
                    pending_tool_calls.append({"name": "", "args": ""})
                if tool_call.function:
                    if tool_call.function.name:
                        pending_tool_calls[index]["name"] += tool_call.function.name
                    if tool_call.function.arguments:
                        pending_tool_calls[index]["args"] += tool_call.function.arguments

        if delta.content:
            text = delta.content
            if in_think:
                if "</think>" in text:
                    think_part, _, normal_part = text.partition("</think>")
                    pending_think += think_part
                    in_think = False
                    chunk["thinking_content"] = pending_think
                    pending_think = ""
                    text = normal_part
                else:
                    pending_think += text
                    continue
            elif not seen_visible:
                leading_ws = len(text) - len(text.lstrip())
                stripped = text[leading_ws:]

                if stripped.startswith("<think>"):
                    in_think = True
                    after_tag = stripped[len("<think>") :]
                    if "</think>" in after_tag:
                        think_part, _, normal_part = after_tag.partition("</think>")
                        chunk["thinking_content"] = think_part
                        in_think = False
                        text = normal_part
                    else:
                        pending_think = after_tag
                        continue
                elif not stripped:
                    continue
                else:
                    seen_visible = True
                    text = stripped

            if not text:
                if chunk:
                    yield chunk
                continue

            if strip_latex:
                text, pending_latex = _consume_latex(pending_latex + text, flush=False)
                if not text:
                    if chunk:
                        yield chunk
                    continue
                text = await _latex_to_text(text)

            if strip_emojis:
                text = await _strip_emojis(text)

            if strip_emphasis:
                text, pending_emphasis = _consume_emphasis(pending_emphasis + text, flush=False)
                if not text:
                    if chunk:
                        yield chunk
                    continue

            if text:
                chunk["content"] = text

        if chunk:
            yield chunk

    if pending_latex:
        flushed_latex, _ = _consume_latex(pending_latex, flush=True)
        if flushed_latex:
            flushed_text = await _latex_to_text(flushed_latex)
            if strip_emojis:
                flushed_text = await _strip_emojis(flushed_text)
            if strip_emphasis:
                pending_emphasis += flushed_text
            elif flushed_text:
                yield {"content": flushed_text}

    if pending_emphasis:
        flushed_emphasis, _ = _consume_emphasis(pending_emphasis, flush=True)
        if flushed_emphasis:
            yield {"content": flushed_emphasis}


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
        raw_temp = options.get(LocalAiConfigKey.TEMPERATURE, 1)
        temperature = float(raw_temp) if isinstance(raw_temp, int | float) else 1.0
        parallel_tool_calls = bool(options.get(LocalAiConfigKey.PARALLEL_TOOL_CALLS, True))

        tools: list[ChatCompletionFunctionToolParam] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        messages: list[ChatCompletionMessageParam] = [
            m
            for content in chat_log.content
            if (m := await _convert_content_to_chat_message(content, self.model)) is not None
        ]

        if options.get(LocalAiConfigKey.MANUAL_PROMPTING, False) and user_input:
            prompt = format_custom_prompt(
                self.hass, str(options.get(CONF_PROMPT, "")), user_input, tools
            )
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
        text_output = getattr(response, "output_text", None)

        if (not text_output) and getattr(response, "output", None):
            text_parts: list[str] = []
            text_parts.extend(
                f"![image]({item.result})"
                for item in response.output or ()
                if isinstance(item, ImageGenerationCall)
            )
            if text_parts:
                text_output = "".join(text_parts)

        LOGGER.debug("Extracted text_output before filtering: %s", text_output)
        if text_output:
            text_output = text_output.strip()
        if strip_emojis and text_output:
            text_output = await _strip_emojis(text_output)
        if strip_latex and text_output:
            text_output = await _latex_to_text(text_output)
        if strip_emphasis and text_output:
            text_output = await _strip_emphasis_markers(text_output)

        LOGGER.debug("Final text_output after filtering: %s", text_output)
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
