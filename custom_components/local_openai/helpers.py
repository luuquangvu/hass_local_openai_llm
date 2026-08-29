"""Helper utilities for Local OpenAI LLM integration."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import re
import unicodedata
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Literal

import demoji
import orjson
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
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
)
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.shared_params import FunctionDefinition
from openai.types.shared_params.response_format_json_schema import JSONSchema
from pylatexenc.latex2text import LatexNodes2Text
from voluptuous_openapi import convert

from .const import (
    CURRENCY_PATTERN,
    DOMAIN,
    GEMINI_MIME_TYPES_SUPPORTED,
    LATEX_MATH_SPAN,
    LOGGER,
)


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


def _clean_json_data(obj: object) -> object:
    """Recursively stringify and strip dict keys, sort keys, and preserve string values."""
    if isinstance(obj, dict):
        return {
            str(k).strip(): _clean_json_data(v)
            for k, v in sorted(obj.items(), key=lambda item: str(item[0]).strip())
        }
    return [_clean_json_data(v) for v in obj] if isinstance(obj, list) else obj


def _clean_and_sort_schema(schema: object) -> object:
    """Clean descriptions/titles and recursively sort dictionary keys and schema properties."""
    if not isinstance(schema, dict):
        return (
            [_clean_and_sort_schema(item) for item in schema]
            if isinstance(schema, list)
            else schema
        )
    cleaned: dict[str, object] = {}
    for k, v in sorted(schema.items(), key=lambda item: str(item[0]).strip()):
        key_str = str(k).strip()
        if key_str == "properties" and isinstance(v, dict):
            cleaned[key_str] = {
                str(pk).strip(): _clean_and_sort_schema(pv)
                for pk, pv in sorted(v.items(), key=lambda item: str(item[0]).strip())
            }
        elif key_str == "required" and isinstance(v, list):
            cleaned[key_str] = sorted(
                {str(item).strip() for item in v if item is not None and str(item).strip()}
            )
        elif key_str in {"description", "title"} and isinstance(v, str):
            cleaned[key_str] = v.strip()
        else:
            cleaned[key_str] = _clean_and_sort_schema(v)
    return cleaned


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
                for prop, prop_info in sorted(properties.items(), key=lambda item: str(item[0])):
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
                required.sort()

    elif schema.get("type") == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            _adjust_schema(items)


def _format_structured_output(
    name: str, schema: vol.Schema, llm_api: llm.APIInstance | None
) -> JSONSchema:
    """Format the schema to be compatible with OpenRouter API."""
    result: JSONSchema = {
        "name": name.strip(),
        "strict": True,
    }
    result_schema = convert(
        schema,
        custom_serializer=(llm_api.custom_serializer if llm_api else llm.selector_serializer),
    )

    _adjust_schema(result_schema)
    cleaned_schema = _clean_and_sort_schema(result_schema)
    if isinstance(cleaned_schema, dict):
        result["schema"] = cleaned_schema
    else:
        result["schema"] = result_schema
    return result


def _format_tool(
    tool: llm.Tool,
    custom_serializer: Callable[[object], object] | None,
) -> ChatCompletionFunctionToolParam:
    """Format tool specification."""
    raw_parameters = convert(tool.parameters, custom_serializer=custom_serializer)
    cleaned_parameters = _clean_and_sort_schema(raw_parameters)
    parameters: dict[str, object] = (
        cleaned_parameters if isinstance(cleaned_parameters, dict) else {}
    )
    description = (
        tool.description.strip()
        if (tool.description and tool.description.strip())
        else "A callable function"
    )
    tool_spec = FunctionDefinition(
        name=tool.name.strip(),
        parameters=parameters,
        description=description,
    )
    return ChatCompletionFunctionToolParam(type="function", function=tool_spec)


def _convert_completion_content_part_to_response_input(
    part: ChatCompletionContentPartParam | ChatCompletionContentPartRefusalParam,
) -> ResponseInputContentParam:
    """Convert a chat completion content part into responses API format."""
    if part["type"] == "text":
        return ResponseInputTextParam(type="input_text", text=part["text"].strip())
    if part["type"] == "image_url":
        image_url = part["image_url"]["url"].strip()
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
            content_str = (
                raw_content if isinstance(raw_content, str) else str(raw_content)
            ).strip()
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
                        content=str(content).strip() if content is not None else "",
                    )
                )
            continue

        if message["role"] == "assistant":
            raw_content = message.get("content") or ""
            content_str = (
                raw_content if isinstance(raw_content, str) else str(raw_content)
            ).strip()
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
                        raw_args = fn["arguments"]
                        arguments_str = raw_args
                        if isinstance(raw_args, str):
                            try:
                                parsed = orjson.loads(raw_args)
                                arguments_str = orjson.dumps(
                                    _clean_json_data(parsed),
                                    option=orjson.OPT_SORT_KEYS,
                                ).decode("utf-8")
                            except orjson.JSONDecodeError:
                                arguments_str = raw_args.strip()
                        response_messages.append(
                            ResponseFunctionToolCallParam(
                                type="function_call",
                                name=fn["name"].strip(),
                                arguments=arguments_str,
                                call_id=tool_call["id"].strip(),
                            )
                        )
            continue

        if message["role"] == "tool":
            raw_output = message.get("content") or ""
            output_str = raw_output if isinstance(raw_output, str) else str(raw_output)
            try:
                parsed_out = orjson.loads(output_str)
                output_str = orjson.dumps(
                    _clean_json_data(parsed_out),
                    option=orjson.OPT_SORT_KEYS,
                ).decode("utf-8")
            except orjson.JSONDecodeError:
                output_str = output_str.strip()
            response_messages.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=str(message["tool_call_id"]).strip(),
                    output=output_str,
                )
            )

    return response_messages


def b64_file(file_path: Path) -> str:
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
            tool_call_id=content.tool_call_id.strip(),
            content=orjson.dumps(
                _clean_json_data(content.tool_result),
                option=orjson.OPT_SORT_KEYS,
            ).decode("utf-8"),
        )

    if isinstance(content, conversation.SystemContent):
        return ChatCompletionSystemMessageParam(
            role="system",
            content=str(content.content).strip() if content.content is not None else "",
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
                ChatCompletionContentPartTextParam(
                    type="text",
                    text=str(content.content).strip(),
                )
            )

        if content_parts:
            return ChatCompletionUserMessageParam(role="user", content=content_parts)
        return None

    if isinstance(content, conversation.AssistantContent):
        param = AssistantMessageWithReasoning(
            role="assistant",
            content=str(content.content).strip() if content.content is not None else "",
        )
        if (thinking := getattr(content, "thinking_content", None)) is not None:
            param["reasoning_content"] = str(thinking).strip()

        if content.tool_calls:
            param["tool_calls"] = [
                ChatCompletionMessageFunctionToolCallParam(
                    type="function",
                    id=tool_call.id.strip(),
                    function=Function(
                        arguments=orjson.dumps(
                            _clean_json_data(tool_call.tool_args),
                            option=orjson.OPT_SORT_KEYS,
                        ).decode("utf-8"),
                        name=tool_call.tool_name.strip(),
                    ),
                )
                for tool_call in content.tool_calls
            ]
        return param

    return None


def _decode_tool_arguments(arguments: str) -> dict[str, object]:
    """Decode tool call arguments."""
    trimmed = arguments.strip() if arguments else ""
    if not trimmed:
        return {}
    try:
        data = orjson.loads(trimmed)
        if isinstance(data, dict):
            cleaned = _clean_json_data(data)
            return cleaned if isinstance(cleaned, dict) else {}
        return {}
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
                    tool_name=tool_call["name"].strip(),
                    tool_args=_decode_tool_arguments(tool_call["args"])
                    if tool_call["args"]
                    else {},
                )
                for tool_call in pending_tool_calls
                if tool_call["name"].strip()
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
