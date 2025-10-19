"""Base entity for Open Router."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
import json
import base64
from typing import TYPE_CHECKING, Any, Literal

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
)

from openai.types.chat.chat_completion_message_function_tool_call_param import Function
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
from .const import DOMAIN, LOGGER, CONF_STRIP_EMOJIS, CONF_MANUAL_PROMPTING, CONF_MAX_MESSAGE_HISTORY, CONF_TEMPERATURE
from .prompt import format_custom_prompt

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


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
    if tool.description:
        tool_spec["description"] = tool.description
    return ChatCompletionFunctionToolParam(type="function", function=tool_spec)


def b64_file(file_path):
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


async def _convert_content_to_chat_message(
    content: conversation.Content,
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

    if role == "user" and content.content:
        messages = []

        if content.attachments:
            loop = asyncio.get_running_loop()
            for attachment in content.attachments or ():
                if not attachment.mime_type.startswith("image/"):
                    raise HomeAssistantError(
                        translation_domain=DOMAIN,
                        translation_key="unsupported_attachment_type",
                    )
                base64_file = await loop.run_in_executor(None, b64_file, attachment.path)
                messages.append(
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url={
                            "url": f"data:{attachment.mime_type};base64,{base64_file}",
                            "detail": "auto",
                        },
                    )
                )

        messages.append(ChatCompletionContentPartTextParam(type="text", text=content.content))
        return ChatCompletionUserMessageParam(
            role="user",
            content=messages,
        )

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
) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]:
    """Transform a streaming OpenAI response to ChatLog format."""

    new_msg = True
    pending_think = ""
    in_think = False
    seen_visible = False
    loop = asyncio.get_running_loop()
    current_tool_call: dict | None = None

    async for event in stream:
        chunk: conversation.AssistantContentDeltaDict = {}

        choice = event.choices[0]
        delta = choice.delta

        if new_msg:
            chunk["role"] = delta.role
            new_msg = False

        if choice.finish_reason and current_tool_call:
            chunk["tool_calls"] = [
                llm.ToolInput(
                    tool_name=current_tool_call["name"],
                    tool_args=json.loads(current_tool_call["args"]) if current_tool_call["args"] else {},
                )
            ]
            current_tool_call = None

        if (tool_calls := delta.tool_calls) is not None:
            tool_call = tool_calls[0]
            if current_tool_call is None:
                current_tool_call = {
                    "name": tool_call.function.name,
                    "args": tool_call.function.arguments or ""
                }
            else:
                current_tool_call["args"] += tool_call.function.arguments

        if (content := delta.content) is not None:
            if strip_emojis:
                content = await loop.run_in_executor(None, demoji.replace, content, "")

            if content == "<think>":
                in_think = True
                pending_think = ""

            if in_think:
                if content == "</think>":
                    in_think = False
                    if pending_think.strip():
                        LOGGER.debug(f"LLM Thought: {pending_think}")
                    pending_think = ""
                elif content != "<think>":
                    pending_think = pending_think + content
            elif content.strip():
                seen_visible = True

            if seen_visible:
                chunk["content"] = content

        if seen_visible or chunk.get('tool_calls') or chunk.get('role'):
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
    ) -> None:
        """Generate an answer for the chat log."""
        options = self.subentry.data
        strip_emojis = options.get(CONF_STRIP_EMOJIS)
        max_message_history = options.get(CONF_MAX_MESSAGE_HISTORY, 0)
        temperature = options.get(CONF_TEMPERATURE, 0.6)

        model_args = {
            "model": self.model,
            "user": chat_log.conversation_id,
            "extra_headers": {
                "X-Title": "Home Assistant",
            },
            "extra_body": {"require_parameters": True},
            "temperature": temperature,
        }

        tools: list[ChatCompletionFunctionToolParam] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        if tools:
            model_args["tools"] = tools

        messages = self._trim_history([
            m
            for content in chat_log.content
            if (m := await _convert_content_to_chat_message(content))
        ], max_message_history)

        # Full manual prompting - wipe out the HASS-compiled prompt, allow the user to take FULL CONTROL here
        # Additional variables for tools and devices are exposed to the jinja prompt
        if options.get(CONF_MANUAL_PROMPTING, False) and user_input:
            prompt = format_custom_prompt(self.hass, options.get(CONF_PROMPT), user_input, tools)
            messages[0] = ChatCompletionSystemMessageParam(role="system", content=prompt)

        model_args["messages"] = messages

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
                result_stream = await client.chat.completions.create(**model_args, stream=True)
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to API: %s", err)
                raise HomeAssistantError("Error talking to API") from err

            try:
                model_args["messages"].extend(
                    [
                        msg
                        async for content in chat_log.async_add_delta_content_stream(
                            self.entity_id,
                            _transform_stream(result_stream, strip_emojis)
                        )
                        if (msg := await _convert_content_to_chat_message(content))
                    ]
                )
            except Exception as err:
                LOGGER.error("Error talking to API: %s", err)

            if not chat_log.unresponded_tool_results:
                break


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
                *messages[int(drop_index):],
            ]

        return messages
