"""AI Task integration for Local OpenAI LLM."""

from __future__ import annotations

import base64
import binascii

import orjson
from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from openai.types.responses.response_output_item import ImageGenerationCall

from . import LocalAiConfigEntry
from .const import (
    LOGGER,
    LocalAiConfigKey,
    LocalAiSubentryType,
)
from .entity import LocalAiEntity
from .helpers import _clean_json_data


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: LocalAiConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != LocalAiSubentryType.AI_TASK_DATA:
            continue

        async_add_entities(
            [LocalAITaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class LocalAITaskEntity(
    LocalAiEntity,
    ai_task.AITaskEntity,
):
    """Local OpenAI LLM AI Task entity."""

    _attr_name = None

    def __init__(self, entry: LocalAiConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the AI Task entity."""
        ai_task.AITaskEntity.__init__(self)
        LocalAiEntity.__init__(self, entry, subentry)

        features = ai_task.AITaskEntityFeature(0)
        if subentry.data.get(LocalAiConfigKey.SUPPORT_ATTACHMENTS, True):
            features |= ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
        if subentry.data.get(LocalAiConfigKey.GENERATE_DATA, True):
            features |= ai_task.AITaskEntityFeature.GENERATE_DATA
        if subentry.data.get(LocalAiConfigKey.GENERATE_IMAGE, True):
            features |= ai_task.AITaskEntityFeature.GENERATE_IMAGE
        self._attr_supported_features = features

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        structure_name = task.name.strip() if task.name else None
        await self._async_handle_chat_log(chat_log, structure_name, task.structure)

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise HomeAssistantError("Last content in chat log is not an AssistantContent")

        text = (chat_log.content[-1].content or "").strip()
        LOGGER.debug("Raw text content from LLM for GenDataTask: %s", text)

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )
        try:
            data = orjson.loads(text)
            LOGGER.debug("Structured data from LLM for GenDataTask: %s", data)
        except orjson.JSONDecodeError as err:
            LOGGER.error("Failed to parse structured response from LLM: %s", err)
            raise HomeAssistantError("Error with structured response") from err

        if isinstance(data, dict | list):
            data = _clean_json_data(data)

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )

    async def _async_generate_image(
        self,
        task: ai_task.GenImageTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenImageTaskResult:
        """Handle a generate image task."""
        structure_name = task.name.strip() if task.name else None
        await self._async_handle_chat_log(chat_log, structure_name, force_image=True)

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise HomeAssistantError("Last content in chat log is not an AssistantContent")

        image_call: ImageGenerationCall | None = None
        for content in reversed(chat_log.content):
            if not isinstance(content, conversation.AssistantContent):
                break
            native = getattr(content, "native", None)
            if isinstance(native, ImageGenerationCall) and native.result:
                image_call = native
                LOGGER.debug("ImageGenerationCall object: %s", image_call)
                break

        if image_call is None or image_call.result is None:
            raise HomeAssistantError("No image returned")

        try:
            image_data = base64.b64decode(image_call.result)
        except (binascii.Error, ValueError) as err:
            LOGGER.error("Failed to decode base64 image data: %s", err)
            raise HomeAssistantError("Invalid image response data") from err

        image_call.result = None

        output_format = getattr(image_call, "output_format", None)
        mime_type = f"image/{output_format}" if output_format else "image/png"

        width: int | None = None
        height: int | None = None
        size = getattr(image_call, "size", None)
        if size:
            try:
                width_str, height_str = str(size).split("x")
                width = int(width_str)
                height = int(height_str)
            except (ValueError, AttributeError):
                width = height = None

        revised_prompt = getattr(image_call, "revised_prompt", None)
        if isinstance(revised_prompt, str):
            revised_prompt = revised_prompt.strip()

        LOGGER.debug(
            "Generated image details: mime_type=%s, width=%s, height=%s, revised_prompt=%s",
            mime_type,
            width,
            height,
            revised_prompt,
        )

        return ai_task.GenImageTaskResult(
            image_data=image_data,
            conversation_id=chat_log.conversation_id,
            mime_type=mime_type,
            width=width,
            height=height,
            model=self.model,
            revised_prompt=revised_prompt,
        )
