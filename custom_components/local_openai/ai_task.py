"""AI Task integration for Local OpenAI LLM."""

from __future__ import annotations

from json import JSONDecodeError
import base64
import binascii

from openai.types.responses.response_output_item import ImageGenerationCall

from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from . import LocalAiConfigEntry
from .const import GEMINI_MODELS, LOGGER
from .entity import LocalAiEntity


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: LocalAiConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "ai_task_data":
            continue

        async_add_entities(
            [LocalAITaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class LocalAITaskEntity(
    ai_task.AITaskEntity,
    LocalAiEntity,
):
    """Local OpenAI LLM AI Task entity."""

    _attr_name = None

    def __init__(self, entry: LocalAiConfigEntry, subentry) -> None:
        """Initialize the AI Task entity."""
        ai_task.AITaskEntity.__init__(self)
        LocalAiEntity.__init__(self, entry, subentry)

        features = (
            ai_task.AITaskEntityFeature.GENERATE_DATA
            | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
        )
        model_name = self.model.lower()
        if any(model_id in model_name for model_id in GEMINI_MODELS):
            features |= ai_task.AITaskEntityFeature.GENERATE_IMAGE
        self._attr_supported_features = features

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        await self._async_handle_chat_log(chat_log, task.name, task.structure)

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise HomeAssistantError(
                "Last content in chat log is not an AssistantContent"
            )

        text = chat_log.content[-1].content or ""
        LOGGER.debug("Raw text content from LLM for GenDataTask: %s", text)

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )
        try:
            data = json_loads(text)
            LOGGER.debug("Structured data from LLM for GenDataTask: %s", data)
        except JSONDecodeError as err:
            LOGGER.error("Failed to parse structured response from LLM: %s", err)
            raise HomeAssistantError("Error with structured response") from err

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
        await self._async_handle_chat_log(chat_log, task.name, force_image=True)

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise HomeAssistantError(
                "Last content in chat log is not an AssistantContent"
            )

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
