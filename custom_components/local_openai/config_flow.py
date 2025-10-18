"""Config flow for Local OpenAI API integration."""

from __future__ import annotations

import logging
from typing import Any
from openai import AsyncOpenAI, OpenAIError

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_MODEL, CONF_PROMPT
from homeassistant.core import callback
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
)

from .const import DOMAIN, RECOMMENDED_CONVERSATION_OPTIONS, CONF_BASE_URL, CONF_STRIP_EMOJIS, CONF_MANUAL_PROMPTING, CONF_MAX_MESSAGE_HISTORY, CONF_TEMPERATURE

_LOGGER = logging.getLogger(__name__)


class LocalAiConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Local OpenAI API."""

    VERSION = 1

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this handler."""
        return {
            "conversation": ConversationFlowHandler,
            "ai_task_data": AITaskDataFlowHandler,
        }

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors = {}
        if user_input is not None:
            self._async_abort_entries_match(user_input)
            client = AsyncOpenAI(
                base_url=user_input.get(CONF_BASE_URL),
                api_key=user_input.get(CONF_API_KEY, ""),
            )

            try:
                await client.models.list()
            except OpenAIError:
                errors["base"] = "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                return self.async_create_entry(
                    title=f"{user_input.get(CONF_MODEL, "Local")} AI Agent",
                    data=user_input,
                )
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_BASE_URL): str,
                    vol.Optional(CONF_API_KEY): str,
                }
            ),
            errors=errors,
        )


class LocalAiSubentryFlowHandler(ConfigSubentryFlow):
    """Handle subentry flow for Local OpenAI API."""


class ConversationFlowHandler(LocalAiSubentryFlowHandler):
    """Handle subentry flow."""

    async def get_schema(self, options: dict = {}):
        hass_apis: list[SelectOptionDict] = [
            SelectOptionDict(
                label=api.name,
                value=api.id,
            )
            for api in llm.async_get_apis(self.hass)
        ]

        client = self._get_entry().runtime_data
        response = await client.models.list()
        downloaded_models: list[SelectOptionDict] = [
            SelectOptionDict(
                label=model.id,
                value=model.id,
            )
            for model in response.data
        ]

        return vol.Schema(
            {
                vol.Required(
                    CONF_MODEL,
                    default=options.get(CONF_MODEL),
                ): SelectSelector(
                    SelectSelectorConfig(options=downloaded_models, custom_value=True)
                ),
                vol.Optional(
                    CONF_PROMPT,
                    default=options.get(CONF_PROMPT, RECOMMENDED_CONVERSATION_OPTIONS[CONF_PROMPT]),
                ): TemplateSelector(),
                vol.Optional(
                    CONF_LLM_HASS_API,
                    default=options.get(CONF_LLM_HASS_API, RECOMMENDED_CONVERSATION_OPTIONS[CONF_LLM_HASS_API]),
                ): SelectSelector(
                    SelectSelectorConfig(options=hass_apis, multiple=True)
                ),
                vol.Optional(
                    CONF_STRIP_EMOJIS,
                    default=options.get(CONF_STRIP_EMOJIS, False),
                ): bool,
                vol.Optional(
                    CONF_MANUAL_PROMPTING,
                    default=options.get(CONF_MANUAL_PROMPTING, False),
                ): bool,
                vol.Optional(
                    CONF_TEMPERATURE,
                    default=options.get(CONF_TEMPERATURE, 0.6),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=0, max=1, step=0.01, mode=NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(
                    CONF_MAX_MESSAGE_HISTORY,
                    default=options.get(CONF_MAX_MESSAGE_HISTORY, 0),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=0,
                        max=50,
                        step=1,
                        mode=NumberSelectorMode.BOX,
                    )
                )
            }
        )

    async def async_step_user(
            self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """User flow to create a sensor subentry."""
        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            return self.async_create_entry(
                title=f"{user_input.get(CONF_MODEL, "Local")} AI Agent", data=user_input
            )

        return self.async_show_form(
            step_id="user",
            data_schema=await self.get_schema(),
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """User flow to create a sensor subentry."""
        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=user_input,
            )

        options = self._get_reconfigure_subentry().data.copy()

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=await self.get_schema(options),
        )


class AITaskDataFlowHandler(LocalAiSubentryFlowHandler):
    """Handle subentry flow."""

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """User flow to create a sensor subentry."""
        if user_input is not None:
            return self.async_create_entry(
                title="Local AI Task", data=user_input
            )
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_MODEL): str
                }
            ),
        )
