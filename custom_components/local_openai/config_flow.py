"""Config flow for Local OpenAI LLM integration."""

from __future__ import annotations

import logging
import re
from typing import Any

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
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
)
from openai import AsyncOpenAI, OpenAIError

from .const import (
    CONF_BASE_URL,
    CONF_MANUAL_PROMPTING,
    CONF_MAX_MESSAGE_HISTORY,
    CONF_PARALLEL_TOOL_CALLS,
    CONF_SERVER_NAME,
    CONF_STRIP_EMOJIS,
    CONF_TEMPERATURE,
    DOMAIN,
    RECOMMENDED_CONVERSATION_OPTIONS,
)

_LOGGER = logging.getLogger(__name__)


class LocalAiConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Local OpenAI LLM."""

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
            _LOGGER.debug(
                f"Initialising OpenAI client with base_url: {user_input[CONF_BASE_URL]}"
            )

            try:
                client = AsyncOpenAI(
                    base_url=user_input.get(CONF_BASE_URL),
                    api_key=user_input.get(CONF_API_KEY, ""),
                    http_client=get_async_client(self.hass),
                )

                _LOGGER.debug("Retrieving model list to ensure server is accessible")
                await client.models.list()
            except OpenAIError as err:
                _LOGGER.exception(f"OpenAI Error: {err}")
                errors["base"] = "cannot_connect"
            except Exception as err:
                _LOGGER.exception(f"Unexpected exception: {err}")
                errors["base"] = "unknown"
            else:
                _LOGGER.debug("Server connection verified")
                return self.async_create_entry(
                    title=f"{user_input.get(CONF_SERVER_NAME, 'Local LLM Server')}",
                    data=user_input,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_SERVER_NAME, default="Local LLM Server"): str,
                    vol.Required(CONF_BASE_URL): str,
                    vol.Optional(CONF_API_KEY): str,
                }
            ),
            errors=errors,
        )


class LocalAiSubentryFlowHandler(ConfigSubentryFlow):
    """Handle subentry flow for Local OpenAI LLM."""

    @staticmethod
    def strip_model_pathing(model_name: str) -> str:
        """llama.cpp at the very least will keep the full model file path supplied from the CLI so lets look to strip that and any .gguf extension."""
        matches = re.search(r"([^\/]*)\.gguf$", model_name.strip())
        return matches[1] if matches else model_name


class ConversationFlowHandler(LocalAiSubentryFlowHandler):
    """Handle subentry flow."""

    def get_llm_apis(self) -> list[SelectOptionDict]:
        return [
            SelectOptionDict(
                label=api.name,
                value=api.id,
            )
            for api in llm.async_get_apis(self.hass)
        ]

    async def get_schema(self, options: dict = {}):
        llm_apis = self.get_llm_apis()
        client = self._get_entry().runtime_data

        try:
            response = await client.models.list()
            downloaded_models: list[SelectOptionDict] = [
                SelectOptionDict(
                    label=model.id,
                    value=model.id,
                )
                for model in response.data
            ]
        except OpenAIError as err:
            _LOGGER.exception(f"OpenAI Error retrieving models list: {err}")
            downloaded_models = []
        except Exception as err:
            _LOGGER.exception(f"Unexpected exception retrieving models list: {err}")
            downloaded_models = []

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
                    default=options.get(
                        CONF_PROMPT, RECOMMENDED_CONVERSATION_OPTIONS[CONF_PROMPT]
                    ),
                ): TemplateSelector(),
                vol.Optional(
                    CONF_LLM_HASS_API,
                    default=options.get(
                        CONF_LLM_HASS_API,
                        RECOMMENDED_CONVERSATION_OPTIONS[CONF_LLM_HASS_API],
                    ),
                ): SelectSelector(
                    SelectSelectorConfig(options=llm_apis, multiple=True)
                ),
                vol.Optional(
                    CONF_PARALLEL_TOOL_CALLS,
                    default=options.get(CONF_PARALLEL_TOOL_CALLS, True),
                ): bool,
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
                ),
            }
        )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """User flow to create a sensor subentry."""
        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            model_name = self.strip_model_pathing(user_input.get(CONF_MODEL, "Local"))
            return self.async_create_entry(
                title=f"{model_name} AI Agent", data=user_input
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

        # Filter out any tool providers that no longer exist
        llm_apis = [api.id for api in llm.async_get_apis(self.hass)]

        options[CONF_LLM_HASS_API] = [
            api for api in options.get(CONF_LLM_HASS_API, []) if api in llm_apis
        ]

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
            model_name = self.strip_model_pathing(user_input.get(CONF_MODEL, "Local"))
            return self.async_create_entry(
                title=f"{model_name} AI Task", data=user_input
            )

        try:
            client = self._get_entry().runtime_data
            response = await client.models.list()
            downloaded_models: list[SelectOptionDict] = [
                SelectOptionDict(
                    label=model.id,
                    value=model.id,
                )
                for model in response.data
            ]
        except OpenAIError as err:
            _LOGGER.exception(f"OpenAI Error retrieving models list: {err}")
            downloaded_models = []
        except Exception as err:
            _LOGGER.exception(f"Unexpected exception retrieving models list: {err}")
            downloaded_models = []

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_MODEL,
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=downloaded_models, custom_value=True
                        )
                    ),
                }
            ),
        )
