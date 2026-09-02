"""The Local OpenAI LLM integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryError, ConfigEntryNotReady
from homeassistant.helpers.httpx_client import get_async_client
from openai import AsyncOpenAI, AuthenticationError, OpenAIError

from .const import LOGGER, TIMEOUT, LocalAiConfigKey

PLATFORMS = [Platform.AI_TASK, Platform.CONVERSATION]

type LocalAiConfigEntry = ConfigEntry[AsyncOpenAI]


async def async_setup_entry(hass: HomeAssistant, entry: LocalAiConfigEntry) -> bool:
    """Set up Local OpenAI LLM from a config entry."""
    LOGGER.debug(
        "Creating AsyncOpenAI client for base_url: %s", entry.data[LocalAiConfigKey.BASE_URL]
    )
    client = AsyncOpenAI(
        base_url=entry.data[LocalAiConfigKey.BASE_URL],
        api_key=entry.data.get(CONF_API_KEY, ""),
        http_client=get_async_client(hass),
    )

    _ = await hass.loop.run_in_executor(None, client.platform_headers)

    try:
        LOGGER.debug("Verifying connection by listing models...")
        async for model in client.with_options(timeout=TIMEOUT).models.list():
            LOGGER.debug("Successfully connected. Found at least one model: %s", model.id)
            break
    except AuthenticationError as err:
        LOGGER.error("Invalid API key: %s", err)
        raise ConfigEntryError("Invalid API key") from err
    except OpenAIError as err:
        LOGGER.warning("Connection to API failed, will retry: %s", err)
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    return True


async def _async_update_listener(hass: HomeAssistant, entry: LocalAiConfigEntry) -> None:
    """Handle update to config entry options."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: LocalAiConfigEntry) -> bool:
    """Unload Local OpenAI LLM entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
