"""Constants for the Local OpenAI LLM integration."""

import logging

from homeassistant.const import CONF_LLM_HASS_API, CONF_PROMPT
from homeassistant.helpers import llm

DOMAIN = "local_openai"
LOGGER = logging.getLogger(__package__)

CONF_RECOMMENDED = "recommended"
CONF_BASE_URL = 'base_url'
CONF_STRIP_EMOJIS = 'strip_emojis'
CONF_MANUAL_PROMPTING = 'manual_prompting'
CONF_MAX_MESSAGE_HISTORY = 'max_message_history'
CONF_TEMPERATURE = 'temperature'

RECOMMENDED_CONVERSATION_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}