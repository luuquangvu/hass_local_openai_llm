"""Constants for the Local OpenAI LLM integration."""

import logging
import re
from typing import Literal

from homeassistant.const import CONF_LLM_HASS_API, CONF_PROMPT
from homeassistant.helpers import llm

DOMAIN = "local_openai"
LOGGER = logging.getLogger(__package__)

CONF_RECOMMENDED = "recommended"
CONF_BASE_URL = "base_url"
CONF_SERVER_NAME = "server_name"
CONF_STRIP_EMOJIS = "strip_emojis"
CONF_STRIP_EMPHASIS = "strip_emphasis"
CONF_STRIP_LATEX = "strip_latex"
CONF_MANUAL_PROMPTING = "manual_prompting"
CONF_TEMPERATURE = "temperature"
CONF_PARALLEL_TOOL_CALLS = "parallel_tool_calls"

GEMINI_MODEL_PREFIXES = ("gemini-",)

RECOMMENDED_CONVERSATION_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}

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

LATEX_MATH_SPAN = re.compile(
    r"""
    \$\$[\s\S]+?\$\$
  | \$(?!\s)[^$\n]+?(?<!\s)\$
  | \\\([^)]*\\\)
  | \\\[[^]]*\\\]
    """,
    re.VERBOSE,
)
