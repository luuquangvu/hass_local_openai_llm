# Local OpenAI LLM _(Home Assistant custom integration)_

This repository tracks the upstream [Local OpenAI LLM](https://github.com/skye-harris/hass_local_openai_llm) project by [@skye-harris](https://github.com/skye-harris). All core capabilities are kept in sync, while this fork layers in a streamlined path for running Google's Gemini models via [Gemini-FastAPI](https://github.com/Nativu5/Gemini-FastAPI).

---

## Highlights

### Upstream feature parity

- OpenAI-compatible endpoint support with streamed responses, conversation history trimming, and manual prompt control.
- Works with Home Assistant Assist features including tool calling, image inputs for AI tasks, temperature tuning, and emoji stripping.
- Conversation agents retain TTS streaming and reconfiguration options from the original integration.

### Gemini-FastAPI enhancements

- Treat Gemini-FastAPI as a drop-in OpenAI API, exposing Google Gemini models to Home Assistant without a Google API key.
- Send multimodal prompts (text, files, images, audio, video, PDF) and receive Gemini-generated text replies.
- Request image generation through the same integration for richer automations.
- Optional Markdown emphasis cleanup to improve announcement-quality responses.

---

## Prerequisites

- A working Home Assistant installation with [HACS](https://hacs.xyz/) available for custom integrations.
- An OpenAI-compatible endpoint: Gemini-FastAPI (proxying Google Gemini through cookies) or any backend supported upstream.
- For Gemini-FastAPI, a Google account with Gemini web access and extracted `__Secure-1PSID` and `__Secure-1PSIDTS` cookie values.

---

## Installation

### Option 1 - HACS (recommended)

[![Add Local OpenAI LLM to HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=luuquangvu&repository=hass_local_openai_llm&category=integration)

1. Open HACS and choose `Integrations`.
2. Search for **Local OpenAI LLM**. If it is not listed, add `https://github.com/luuquangvu/hass_local_openai_llm` as a custom repository of type `Integration`, then refresh the list.
3. Install the integration and restart Home Assistant when prompted.

### Option 2 - Manual copy

1. Download the latest release archive from this repository.
2. Copy the `custom_components/local_openai_llm` folder into the `custom_components` directory in your Home Assistant configuration.
3. Restart Home Assistant.

---

## Configuration

1. Navigate to `Settings` → `Devices & Services` in Home Assistant.
2. Click `Add Integration` and search for **Local OpenAI LLM**.
3. Enter the connection details for your local LLM endpoint, including the Server URL, and optionally the API key and Server Name.
   - Server URL: http://host:port/v1 (for example, http://127.0.0.1:8000/v1)
4. Complete the setup wizard to expose the conversation and image generation services.

---

## Acknowledgements

- Built on top of the excellent [Local OpenAI LLM](https://github.com/skye-harris/hass_local_openai_llm) project by [@skye-harris](https://github.com/skye-harris); this fork mirrors upstream improvements and releases.
- Extra thanks to the [Gemini-FastAPI](https://github.com/Nativu5/Gemini-FastAPI) community for providing an accessible FastAPI wrapper around Gemini services.
