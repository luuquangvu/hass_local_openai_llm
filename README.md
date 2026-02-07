**[🇻🇳 Tiếng Việt](README.vi.md)**

# Local OpenAI LLM for Home Assistant

This repository is a specialized fork of the original [Local OpenAI LLM](https://github.com/skye-harris/hass_local_openai_llm) project by [@skye-harris](https://github.com/skye-harris). It maintains core capabilities while adding powerful enhancements specifically designed for running Google's Gemini models for free via [Gemini-FastAPI](https://github.com/luuquangvu/ha-addons).

---

## Key Features

- **Extended Context Awareness**: Optimized to fully leverage the model's long-context capabilities and maximize context-caching efficiency, ensuring the agent maintains consistent memory throughout long-running conversations.
- **Multimodal Mastery**: Seamlessly send text, images, audio, video, and PDF files directly to Google Gemini for advanced analysis and reasoning.
- **Completely Free & No API Key**: Access powerful Google Gemini models for free as a drop-in OpenAI replacement. No Google Cloud project or official API key required (powered by [Gemini-FastAPI](https://github.com/luuquangvu/ha-addons)).
- **Native Home Assistant Integration**: Deeply integrated with Assist, supporting tool calling (intent handling), image inputs for AI tasks, and temperature tuning.
- **Manual Prompt Control**: Take full control of system instructions with Jinja2 template support for precise response shaping and personality.
- **Image Generation**: Integrated support for generating images directly through the conversation agent or dedicated services.
- **Announcement-Ready Output**: Built-in emoji stripping, Markdown emphasis cleanup, and LaTeX removal to ensure clear, high-quality TTS announcements.

---

## Prerequisites

- **Home Assistant** with [HACS](https://hacs.xyz/) installed.
- **Gemini-FastAPI Server**: [Available here](https://github.com/luuquangvu/ha-addons) (recommended for seamless Gemini access).
- **Gemini Credentials**: Valid `__Secure-1PSID` and `__Secure-1PSIDTS` cookies (required by your Gemini-FastAPI instance).

---

## Installation

### Option 1: HACS (Recommended)

[![Add Local OpenAI LLM to HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=luuquangvu&repository=hass_local_openai_llm&category=integration)

1. Open **HACS** and select **Integrations**.
2. Search for **Local OpenAI LLM**.
3. If not found, add `https://github.com/luuquangvu/hass_local_openai_llm` as a **Custom Repository** (Category: Integration).
4. Click **Download**, then restart Home Assistant.

### Option 2: Manual Installation

1. Download this repository as a ZIP file or clone it using Git.
2. Copy the `custom_components/local_openai` directory into your Home Assistant's `custom_components/` folder.
3. Restart Home Assistant.

---

## Configuration

1. Go to **Settings** → **Devices & Services**.
2. Click **Add Integration** and search for **Local OpenAI LLM**.
3. Provide your server details:
   - **Server Name**: A friendly name (e.g., `Gemini AI`).
   - **Server URL**: The full endpoint (e.g., `http://127.0.0.1:8000/v1`). **Note:** Must include the `/v1` suffix.
   - **API Key**: Optional (use the API key if configured in your Gemini-FastAPI `config.yaml`).
4. Follow the setup wizard to create **Conversation Agents** or **AI Tasks**.

---

## Acknowledgements

- Based on the excellent [Local OpenAI LLM](https://github.com/skye-harris/hass_local_openai_llm) project by [@skye-harris](https://github.com/skye-harris).
- Powered by [Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) & [Gemini-FastAPI](https://github.com/luuquangvu/ha-addons).
