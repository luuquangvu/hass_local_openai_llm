# Local OpenAI LLM <small>_(Custom Integration for Home Assistant)_</small>

**Allows use of generic OpenAI-compatible LLM services, such as (but not limited to):**
- llama.cpp
- vLLM
- LM Studio

**This integration has been forked from Home Assistants OpenRouter integration, with the following changes:**
- Added server URL to the initial server configuration
- Made the API Key optional during initial server configuration: can be left blank if your local server does not require one
- Uses streamed LLM responses
- Conversation Agents support TTS streaming
- Automatically strips `<think>` tags from responses
- Added support for image inputs for AI Task Agents
- Added support for reconfiguring Conversation Agents
- Added option to trim conversation history to help stay within your context window
- Added temperature control
- Added option to strip emojis from responses
- Added option to take full manual control of the prompt
  - This will remove ALL content that Home Assistant normally inserts when compiling the system prompt that's sent to the LLM
  - Additional variables are exposed to the prompt jinja template for tools, entities, voice-satellite area, etc
  - **For advanced use only: not recommended for most users, and not yet documented here**

---

## Installation

### Install via HACS (recommended)

Have [HACS](https://hacs.xyz/) installed, this will allow you to update easily.

Adding Tools for Assist to HACS can be using this button:  
  [![image](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=skye-harris&repository=hass_local_openai_llm&category=integration)

<br>

> [!NOTE]
> If the button above doesn't work, add `https://github.com/skye-harris/hass_local_openai_llm` as a custom repository of type Integration in HACS.

* Click install on the `Local OpenAI LLM` integration.
* Restart Home Assistant.

<details><summary>Manual Install</summary>

* Copy the `local_openai`  folder from [latest release](https://github.com/skye-harris/hass_local_openai_llm/releases/latest) to the [
  `custom_components` folder](https://developers.home-assistant.io/docs/creating_integration_file_structure/#where-home-assistant-looks-for-integrations) in your config directory.
* Restart the Home Assistant.

</details>

## Integration Configuration

After installation, configure the integration through Home Assistant's UI:

1. Go to `Settings` → `Devices & Services`.
2. Click `Add Integration`.
3. Search for `Local OpenAI LLM`.
4. Follow the setup wizard to configure your desired services.

### Configuration Notes

- The Server URL must be a fully qualified URL pointing to an OpenAI-compatible API.
  - This typically ends with `/v1` but may differ depending on your server configuration. 
- If you have the `Extended OpenAI Conversation` integration installed, this has a dependency of an older version of the OpenAI client library.
  - It is strongly recommended this be uninstalled to ensure that HACS installs the correct OpenAI client library.
- Assist requires a fairly lengthy context for tooling and entity definitions. 
  - It is strongly recommended to use _at least_ 8k context size and to limit history length to avoid context overflow issues.
  - This is not configurable through OpenAI-compatible APIs, and needs to be configured with the inference server directly.
  

## Additional

Looking to add some more functionality to your Home Assistant conversation agent, such as web and localised business/location search? Check out my [Tools for Assist](https://github.com/skye-harris/llm_intents) integration here!

## Acknowledgements

- This integration is forked from the [OpenRouter](https://github.com/home-assistant/core/tree/dev/homeassistant/components/open_router) integration for Home Assistant by [@joostlek](https://github.com/joostlek)
- This integration uses code from the [Local LLMs](https://github.com/acon96/home-llm) integration for Home Assistant by [@acon96](https://github.com/acon96/home-llm) 
