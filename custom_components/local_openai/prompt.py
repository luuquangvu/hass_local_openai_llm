"""Prompt building and formatting utilities for Local OpenAI LLM."""

from __future__ import annotations

import webcolors
from homeassistant.components import conversation
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import (
    area_registry as ar,
)
from homeassistant.helpers import (
    device_registry as dr,
)
from homeassistant.helpers import (
    entity_registry as er,
)
from homeassistant.helpers import (
    floor_registry as fr,
)
from homeassistant.helpers import (
    template,
)
from webcolors import CSS3

from .const import LOGGER

DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE = [
    "rgb_color",
    "brightness",
    "temperature",
    "humidity",
    "fan_mode",
    "hvac_mode",
    "media_title",
    "volume_level",
    "item",
    "wind_speed",
]


CSS3_NAME_TO_RGB = {name: webcolors.name_to_rgb(name, CSS3) for name in webcolors.names(CSS3)}


def closest_color(requested_color: tuple[int, int, int]) -> str:
    """Find the closest CSS3 color name for an RGB tuple."""
    min_colors = {}

    for name, rgb in CSS3_NAME_TO_RGB.items():
        r_c, g_c, b_c = rgb
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def get_entities(hass: HomeAssistant) -> list[dict[str, object]]:
    """Gather exposed entities and their formatted states."""
    extra_attributes_to_expose = DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE

    def expose_attributes(_attributes: dict[str, object]) -> list[str]:
        result: list[str] = []
        for attribute_name in extra_attributes_to_expose:
            if attribute_name not in _attributes:
                continue

            value = _attributes[attribute_name]
            if value is not None:
                unit_suffix = _attributes.get(f"{attribute_name}_unit")
                if unit_suffix:
                    value = f"{value} {unit_suffix}"
                elif attribute_name == "temperature" and isinstance(value, int | float):
                    suffix = (
                        _attributes.get("unit_of_measurement") or hass.config.units.temperature_unit
                    )
                    formatted_temp = (
                        f"{value:.1f}".rstrip("0").rstrip(".")
                        if isinstance(value, float)
                        else str(value)
                    )
                    value = f"{formatted_temp} {suffix}"
                elif (
                    attribute_name == "rgb_color"
                    and isinstance(value, tuple | list)
                    and len(value) == 3
                ):
                    color_tuple = (int(value[0]), int(value[1]), int(value[2]))
                    value = f"{closest_color(color_tuple)} {value}"
                elif attribute_name == "volume_level" and isinstance(value, int | float):
                    value = f"vol={int(value * 100)}"
                elif attribute_name == "brightness" and isinstance(value, int | float):
                    value = f"{int(value / 255 * 100)}%"
                elif attribute_name == "humidity":
                    value = f"{value}%"

                result.append(str(value))
        return result

    entities_to_expose = get_exposed_entities(hass)
    devices: list[dict[str, object]] = []

    for name, attributes in entities_to_expose.items():
        state = str(attributes["state"])
        exposed_attributes = expose_attributes(attributes)

        device_attribs: dict[str, object] = {
            "entity_id": name,
            "name": attributes.get("friendly_name"),
            "state": state,
            "attributes": exposed_attributes,
            "area_name": attributes.get("area_name"),
            "area_id": attributes.get("area_id"),
            "is_alias": False,
        }

        if "aliases" in attributes:
            device_attribs["aliases"] = attributes["aliases"]

        devices.append(device_attribs)

    return devices


def get_exposed_entities(hass: HomeAssistant) -> dict[str, dict[str, object]]:
    """Gather exposed Home Assistant entities and their domain states."""
    entity_states: dict[str, dict[str, object]] = {}
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)
    area_registry = ar.async_get(hass)

    for state in hass.states.async_all():
        if not async_should_expose(hass, CONVERSATION_DOMAIN, state.entity_id):
            continue

        entity = entity_registry.async_get(state.entity_id)
        device = None
        if entity and entity.device_id:
            device = device_registry.async_get(entity.device_id)

        attributes: dict[str, object] = dict(state.attributes)
        attributes["state"] = state.state

        if entity:
            if entity.aliases:
                attributes["aliases"] = entity.aliases

            if entity.unit_of_measurement:
                attributes["state"] = f"{attributes['state']} {entity.unit_of_measurement}"

        area_id = None
        if device and device.area_id:
            area_id = device.area_id
        if entity and entity.area_id:
            area_id = entity.area_id

        if area_id and (area := area_registry.async_get_area(area_id)):
            attributes["area_id"] = area.id
            attributes["area_name"] = area.name

        entity_states[state.entity_id] = attributes

    return entity_states


def format_custom_prompt(
    hass: HomeAssistant,
    agent_prompt: str,
    user_input: conversation.ConversationInput,
    tools: object = None,
) -> str:
    """Format and render custom prompt template with exposed devices and context."""
    devices = get_entities(hass)
    LOGGER.debug("Exposed devices for prompt: %s", devices)

    area: ar.AreaEntry | None = None
    floor: fr.FloorEntry | None = None
    device_name = None
    if user_input.device_id:
        device_reg = dr.async_get(hass)
        if device := device_reg.async_get(user_input.device_id):
            device_name = device.name
            area_reg = ar.async_get(hass)
            if device.area_id and (found_area := area_reg.async_get_area(device.area_id)):
                area = found_area
                if area.floor_id:
                    floor_reg = fr.async_get(hass)
                    floor = floor_reg.async_get_floor(area.floor_id)

    LOGGER.debug(
        "Context for prompt: area=%s, floor=%s, device_name=%s",
        area,
        floor,
        device_name,
    )

    try:
        rendered_prompt = template.Template(
            agent_prompt,
            hass,
        ).async_render(
            {
                "tools": tools,
                "devices": devices,
                "floor": floor,
                "area": area,
                "device": device_name,
                "extra_system_prompt": user_input.extra_system_prompt,
            },
            parse_result=False,
        )
    except TemplateError as err:
        LOGGER.error("Error rendering custom prompt: %s", err)
        raise HomeAssistantError(f"Error rendering custom prompt: {err}") from err
    LOGGER.debug("Final rendered manual prompt: %s", rendered_prompt)
    return str(rendered_prompt)
