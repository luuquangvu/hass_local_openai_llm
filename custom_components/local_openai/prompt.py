import webcolors
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
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


CSS3_NAME_TO_RGB = {
    name: webcolors.name_to_rgb(name, CSS3) for name in webcolors.names(CSS3)
}


def closest_color(requested_color):
    """
    Lovingly borrowed from https://github.com/acon96/home-llm.
    """
    min_colors = {}

    for name, rgb in CSS3_NAME_TO_RGB.items():
        r_c, g_c, b_c = rgb
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def async_get_entities(hass) -> list:
    """
    Lovingly borrowed from https://github.com/acon96/home-llm (_generate_system_prompt).
    """
    extra_attributes_to_expose = DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE

    def expose_attributes(attributes) -> list[str]:
        result = []
        for attribute_name in extra_attributes_to_expose:
            if attribute_name not in attributes:
                continue

            value = attributes[attribute_name]
            if value is not None:
                # try to apply unit if present
                unit_suffix = attributes.get(f"{attribute_name}_unit")
                if unit_suffix:
                    value = f"{value} {unit_suffix}"
                elif attribute_name == "temperature":
                    # try to get unit or guess otherwise
                    suffix = "F" if value > 50 else "C"
                    value = f"{int(value)} {suffix}"
                elif attribute_name == "rgb_color":
                    value = f"{closest_color(value)} {value}"
                elif attribute_name == "volume_level":
                    value = f"vol={int(value * 100)}"
                elif attribute_name == "brightness":
                    value = f"{int(value / 255 * 100)}%"
                elif attribute_name == "humidity":
                    value = f"{value}%"

                result.append(str(value))
        return result

    entities_to_expose, domains = async_get_exposed_entities(hass)
    devices = []
    formatted_devices = ""

    # expose devices and their alias as well
    for name, attributes in entities_to_expose.items():
        state = attributes["state"]
        exposed_attributes = expose_attributes(attributes)
        str_attributes = ";".join([state] + exposed_attributes)

        formatted_devices = (
            formatted_devices
            + f"{name} '{attributes.get('friendly_name')}' = {str_attributes}\n"
        )
        device_attribs = {
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


def async_get_exposed_entities(hass) -> tuple[dict[str, str], list[str]]:
    """
    Gather exposed entity states.

    Lovingly borrowed from https://github.com/acon96/home-llm.
    """
    entity_states = {}
    domains = set()
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

        attributes = dict(state.attributes)
        attributes["state"] = state.state

        if entity:
            if entity.aliases:
                attributes["aliases"] = entity.aliases

            if entity.unit_of_measurement:
                attributes["state"] = (
                    attributes["state"] + " " + entity.unit_of_measurement
                )

        # area could be on device or entity. prefer device area
        area_id = None
        if device and device.area_id:
            area_id = device.area_id
        if entity and entity.area_id:
            area_id = entity.area_id

        if area_id:
            area = area_registry.async_get_area(entity.area_id)
            if area:
                attributes["area_id"] = area.id
                attributes["area_name"] = area.name

        entity_states[state.entity_id] = attributes
        domains.add(state.domain)

    return entity_states, list(domains)


def format_custom_prompt(hass, agent_prompt: str, user_input, tools):
    devices = async_get_entities(hass)
    LOGGER.debug("Exposed devices for prompt: %s", devices)

    area: ar.AreaEntry | None = None
    floor: fr.FloorEntry | None = None
    device_name = None
    if user_input.device_id:
        device_reg = dr.async_get(hass)
        device = device_reg.async_get(user_input.device_id)

        if device:
            device_name = device.name
            area_reg = ar.async_get(hass)
            if device.area_id and (area := area_reg.async_get_area(device.area_id)):
                floor_reg = fr.async_get(hass)
                if area.floor_id:
                    floor = floor_reg.async_get_floor(area.floor_id)

    LOGGER.debug(
        "Context for prompt: area=%s, floor=%s, device_name=%s",
        area,
        floor,
        device_name,
    )

    # Render prompt
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
    LOGGER.debug("Final rendered manual prompt: %s", rendered_prompt)
    return rendered_prompt
