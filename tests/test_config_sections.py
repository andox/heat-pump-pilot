from runtime_settings import build_runtime_settings
"""Exercise real form builders with HA selectors replaced by schema validators.

Home Assistant itself is unavailable in the Windows development environment.
Voluptuous validates defaults/nesting; small stand-ins cover HA's flow plumbing.
"""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

vol = pytest.importorskip("voluptuous")
from config_sections import SECTIONS, flatten_input, group_for

ROOT = Path(__file__).resolve().parents[1] / "custom_components/heat_pump_pilot"


class Flow:
    def __init_subclass__(cls, **kwargs):
        pass

    def async_create_entry(self, **kwargs):
        return kwargs

    def async_show_form(self, **kwargs):
        return kwargs


def load_flow(flow_name="OptionsFlowHandler"):
    tree = ast.parse((ROOT / "config_flow.py").read_text(encoding="utf-8"))
    tree.body = [
        n
        for n in tree.body
        if not (isinstance(n, ast.ImportFrom) and n.module.startswith("homeassistant"))
    ]
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            node.level = 0

    class Selectors:
        NumberSelectorMode = SimpleNamespace(BOX="box", SLIDER="slider")
        SelectSelectorMode = SimpleNamespace(DROPDOWN="dropdown")

        def __getattr__(self, name):
            if name.endswith("Config"):
                return lambda **kwargs: kwargs
            if name == "BooleanSelector":
                return lambda: bool
            if name == "TimeSelector":
                return lambda: str
            if name == "EntitySelector":
                return lambda config: [str] if config.get("multiple") else str
            if name == "TextSelector":
                return lambda config: str
            if name == "NumberSelector":
                return lambda config: vol.All(
                    vol.Coerce(float), vol.Range(min=config["min"], max=config["max"])
                )
            if name == "SelectSelector":
                return lambda config: vol.In(config["options"])
            raise AttributeError(name)

    def section(schema, options):
        schema.section_options = options
        return schema

    namespace = dict(
        config_entries=SimpleNamespace(ConfigFlow=Flow, OptionsFlow=Flow),
        callback=lambda f: f,
        section=section,
        selector=Selectors(),
    )
    exec(compile(tree, str(ROOT / "config_flow.py"), "exec"), namespace)
    return namespace[flow_name]


def initial_form_data(schema):
    """Mirror HA frontend initialization, before server-side validation.

    compute-initial-ha-form-data.ts gives a group default precedence over its
    children's defaults. Calling schema({}) instead hides an empty-group bug
    because Voluptuous fills missing child fields during validation.
    """
    data = {}
    for marker, validator in schema.schema.items():
        if marker.default is not vol.UNDEFINED:
            data[marker.schema] = marker.default()
        elif isinstance(validator, vol.Schema):
            data[marker.schema] = initial_form_data(validator)
    return data


def entry():
    return SimpleNamespace(
        entry_id="pilot",
        options={"learning_model": "ekf", "learning_window_hours": 24},
        data={
            "indoor_temp_entity": "sensor.indoor",
            "outdoor_temp_entity": "sensor.outdoor",
            "price_entity": "sensor.price",
            "weather_forecast_entity": "weather.home",
            "controlled_entity": "number.virtual",
        },
    )


def test_grouped_schema_defaults_validate_and_have_translations():
    flow = load_flow()(entry())
    schema = flow._build_options_schema()
    submitted = initial_form_data(schema)
    schema(submitted)
    flat = flatten_input(submitted)
    assert flat["controlled_entity"] == "number.virtual"
    assert flat["learning_model"] == "ekf"
    assert flat["learning_interval_minutes"] == 60
    assert flat["heating_supply_temp_entity"] is None
    keys = [k.schema for k in schema.schema]
    first_advanced = next(i for i, k in enumerate(keys) if k.startswith("advanced_"))
    assert all(k.startswith("advanced_") for k in keys[first_advanced:])
    translations = json.loads((ROOT / "strings.json").read_text(encoding="utf-8"))[
        "options"
    ]["step"]["init"]
    for key, validator in schema.schema.items():
        if key.schema in SECTIONS:
            assert validator.section_options["collapsed"] == key.schema.startswith(
                "advanced_"
            )
            for field in validator.schema:
                assert field.schema in translations["sections"][key.schema]["data"]
        else:
            assert key.schema in translations["data"]


def test_grouped_save_keeps_learning_options_and_sensor_selection():
    original = entry()
    flow = load_flow()(original)
    changes = []
    flow.hass = SimpleNamespace(
        config_entries=SimpleNamespace(
            async_entries=lambda domain: [],
            async_update_entry=lambda *a, **kw: changes.append(kw),
        )
    )
    inputs = initial_form_data(flow._build_options_schema())
    inputs["advanced_learning"].update(
        learning_model="adaptive",
        learning_interval_minutes=120,
        pump_response_enabled=False,
    )
    inputs["advanced_summer"].update(
        summer_heat_window_enabled=True,
        summer_heat_window_max_price=0.42,
        summer_heat_window_duration_minutes=90,
        summer_heat_window_demand_window_hours=36,
        summer_heat_window_max_heat_demand_ratio=0.12,
        summer_heat_window_virtual_heat_offset=16,
    )
    result = asyncio.run(flow.async_step_init(inputs))
    assert result["data"]["learning_model"] == "adaptive"
    assert result["data"]["learning_interval_minutes"] == 120
    assert result["data"]["pump_response_enabled"] is False
    assert result["data"]["learning_window_hours"] == "24"
    for key, value in inputs["advanced_summer"].items():
        assert result["data"][key] == value
    runtime = build_runtime_settings(result["data"])
    assert runtime.summer_heat_window_enabled is True
    assert runtime.summer_heat_window_virtual_heat_offset == 16
    assert runtime.learning_model == "adaptive"
    assert changes[0]["data"]["controlled_entity"] == "number.virtual"


def test_omitted_sections_preserve_values_and_explicit_clearing_works():
    existing = {
        "pump_response_enabled": False,
        "learning_interval_minutes": 120,
        "controlled_entity": "number.virtual",
    }
    result = flatten_input({"target_temperature": 22}, existing)
    assert result["learning_interval_minutes"] == 120
    assert result["controlled_entity"] == "number.virtual"
    assert flatten_input({"connections": {}}, existing)["controlled_entity"] is None
    assert group_for("learning_supply_temp_on_margin") == "advanced_detection"


def test_error_redisplay_retains_submitted_choices():
    flow = load_flow()(entry())
    submitted = initial_form_data(
        flow._build_options_schema(
            {"learning_model": "adaptive", "indoor_temp_entity": "sensor.changed"}
        )
    )
    assert submitted["connections"]["indoor_temp_entity"] == "sensor.changed"
    assert submitted["advanced_learning"]["learning_model"] == "adaptive"


def test_saved_dropdowns_numbers_and_disabled_options_are_prefilled():
    configured = entry()
    configured.options.update(
        {
            "learning_model": "rls",
            "price_baseline_window_hours": 48,
            "price_absolute_low_window_days": 14,
            "price_penalty_curve": "quadratic",
            "price_absolute_low_threshold": "off",
            "rls_forgetting_factor": 0.97,
            "pump_response_enabled": False,
            "virtual_outdoor_smoothing_alpha": 0.0,
            "heating_supply_temp_entity": "sensor.flow",
        }
    )
    schema = load_flow()(configured)._build_options_schema()
    visible = initial_form_data(schema)
    assert visible["advanced_prices"]["price_baseline_window_hours"] == "48"
    assert visible["advanced_prices"]["price_absolute_low_window_days"] == "14"
    assert visible["advanced_prices"]["price_penalty_curve"] == "quadratic"
    assert visible["advanced_prices"]["price_absolute_low_threshold"] == "off"
    assert visible["advanced_learning"]["learning_model"] == "rls"
    assert visible["advanced_learning"]["rls_forgetting_factor"] == 0.97
    assert visible["advanced_learning"]["pump_response_enabled"] is False
    assert visible["advanced_control"]["virtual_outdoor_smoothing_alpha"] == 0.0
    assert visible["connections"]["heating_supply_temp_entity"] == "sensor.flow"
    schema(visible)


def test_initial_setup_also_prefills_grouped_defaults():
    flow = load_flow("ConfigFlow")()
    result = asyncio.run(flow.async_step_user())
    visible = initial_form_data(result["data_schema"])
    assert visible["advanced_learning"]["learning_model"] == "adaptive"
    assert visible["advanced_learning"]["rls_forgetting_factor"] == 0.99


def test_ufh_settings_save_multiple_switches_independently_of_summer():
    flow=load_flow()(entry())
    flow.hass=SimpleNamespace(config_entries=SimpleNamespace(async_entries=lambda domain:[],async_update_entry=lambda *a,**k:None))
    inputs=initial_form_data(flow._build_options_schema())
    inputs['advanced_ufh'].update(ufh_enabled=True,ufh_supply_entity='sensor.supply',ufh_switches=['switch.a','switch.b'])
    inputs['advanced_ufh_exercise'].update(ufh_exercise_enabled=True,ufh_exercise_time='12:30:00')
    inputs['advanced_summer']['summer_heat_window_enabled']=False
    result=asyncio.run(flow.async_step_init(inputs))
    assert result['data']['ufh_switches']==['switch.a','switch.b']
    assert result['data']['ufh_exercise_enabled'] is True
    assert result['data']['summer_heat_window_enabled'] is False
    assert build_runtime_settings(result['data']).learning_model=='ekf'


def test_ufh_missing_entities_and_reversed_thresholds_rejected():
    flow=load_flow()(entry())
    flow.hass=SimpleNamespace(config_entries=SimpleNamespace(async_entries=lambda domain:[]))
    inputs=initial_form_data(flow._build_options_schema())
    inputs['advanced_ufh']['ufh_enabled']=True
    result=asyncio.run(flow.async_step_init(inputs))
    assert result['errors']['base']=='ufh_missing_entities'
    inputs['advanced_ufh'].update(ufh_supply_entity='sensor.supply',ufh_switches=['switch.a'],ufh_on_temperature=20,ufh_off_temperature=25)
    result=asyncio.run(flow.async_step_init(inputs))
    assert result['errors']['base']=='ufh_invalid_thresholds'
