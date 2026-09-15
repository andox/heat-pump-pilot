"""Home Assistant lifecycle and switch adapter for UFH circulation."""
from __future__ import annotations

import asyncio
from datetime import timedelta
import logging

from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_interval
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import TemperatureConverter

from .const import DOMAIN
from .json_storage import AtomicJsonStorage
from .ufh_controller import UfhController
from .ufh_settings import validate_ufh

_LOGGER = logging.getLogger(__name__)
UFH_SIGNAL = 'heat_pump_pilot_ufh_updated'


class UfhCoordinator:
    """Operate independently of the climate entity's HVAC mode and MPC cycle."""

    def __init__(self, hass, entry):
        self.hass, self.entry = hass, entry
        self.model = UfhController(entry.options)
        self.store = AtomicJsonStorage(hass.config.path('.storage', f'{DOMAIN}_{entry.entry_id}_ufh.json'))
        self._lock = asyncio.Lock()
        self._unsubs = []
        self._task = None
        self._pending = False
        self._stopped = False
        self._saved = None
        self.diagnostics = {'state': 'disabled', 'pumps': {}}

    async def async_start(self):
        payload = await self.hass.async_add_executor_job(self.store.load)
        self.model.restore(payload, dt_util.utcnow().timestamp())
        self._subscribe()
        self._stop_unsub = self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, self._on_stop)
        await self.async_refresh()

    def _subscribe(self):
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []
        c = self.model.options
        if not c['ufh_enabled'] or validate_ufh(c):
            return
        entities = list(dict.fromkeys([c['ufh_supply_entity'], *c['ufh_switches']]))
        self._unsubs = [
            async_track_state_change_event(self.hass, entities, self._changed),
            async_track_time_interval(self.hass, self._changed, timedelta(seconds=10)),
        ]

    @callback
    def _changed(self, _event):
        if self._stopped:
            return
        self._pending = True
        if self._task is None or self._task.done():
            self._task = self.hass.async_create_task(self._drain())

    async def _drain(self):
        while self._pending and not self._stopped:
            self._pending = False
            await self.async_refresh()

    async def async_update_options(self):
        # Finish any in-flight service call before changing ownership/options.
        if self._task and not self._task.done():
            await self._task
        async with self._lock:
            if self._stopped:
                return
            self.model.configure(self.entry.options)
            self._subscribe()
            await self._refresh()

    async def _save(self):
        snapshot = self.model.export_state()
        if snapshot != self._saved:
            await self.hass.async_add_executor_job(self.store.save, snapshot)
            self._saved = snapshot

    async def async_refresh(self):
        async with self._lock:
            await self._refresh()

    async def _refresh(self):
        if self._stopped:
            return
        now = dt_util.utcnow()
        c = self.model.options
        if error := validate_ufh(c):
            self.diagnostics = {'state': error, 'pumps': {}}
            async_dispatcher_send(self.hass, f'{UFH_SIGNAL}_{self.entry.entry_id}')
            return
        sensor = self.hass.states.get(c['ufh_supply_entity']) if c['ufh_supply_entity'] else None
        temperature, fresh = None, False
        if sensor is not None:
            try:
                temperature = TemperatureConverter.convert(float(sensor.state), sensor.attributes.get('unit_of_measurement', '°C'), '°C')
                reported = getattr(sensor, 'last_reported', sensor.last_updated)
                fresh = 0 <= (now - reported).total_seconds() <= float(c['ufh_stale_minutes']) * 60
                if sensor.attributes.get('restored', False):
                    fresh = False
            except (ValueError, TypeError):
                pass
        states = {}
        for entity in c['ufh_switches']:
            state = self.hass.states.get(entity)
            states[entity] = state.state if state and not state.attributes.get('restored', False) else None
        commands = self.model.evaluate(now.timestamp(), dt_util.as_local(now), temperature, states,
            sensor_fresh=fresh, monitor_only=bool(self.entry.options.get('monitor_only', False)))
        failures = {}
        for entity, turn_on in commands.items():
            if self._stopped:
                break
            try:
                async with asyncio.timeout(10):
                    await self.hass.services.async_call('switch', 'turn_on' if turn_on else 'turn_off',
                        {'entity_id': entity}, blocking=True)
            except Exception as exc:
                failures[entity] = type(exc).__name__
                _LOGGER.warning('UFH switch command failed for %s: %s', entity, exc)
        self.diagnostics = {
            'state': 'switch_error' if failures else self.model.status,
            'supply_temperature': temperature,
            'sensor_fresh': fresh,
            'pumps': {e: {'state': states[e], 'reason': self.model.reasons.get(e, self.model.status),
                          'command': commands.get(e),
                          'state_since': self.model.pumps[e].since,
                          'last_confirmed_state': self.model.pumps[e].state,
                          'exercise_idle_since': self.model.pumps[e].since if self.model.pumps[e].state == 'off' else None,
                          'exercise_day': self.model.pumps[e].exercise_day,
                          'exercise_until': self.model.pumps[e].exercise_until} for e in states},
            'switch_errors': failures,
        }
        try:
            await self._save()
        except OSError:
            self.diagnostics['storage_error'] = True
            _LOGGER.exception('Could not persist UFH circulation state')
        async_dispatcher_send(self.hass, f'{UFH_SIGNAL}_{self.entry.entry_id}')

    async def _on_stop(self, _event):
        await self.async_stop()

    async def async_stop(self):
        self._stopped = True
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []
        if getattr(self, '_stop_unsub', None):
            self._stop_unsub()
            self._stop_unsub = None
        if self._task and not self._task.done():
            await self._task
        async with self._lock:
            await self._save()
