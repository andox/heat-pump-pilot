"""Read-only local replay of a saved HA climate/decision snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "custom_components" / "heat_pump_pilot"))
from mpc_controller import MpcController
from pump_response import ResponseParameters
from response_optimizer import VirtualActuator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--price-curve", choices=("linear", "sqrt", "quadratic"), required=True,
                        help="The actual configured curve; older snapshots do not expose it.")
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8-sig"))
    if isinstance(snapshot, list):
        climate = next(s["attributes"] for s in snapshot if s["entity_id"] == "climate.heat_pump_pilot")
        decision = next(s["attributes"] for s in snapshot if s["entity_id"] == "sensor.heat_pump_pilot_decision")
    else:
        climate, decision = snapshot["climate"], snapshot["decision"]
    rows = []
    for tolerance, weight in dict.fromkeys([
        (climate["comfort_temperature_tolerance"], climate["price_comfort_weight"]),
        (climate["comfort_temperature_tolerance"], 0.8), (1.2, 0.8),
    ]):
        controller = MpcController(
            target_temperature=climate["temperature"], price_comfort_weight=weight,
            comfort_temperature_tolerance=tolerance, prediction_horizon_hours=24,
            price_penalty_curve=args.price_curve,
            heat_loss_coeff=climate["estimated_heat_loss_coefficient"],
            heat_gain_coeff=climate["estimated_heat_gain_coefficient"],
            background_gain=climate["estimated_background_gain"],
        )
        # Deliberately explicit unlearned fallback: saved state alone does not
        # include the fresh request queue needed for learned-response replay.
        actuator = VirtualActuator(
            climate["virtual_outdoor_heat_offset"], climate["virtual_outdoor_min_temp"],
            climate["virtual_outdoor_smoothing_alpha"] if climate["virtual_outdoor_smoothing_enabled"] else 1,
            climate["suggested_virtual_outdoor_temperature"],
            initial_duty=climate["effective_requested_duty_ratio"], learned_response=False,
        )
        start = time.perf_counter()
        _, result = controller.suggest_control(
            climate["current_temperature"], decision["outdoor_forecast"], decision["price_forecast"],
            price_baseline_override=decision["price_baseline"],
            response_context=(ResponseParameters(), (0, ()), actuator),
        )
        temperatures = result.predicted_temperatures
        rows.append(dict(tolerance=tolerance, price_weight=weight,
            minimum_c=min(temperatures), maximum_c=max(temperatures),
            heating_equivalent_hours=sum(result.predicted_heating)*0.25,
            first_request_hours=next((i*0.25 for i, d in enumerate(result.duty_sequence) if d>0), None),
            below_lower_degree_hours=sum(max(0, climate["temperature"]-tolerance-t)*0.25 for t in temperatures[1:]),
            comfort_status=result.comfort_status, compute_seconds=time.perf_counter()-start))
    print(json.dumps({"snapshot_control_time":climate["last_control_time"],
        "model":"unlearned proportional fallback; simulation, not measured savings", "plans":rows},indent=2))


if __name__ == "__main__":
    main()
