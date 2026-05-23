from __future__ import annotations

import pandas as pd

from semantic_weather_common import (
    BASE_DIR,
    CRITICAL_EVENTS,
    add_semantic_columns,
    critical_flags,
)


OUTPUT_FILE = BASE_DIR / "critical_scenarios.csv"


SCENARIOS = [
    {
        "scenario": "granica zamarzania i mozliwy snieg",
        "actual_temp": -0.6,
        "predicted_temp": 0.6,
        "actual_rain": 0.4,
        "predicted_rain": 0.4,
        "actual_wind": 1.5,
        "predicted_wind": 1.5,
        "why_critical": "maly blad temperatury zmienia interpretacje opadu: deszcz/snieg lub oblodzenie",
    },
    {
        "scenario": "pominiety opad istotny",
        "actual_temp": 3.0,
        "predicted_temp": 3.2,
        "actual_rain": 0.8,
        "predicted_rain": 0.1,
        "actual_wind": 2.0,
        "predicted_wind": 2.0,
        "why_critical": "liczbowo opad wydaje sie niewielki, ale przekracza prog zmiany decyzji",
    },
    {
        "scenario": "falszywy alarm opadowy",
        "actual_temp": 7.0,
        "predicted_temp": 7.1,
        "actual_rain": 0.0,
        "predicted_rain": 0.7,
        "actual_wind": 2.5,
        "predicted_wind": 2.5,
        "why_critical": "model moze niepotrzebnie odrzucic dobre okno czasowe",
    },
    {
        "scenario": "granica silnego wiatru",
        "actual_temp": 16.0,
        "predicted_temp": 16.0,
        "actual_rain": 0.0,
        "predicted_rain": 0.0,
        "actual_wind": 8.1,
        "predicted_wind": 7.8,
        "why_critical": "niewielki blad wiatru zmienia ocene bezpieczenstwa i komfortu",
    },
    {
        "scenario": "prog upalu",
        "actual_temp": 30.2,
        "predicted_temp": 29.4,
        "actual_rain": 0.0,
        "predicted_rain": 0.0,
        "actual_wind": 1.0,
        "predicted_wind": 1.0,
        "why_critical": "prog 30 C oznacza ryzyko przegrzania, mimo malej roznicy liczbowej",
    },
    {
        "scenario": "prog mrozu silniejszego",
        "actual_temp": -5.3,
        "predicted_temp": -4.7,
        "actual_rain": 0.0,
        "predicted_rain": 0.0,
        "actual_wind": 1.0,
        "predicted_wind": 1.0,
        "why_critical": "prog -5 C moze zmieniac ocene ryzyka dla pieszych i transportu",
    },
]


def build_frame() -> pd.DataFrame:
    rows = []
    for index, scenario in enumerate(SCENARIOS):
        rows.append(
            {
                "scenario_id": index + 1,
                "scenario": scenario["scenario"],
                "time": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=index),
                "temp_ground_truth": scenario["actual_temp"],
                "opady_ground_truth": scenario["actual_rain"],
                "wiatr_ground_truth": scenario["actual_wind"],
                "temp_model": scenario["predicted_temp"],
                "opady_model": scenario["predicted_rain"],
                "wiatr_model": scenario["predicted_wind"],
                "why_critical": scenario["why_critical"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = build_frame()
    df = add_semantic_columns(df, "ground_truth")
    df = add_semantic_columns(df, "model")

    actual_flags = critical_flags(df, "ground_truth")
    predicted_flags = critical_flags(df, "model")

    for event_key, event_name in CRITICAL_EVENTS.items():
        df[f"{event_key}_actual"] = actual_flags[event_key]
        df[f"{event_key}_predicted"] = predicted_flags[event_key]
        df[f"{event_key}_changed"] = actual_flags[event_key] != predicted_flags[event_key]

    changed_columns = [f"{event_key}_changed" for event_key in CRITICAL_EVENTS]
    df["critical_changes"] = df[changed_columns].sum(axis=1)
    df["decision_changed"] = df["decision_ground_truth"] != df["decision_model"]

    output_columns = [
        "scenario_id",
        "scenario",
        "temp_ground_truth",
        "temp_model",
        "opady_ground_truth",
        "opady_model",
        "wiatr_ground_truth",
        "wiatr_model",
        "decision_ground_truth",
        "decision_model",
        "critical_changes",
        "decision_changed",
        "why_critical",
    ]
    df[output_columns].to_csv(OUTPUT_FILE, index=False)

    print(f"Zapisano: {OUTPUT_FILE.name}")
    print(df[output_columns].to_string(index=False))


if __name__ == "__main__":
    main()
