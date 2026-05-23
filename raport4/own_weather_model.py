from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from semantic_weather_common import (
    BASE_DIR,
    GROUND_TRUTH_DIR,
    SOURCES,
    add_semantic_columns,
    critical_mismatch_summary,
    read_metric,
    read_source,
)


OUTPUT_DIR = BASE_DIR / "own_model"
WEIGHTS_FILE = BASE_DIR / "own_model_weights.csv"
BACKTEST_FILE = BASE_DIR / "own_model_backtest.csv"
SEMANTIC_FILE = BASE_DIR / "own_model_semantic.csv"

MODEL_SOURCES = [source for source in SOURCES if source.key != "own_model"]
METRICS = {
    "temp": {"file": "temp.csv", "min_value": None},
    "opady": {"file": "opady.csv", "min_value": 0.0},
    "wiatr": {"file": "wiatr.csv", "min_value": 0.0},
}


def cap_and_normalize(weights: dict[str, float], cap: float = 0.45) -> dict[str, float]:
    capped = dict(weights)
    for _ in range(10):
        total = sum(capped.values())
        if total <= 0:
            equal = 1 / len(capped)
            return {key: equal for key in capped}

        capped = {key: value / total for key, value in capped.items()}
        above = {key: value for key, value in capped.items() if value > cap}
        if not above:
            return capped

        fixed_mass = len(above) * cap
        free_keys = [key for key in capped if key not in above]
        free_mass = sum(capped[key] for key in free_keys)
        for key in above:
            capped[key] = cap
        if free_keys and free_mass > 0:
            scale = (1 - fixed_mass) / free_mass
            for key in free_keys:
                capped[key] *= scale

    total = sum(capped.values())
    return {key: value / total for key, value in capped.items()}


def critical_weighted_error(metric: str, actual: pd.Series, predicted: pd.Series) -> float:
    error = (actual - predicted).abs()

    if metric == "temp":
        penalty = 4.0 * ((actual <= 0.0) != (predicted <= 0.0))
        penalty += 2.0 * ((actual <= -5.0) != (predicted <= -5.0))
        penalty += 2.0 * ((actual >= 30.0) != (predicted >= 30.0))
        return float((error + penalty).mean())

    if metric == "opady":
        penalty = 2.0 * ((actual > 0.1) != (predicted > 0.1))
        penalty += 4.0 * ((actual > 0.5) != (predicted > 0.5))
        penalty += 6.0 * ((actual > 2.0) != (predicted > 2.0))
        return float((error + penalty).mean())

    if metric == "wiatr":
        penalty = 2.0 * ((actual >= 5.0) != (predicted >= 5.0))
        penalty += 4.0 * ((actual >= 8.0) != (predicted >= 8.0))
        return float((error + penalty).mean())

    raise ValueError(f"Unknown metric: {metric}")


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return math.sqrt(float(((actual - predicted) ** 2).mean()))


def build_metric_frame(metric: str) -> pd.DataFrame:
    ground_truth = read_metric(
        GROUND_TRUTH_DIR / METRICS[metric]["file"],
        metric,
        "ground_truth",
    )
    frame = ground_truth

    for source in MODEL_SOURCES:
        try:
            source_metric = read_metric(source.directory / METRICS[metric]["file"], metric, source.key)
        except Exception as error:
            print(f"Pomijam {source.name} dla {metric}: {error}")
            continue
        frame = frame.merge(source_metric, on="time", how="inner")

    if frame.empty:
        raise ValueError(f"Brak wspolnych danych dla metryki {metric}")

    return frame.sort_values("time").reset_index(drop=True)


def learn_weights(frame: pd.DataFrame, metric: str, train_size: int) -> dict[str, float]:
    train = frame.iloc[:train_size]
    actual = train[f"{metric}_ground_truth"]
    raw_weights = {}

    for source in MODEL_SOURCES:
        column = f"{metric}_{source.key}"
        if column not in train.columns:
            continue
        error = critical_weighted_error(metric, actual, train[column])
        raw_weights[source.key] = 1.0 / ((error + 0.25) ** 2)

    return cap_and_normalize(raw_weights)


def apply_critical_corrections(row: pd.Series, predictions: dict[str, float], weights: dict[str, dict[str, float]]) -> dict[str, float]:
    corrected = dict(predictions)

    if abs(corrected["temp"]) <= 0.8 and corrected["opady"] > 0.1:
        freezing_weight = sum(
            weights["temp"].get(source.key, 0.0)
            for source in MODEL_SOURCES
            if f"temp_{source.key}" in row and row[f"temp_{source.key}"] <= 0.0
        )
        if freezing_weight >= 0.40:
            corrected["temp"] = min(corrected["temp"], -0.1)
        else:
            corrected["temp"] = max(corrected["temp"], 0.1)

    rain_risk_weight = sum(
        weights["opady"].get(source.key, 0.0)
        for source in MODEL_SOURCES
        if f"opady_{source.key}" in row and row[f"opady_{source.key}"] > 0.5
    )
    if 0.35 <= corrected["opady"] <= 0.55 and rain_risk_weight >= 0.40:
        corrected["opady"] = 0.51

    wind_risk_weight = sum(
        weights["wiatr"].get(source.key, 0.0)
        for source in MODEL_SOURCES
        if f"wiatr_{source.key}" in row and row[f"wiatr_{source.key}"] >= 8.0
    )
    if 7.4 <= corrected["wiatr"] < 8.0 and wind_risk_weight >= 0.40:
        corrected["wiatr"] = 8.0

    return corrected


def predict_metric(frame: pd.DataFrame, metric: str, weights: dict[str, float]) -> pd.Series:
    prediction = pd.Series(0.0, index=frame.index)
    used_weight = 0.0
    for source_key, weight in weights.items():
        column = f"{metric}_{source_key}"
        if column in frame.columns:
            prediction += frame[column] * weight
            used_weight += weight

    if used_weight == 0:
        raise ValueError(f"Nie ma wag dla {metric}")

    prediction = prediction / used_weight
    min_value = METRICS[metric]["min_value"]
    if min_value is not None:
        prediction = prediction.clip(lower=min_value)
    return prediction


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    metric_frames = {metric: build_metric_frame(metric) for metric in METRICS}
    train_size = max(8, int(len(metric_frames["temp"]) * 0.7))
    weights = {
        metric: learn_weights(frame, metric, train_size)
        for metric, frame in metric_frames.items()
    }

    base = metric_frames["temp"][["time"]].copy()
    for metric, frame in metric_frames.items():
        prediction = predict_metric(frame, metric, weights[metric])
        base = base.merge(
            pd.DataFrame({"time": frame["time"], metric: prediction}),
            on="time",
            how="inner",
        )

    wide = base.copy()
    for metric, frame in metric_frames.items():
        source_columns = ["time"] + [
            f"{metric}_{source.key}"
            for source in MODEL_SOURCES
            if f"{metric}_{source.key}" in frame.columns
        ]
        wide = wide.merge(frame[source_columns], on="time", how="inner")

    corrected_rows = []
    for _, row in wide.iterrows():
        corrected = apply_critical_corrections(
            row,
            {"temp": row["temp"], "opady": row["opady"], "wiatr": row["wiatr"]},
            weights,
        )
        corrected_rows.append(corrected)
    corrected_df = pd.DataFrame(corrected_rows)
    base[["temp", "opady", "wiatr"]] = corrected_df[["temp", "opady", "wiatr"]]

    base[["time", "temp"]].to_csv(OUTPUT_DIR / "temp.csv", index=False)
    base[["time", "opady"]].to_csv(OUTPUT_DIR / "opady.csv", index=False)
    base[["time", "wiatr"]].to_csv(OUTPUT_DIR / "wiatr.csv", index=False)

    weight_rows = [
        {"metric": metric, "source": source_key, "weight": weight}
        for metric, metric_weights in weights.items()
        for source_key, weight in metric_weights.items()
    ]
    pd.DataFrame(weight_rows).to_csv(WEIGHTS_FILE, index=False)

    ground_truth = read_source(GROUND_TRUTH_DIR, "ground_truth")
    own = read_source(OUTPUT_DIR, "own_model")
    joined = ground_truth.merge(own, on="time", how="inner")
    joined = add_semantic_columns(joined, "ground_truth")
    joined = add_semantic_columns(joined, "own_model")

    test = joined.iloc[train_size:]
    backtest_rows = []
    for metric in METRICS:
        actual = test[f"{metric}_ground_truth"]
        predicted = test[f"{metric}_own_model"]
        backtest_rows.append(
            {
                "metric": metric,
                "train_hours": train_size,
                "test_hours": len(test),
                "mae": float((actual - predicted).abs().mean()),
                "rmse": rmse(actual, predicted),
                "critical_weighted_error": critical_weighted_error(metric, actual, predicted),
            }
        )

    decision_accuracy = (test["decision_ground_truth"] == test["decision_own_model"]).mean()
    critical = critical_mismatch_summary(test, "own_model")
    backtest_rows.append(
        {
            "metric": "decision",
            "train_hours": train_size,
            "test_hours": len(test),
            "mae": None,
            "rmse": None,
            "critical_weighted_error": None,
            "decision_accuracy_percent": decision_accuracy * 100,
            **critical,
        }
    )
    pd.DataFrame(backtest_rows).to_csv(BACKTEST_FILE, index=False)
    joined.to_csv(SEMANTIC_FILE, index=False)

    print("Zapisano prognoze naszego modelu:")
    print(f"  {OUTPUT_DIR / 'temp.csv'}")
    print(f"  {OUTPUT_DIR / 'opady.csv'}")
    print(f"  {OUTPUT_DIR / 'wiatr.csv'}")
    print(f"  {WEIGHTS_FILE.name}")
    print(f"  {BACKTEST_FILE.name}")
    print()
    print(pd.DataFrame(weight_rows).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print()
    print(pd.DataFrame(backtest_rows).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
