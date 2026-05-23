from __future__ import annotations

import math

import pandas as pd

from semantic_weather_common import (
    BASE_DIR,
    GROUND_TRUTH_DIR,
    add_semantic_columns,
    critical_mismatch_summary,
    read_metric,
    read_source,
)


OUTPUT_DIR = BASE_DIR / "own_model"
WEIGHTS_FILE = BASE_DIR / "own_model_weights.csv"
BACKTEST_FILE = BASE_DIR / "own_model_backtest.csv"
SEMANTIC_FILE = BASE_DIR / "own_model_semantic.csv"

METRICS = {
    "temp": {"file": "temp.csv", "min_value": None},
    "opady": {"file": "opady.csv", "min_value": 0.0},
    "wiatr": {"file": "wiatr.csv", "min_value": 0.0},
}

COMPONENT_WEIGHT_CANDIDATES = [
    {"last": 1.00, "mean3": 0.00, "mean6": 0.00, "season24": 0.00, "trend": 0.00},
    {"last": 0.60, "mean3": 0.30, "mean6": 0.10, "season24": 0.00, "trend": 0.00},
    {"last": 0.45, "mean3": 0.25, "mean6": 0.10, "season24": 0.20, "trend": 0.00},
    {"last": 0.40, "mean3": 0.25, "mean6": 0.15, "season24": 0.15, "trend": 0.05},
    {"last": 0.55, "mean3": 0.15, "mean6": 0.00, "season24": 0.00, "trend": 0.30},
    {"last": 0.35, "mean3": 0.35, "mean6": 0.20, "season24": 0.10, "trend": 0.00},
]


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return math.sqrt(float(((actual - predicted) ** 2).mean()))


def critical_negative_points(metric: str, actual: pd.Series, predicted: pd.Series) -> pd.Series:
    points = pd.Series(0.0, index=actual.index)

    if metric == "temp":
        points -= 8.0 * ((actual <= 0.0) != (predicted <= 0.0))
        points -= 4.0 * ((actual <= -5.0) != (predicted <= -5.0))
        points -= 4.0 * ((actual >= 30.0) != (predicted >= 30.0))
        return points

    if metric == "opady":
        points -= 3.0 * ((actual > 0.1) != (predicted > 0.1))
        points -= 8.0 * ((actual > 0.5) != (predicted > 0.5))
        points -= 12.0 * ((actual > 2.0) != (predicted > 2.0))
        return points

    if metric == "wiatr":
        points -= 5.0 * ((actual >= 5.0) != (predicted >= 5.0))
        points -= 10.0 * ((actual >= 8.0) != (predicted >= 8.0))
        return points

    raise ValueError(f"Unknown metric: {metric}")


def critical_weighted_error(metric: str, actual: pd.Series, predicted: pd.Series) -> float:
    error = (actual - predicted).abs()
    negative_points = critical_negative_points(metric, actual, predicted)
    return float((error - negative_points).mean())


def read_history(metric: str) -> pd.DataFrame:
    return read_metric(
        GROUND_TRUTH_DIR / METRICS[metric]["file"],
        metric,
        "ground_truth",
    ).reset_index(drop=True)


def historical_components(values: pd.Series, index: int) -> dict[str, float]:
    history = values.iloc[:index].dropna()
    if history.empty:
        raise ValueError("Historical model needs at least one previous observation.")

    last = float(history.iloc[-1])
    previous = float(history.iloc[-2]) if len(history) >= 2 else last
    return {
        "last": last,
        "mean3": float(history.tail(3).mean()),
        "mean6": float(history.tail(6).mean()),
        "season24": float(values.iloc[index - 24]) if index >= 24 else last,
        "trend": last + (last - previous),
    }


def predict_from_components(components: dict[str, float], weights: dict[str, float]) -> float:
    return sum(components[name] * weight for name, weight in weights.items())


def predict_series(frame: pd.DataFrame, metric: str, weights: dict[str, float]) -> pd.DataFrame:
    value_col = f"{metric}_ground_truth"
    predictions = []

    for index in range(1, len(frame)):
        components = historical_components(frame[value_col], index)
        prediction = predict_from_components(components, weights)

        min_value = METRICS[metric]["min_value"]
        if min_value is not None:
            prediction = max(min_value, prediction)

        predictions.append(
            {
                "time": frame.loc[index, "time"],
                metric: prediction,
            }
        )

    return pd.DataFrame(predictions)


def tune_weights(frame: pd.DataFrame, metric: str, train_size: int) -> dict[str, float]:
    value_col = f"{metric}_ground_truth"
    best_weights = COMPONENT_WEIGHT_CANDIDATES[0]
    best_error = float("inf")

    for candidate in COMPONENT_WEIGHT_CANDIDATES:
        rows = []
        for index in range(1, train_size):
            components = historical_components(frame[value_col], index)
            prediction = predict_from_components(components, candidate)
            min_value = METRICS[metric]["min_value"]
            if min_value is not None:
                prediction = max(min_value, prediction)
            rows.append({"actual": frame.loc[index, value_col], "predicted": prediction})

        result = pd.DataFrame(rows)
        error = critical_weighted_error(metric, result["actual"], result["predicted"])
        if error < best_error:
            best_error = error
            best_weights = candidate

    return best_weights


def apply_history_critical_corrections(
    history: pd.DataFrame,
    prediction: dict[str, float],
) -> dict[str, float]:
    corrected = dict(prediction)
    recent_temp = history["temp_ground_truth"].tail(3)
    recent_rain = history["opady_ground_truth"].tail(3)
    recent_wind = history["wiatr_ground_truth"].tail(3)

    if abs(corrected["temp"]) <= 0.8 and corrected["opady"] > 0.1:
        freezing_share = float((recent_temp <= 0.0).mean()) if not recent_temp.empty else 0.0
        if freezing_share >= 0.5:
            corrected["temp"] = min(corrected["temp"], -0.1)
        else:
            corrected["temp"] = max(corrected["temp"], 0.1)

    if 0.35 <= corrected["opady"] <= 0.55 and recent_rain.mean() > 0.5:
        corrected["opady"] = 0.51

    if 7.4 <= corrected["wiatr"] < 8.0 and recent_wind.max() >= 8.0:
        corrected["wiatr"] = 8.0

    return corrected


def build_own_forecast(metric_frames: dict[str, pd.DataFrame], weights: dict[str, dict[str, float]]) -> pd.DataFrame:
    predictions = {
        metric: predict_series(frame, metric, weights[metric])
        for metric, frame in metric_frames.items()
    }

    base = predictions["temp"]
    for metric in ["opady", "wiatr"]:
        base = base.merge(predictions[metric], on="time", how="inner")

    history = read_source(GROUND_TRUTH_DIR, "ground_truth").reset_index(drop=True)
    base = base.merge(history, on="time", how="inner")

    corrected_rows = []
    for _, row in base.iterrows():
        history_before_row = history[history["time"] < row["time"]]
        corrected = apply_history_critical_corrections(
            history_before_row,
            {"temp": row["temp"], "opady": row["opady"], "wiatr": row["wiatr"]},
        )
        corrected_rows.append(corrected)

    corrected_df = pd.DataFrame(corrected_rows)
    base[["temp", "opady", "wiatr"]] = corrected_df[["temp", "opady", "wiatr"]]
    return base[["time", "temp", "opady", "wiatr"]]


def save_outputs(forecast: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    forecast[["time", "temp"]].to_csv(OUTPUT_DIR / "temp.csv", index=False)
    forecast[["time", "opady"]].to_csv(OUTPUT_DIR / "opady.csv", index=False)
    forecast[["time", "wiatr"]].to_csv(OUTPUT_DIR / "wiatr.csv", index=False)


def main() -> None:
    metric_frames = {metric: read_history(metric) for metric in METRICS}
    train_size = max(8, int(len(metric_frames["temp"]) * 0.7))
    test_start_time = metric_frames["temp"].loc[train_size, "time"]

    weights = {
        metric: tune_weights(frame, metric, train_size)
        for metric, frame in metric_frames.items()
    }
    forecast = build_own_forecast(metric_frames, weights)
    save_outputs(forecast)

    weight_rows = [
        {"metric": metric, "component": component, "weight": weight}
        for metric, metric_weights in weights.items()
        for component, weight in metric_weights.items()
    ]
    pd.DataFrame(weight_rows).to_csv(WEIGHTS_FILE, index=False)

    ground_truth = read_source(GROUND_TRUTH_DIR, "ground_truth")
    own = read_source(OUTPUT_DIR, "own_model")
    joined = ground_truth.merge(own, on="time", how="inner")
    joined = add_semantic_columns(joined, "ground_truth")
    joined = add_semantic_columns(joined, "own_model")

    test = joined[joined["time"] >= test_start_time]
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
                "critical_negative_points": float(
                    critical_negative_points(metric, actual, predicted).sum()
                ),
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
            "critical_negative_points": None,
            "critical_weighted_error": None,
            "decision_accuracy_percent": decision_accuracy * 100,
            **critical,
        }
    )

    pd.DataFrame(backtest_rows).to_csv(BACKTEST_FILE, index=False)
    joined.to_csv(SEMANTIC_FILE, index=False)

    print("Zapisano historyczna prognoze naszego modelu:")
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
