from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from semantic_weather_common import (
    BASE_DIR,
    GROUND_TRUTH_DIR,
    SOURCES,
    add_semantic_columns,
    binary_metrics,
    critical_mismatch_rows,
    critical_mismatch_summary,
    read_source,
    semantic_similarity,
)


OUTPUT_SUMMARY = BASE_DIR / "semantic_summary.csv"
OUTPUT_CRITICAL_SUMMARY = BASE_DIR / "critical_points_summary.csv"
OUTPUT_CRITICAL_MISMATCHES = BASE_DIR / "critical_points_mismatches.csv"
OUTPUT_INDEX = BASE_DIR / "semantic_index_preview.csv"
OUTPUT_DECISION_PLOT = BASE_DIR / "semantic_decision_accuracy.png"
OUTPUT_TIMELINE = BASE_DIR / "semantic_timeline.png"


def build_joined_frame(source_key: str, source_dir) -> pd.DataFrame:
    ground_truth = read_source(GROUND_TRUTH_DIR, "ground_truth")
    ground_truth = add_semantic_columns(ground_truth, "ground_truth")

    source = read_source(source_dir, source_key)
    source = add_semantic_columns(source, source_key)

    joined = ground_truth.merge(source, on="time", how="inner")
    if joined.empty:
        raise ValueError(f"{source_key}: no common timestamps with ground truth")

    joined[f"semantic_similarity_{source_key}"] = joined.apply(
        lambda row: semantic_similarity(row, source_key),
        axis=1,
    )
    joined[f"bad_pred_{source_key}"] = joined[f"decision_{source_key}"] == "zle"
    joined["bad_ground_truth"] = joined["decision_ground_truth"] == "zle"
    return joined


def summarize_source(source, joined: pd.DataFrame) -> dict[str, float | str | int]:
    adverse = binary_metrics(joined[f"bad_pred_{source.key}"], joined["bad_ground_truth"])
    critical = critical_mismatch_summary(joined, source.key)
    return {
        "model": source.name,
        "key": source.key,
        "hours": len(joined),
        "semantic_similarity_percent": joined[f"semantic_similarity_{source.key}"].mean()
        * 100,
        "decision_accuracy_percent": (
            joined[f"decision_{source.key}"] == joined["decision_ground_truth"]
        ).mean()
        * 100,
        "temp_label_accuracy_percent": (
            joined[f"temp_label_{source.key}"] == joined["temp_label_ground_truth"]
        ).mean()
        * 100,
        "rain_label_accuracy_percent": (
            joined[f"rain_label_{source.key}"] == joined["rain_label_ground_truth"]
        ).mean()
        * 100,
        "wind_label_accuracy_percent": (
            joined[f"wind_label_{source.key}"] == joined["wind_label_ground_truth"]
        ).mean()
        * 100,
        "adverse_precision_percent": adverse["precision"] * 100,
        "adverse_recall_percent": adverse["recall"] * 100,
        "adverse_f1_percent": adverse["f1"] * 100,
        **critical,
    }


def save_semantic_index_preview() -> None:
    ground_truth = add_semantic_columns(read_source(GROUND_TRUTH_DIR, "ground_truth"), "ground_truth")
    preview = ground_truth[
        [
            "time",
            "temp_ground_truth",
            "opady_ground_truth",
            "wiatr_ground_truth",
            "temp_label_ground_truth",
            "rain_label_ground_truth",
            "wind_label_ground_truth",
            "decision_ground_truth",
        ]
    ].head(48)
    preview.to_csv(OUTPUT_INDEX, index=False)


def plot_decision_accuracy(summary: pd.DataFrame) -> None:
    ordered = summary.sort_values("semantic_similarity_percent", ascending=False)
    x = range(len(ordered))

    plt.figure(figsize=(11, 6))
    plt.bar(
        [value - 0.18 for value in x],
        ordered["semantic_similarity_percent"],
        width=0.36,
        label="Zgodnosc semantyczna",
        color="#2f6f73",
    )
    plt.bar(
        [value + 0.18 for value in x],
        ordered["decision_accuracy_percent"],
        width=0.36,
        label="Zgodnosc decyzji",
        color="#d08c2f",
    )
    plt.xticks(list(x), ordered["model"], rotation=20, ha="right")
    plt.ylabel("Wynik [%]")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DECISION_PLOT, dpi=300)
    plt.close()


def plot_timeline(joined_by_source: dict[str, pd.DataFrame], summary: pd.DataFrame) -> None:
    decision_to_value = {"zle": 0, "ostroznie": 1, "dobre": 2}
    top_keys = list(
        summary.sort_values("semantic_similarity_percent", ascending=False)["key"].head(3)
    )
    first_key = top_keys[0]
    base = joined_by_source[first_key]

    plt.figure(figsize=(13, 6))
    plt.step(
        base["time"],
        base["decision_ground_truth"].map(decision_to_value),
        where="post",
        label="Ground truth",
        color="black",
        linewidth=2.5,
    )

    colors = ["#2f6f73", "#d08c2f", "#6b5ca5"]
    for color, key in zip(colors, top_keys):
        joined = joined_by_source[key]
        model_name = summary.loc[summary["key"] == key, "model"].iloc[0]
        plt.step(
            joined["time"],
            joined[f"decision_{key}"].map(decision_to_value),
            where="post",
            label=model_name,
            alpha=0.8,
            color=color,
        )

    plt.yticks([0, 1, 2], ["zle", "ostroznie", "dobre"])
    plt.xlabel("Czas")
    plt.ylabel("Klasa decyzji")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_TIMELINE, dpi=300)
    plt.close()


def main() -> None:
    joined_by_source = {}
    summaries = []
    critical_rows = []

    for source in SOURCES:
        try:
            joined = build_joined_frame(source.key, source.directory)
        except Exception as error:
            print(f"Pomijam {source.name}: {error}")
            continue

        joined_by_source[source.key] = joined
        summaries.append(summarize_source(source, joined))
        source_critical_rows = critical_mismatch_rows(joined, source.key)
        if not source_critical_rows.empty:
            source_critical_rows.insert(0, "model_name", source.name)
            critical_rows.append(source_critical_rows)

    if not summaries:
        raise SystemExit("Brak danych do analizy semantycznej.")

    summary = pd.DataFrame(summaries)
    summary = summary.sort_values(
        ["semantic_similarity_percent", "decision_accuracy_percent"],
        ascending=False,
    )
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    critical_columns = [
        "model",
        "key",
        "hours",
        "critical_reference_cases",
        "critical_predicted_cases",
        "critical_mismatch_cases",
        "critical_mismatch_percent",
        "freezing_boundary_mismatches",
        "snow_or_ice_risk_mismatches",
        "significant_precipitation_mismatches",
        "heavy_precipitation_mismatches",
        "strong_wind_mismatches",
        "thermal_extreme_mismatches",
    ]
    critical_summary = summary[critical_columns].sort_values(
        ["critical_mismatch_cases", "critical_mismatch_percent"],
        ascending=[True, True],
    )
    critical_summary.to_csv(OUTPUT_CRITICAL_SUMMARY, index=False)
    if critical_rows:
        pd.concat(critical_rows, ignore_index=True).to_csv(
            OUTPUT_CRITICAL_MISMATCHES,
            index=False,
        )
    else:
        pd.DataFrame(
            columns=[
                "model_name",
                "time",
                "event",
                "event_name",
                "model",
                "actual_event",
                "predicted_event",
                "temp_ground_truth",
                "opady_ground_truth",
                "wiatr_ground_truth",
            ]
        ).to_csv(OUTPUT_CRITICAL_MISMATCHES, index=False)
    save_semantic_index_preview()
    plot_decision_accuracy(summary)
    plot_timeline(joined_by_source, summary)

    print("Zapisano:")
    print(f"  {OUTPUT_SUMMARY.name}")
    print(f"  {OUTPUT_CRITICAL_SUMMARY.name}")
    print(f"  {OUTPUT_CRITICAL_MISMATCHES.name}")
    print(f"  {OUTPUT_INDEX.name}")
    print(f"  {OUTPUT_DECISION_PLOT.name}")
    print(f"  {OUTPUT_TIMELINE.name}")
    print()
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
