from __future__ import annotations

import pandas as pd

from semantic_weather_common import (
    BASE_DIR,
    GROUND_TRUTH_DIR,
    SEARCH_QUERIES,
    SOURCES,
    add_semantic_columns,
    binary_metrics,
    read_source,
)


OUTPUT_QUERY_METRICS = BASE_DIR / "query_metrics.csv"
OUTPUT_QUERY_RESULTS = BASE_DIR / "query_results_preview.csv"


def build_joined_frame(source_key: str, source_dir) -> pd.DataFrame:
    ground_truth = add_semantic_columns(read_source(GROUND_TRUTH_DIR, "ground_truth"), "ground_truth")
    source = add_semantic_columns(read_source(source_dir, source_key), source_key)
    joined = ground_truth.merge(source, on="time", how="inner")
    if joined.empty:
        raise ValueError(f"{source_key}: no common timestamps with ground truth")
    return joined


def main() -> None:
    metric_rows = []
    preview_rows = []

    for source in SOURCES:
        try:
            joined = build_joined_frame(source.key, source.directory)
        except Exception as error:
            print(f"Pomijam {source.name}: {error}")
            continue

        for query_id, query_name, query_fn in SEARCH_QUERIES:
            predicted = query_fn(joined, source.key)
            actual = query_fn(joined, "ground_truth")
            metrics = binary_metrics(predicted, actual)

            metric_rows.append(
                {
                    "query_id": query_id,
                    "query": query_name,
                    "model": source.name,
                    "key": source.key,
                    "hours": len(joined),
                    "retrieved": int(predicted.sum()),
                    "relevant": int(actual.sum()),
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "precision_percent": metrics["precision"] * 100,
                    "recall_percent": metrics["recall"] * 100,
                    "f1_percent": metrics["f1"] * 100,
                    "success_rate_percent": metrics["success_rate"] * 100,
                }
            )

            preview = joined.loc[
                predicted,
                [
                    "time",
                    f"temp_label_{source.key}",
                    f"rain_label_{source.key}",
                    f"wind_label_{source.key}",
                    f"decision_{source.key}",
                ],
            ].head(6)
            for _, row in preview.iterrows():
                preview_rows.append(
                    {
                        "query_id": query_id,
                        "query": query_name,
                        "model": source.name,
                        "time": row["time"],
                        "temp_label": row[f"temp_label_{source.key}"],
                        "rain_label": row[f"rain_label_{source.key}"],
                        "wind_label": row[f"wind_label_{source.key}"],
                        "decision": row[f"decision_{source.key}"],
                    }
                )

    if not metric_rows:
        raise SystemExit("Brak danych do testow wyszukiwania.")

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["query_id", "f1_percent", "success_rate_percent"],
        ascending=[True, False, False],
    )
    metrics_df.to_csv(OUTPUT_QUERY_METRICS, index=False)
    pd.DataFrame(preview_rows).to_csv(OUTPUT_QUERY_RESULTS, index=False)

    print("Zapisano:")
    print(f"  {OUTPUT_QUERY_METRICS.name}")
    print(f"  {OUTPUT_QUERY_RESULTS.name}")
    print()
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
