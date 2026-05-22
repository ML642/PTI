from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_DIR = BASE_DIR / "default_model"

METRICS = [
    {
        "name": "Wiatr",
        "key": "wiatr",
        "file": "wiatr.csv",
        "unit": "m/s",
        "thresholds": (0.5, 1.5, 3.0),
        "ground_truth_multiplier": 1.0,
        "output": BASE_DIR / "wykres_wiatr.png",
    },
    {
        "name": "Opady",
        "key": "opady",
        "file": "opady.csv",
        "unit": "mm",
        "thresholds": (0.1, 0.5, 2.0),
        "ground_truth_multiplier": 1.0,
        "output": BASE_DIR / "wykres_opady.png",
    },
]

<<<<<<< HEAD
=======
<<<<<<< HEAD
# ДОДАНО GFS У ЦЕЙ СПИСОК
=======
>>>>>>> 01c3c019d326d2f025e1b93d51ffd6c48f8b06eb
>>>>>>> 301eb397d7ed97a2c6a5ae04bc8fb680020c3c67
MODEL_SOURCES = [
    {
        "name": "AIFS",
        "key": "aifs",
        "dir": BASE_DIR / "aifs",
        "color": "blue",
        "linestyle": "--",
    },
    {
        "name": "GraphCast",
        "key": "graphcast",
        "dir": BASE_DIR / "graphcast",
        "color": "red",
        "linestyle": ":",
    },
    {
        "name": "Open-Meteo",
        "key": "open_meteo",
        "dir": BASE_DIR / "open_meteo",
        "color": "green",
        "linestyle": "-.",
    },
    {
        "name": "Yr.no",
        "key": "yr_no",
        "dir": BASE_DIR / "yr_no",
        "color": "purple",
        "linestyle": (0, (3, 1, 1, 1)),
    },
    {
        "name": "GFS",
        "key": "gfs",
        "dir": BASE_DIR / "gfs",
        "color": "orange",
        "linestyle": "-",
    },
]


def read_metric(path, metric_key, source_key, multiplier=1.0):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()

    expected_columns = {"time", metric_key}

    if not expected_columns.issubset(df.columns):
        columns = ", ".join(df.columns)
        raise ValueError(
            f"{source_key}: expected columns time,{metric_key} in {path}, got: {columns}"
        )

    value_column = f"{metric_key}_{source_key}"
    df = df[["time", metric_key]].rename(columns={metric_key: value_column})
    df[value_column] = df[value_column] * multiplier
    df["time"] = pd.to_datetime(df["time"])
    return df


def date_range_text(df):
    if df.empty:
        return "Brak danych"
    return f"{df['time'].min():%Y-%m-%d %H:%M} - {df['time'].max():%Y-%m-%d %H:%M}"


def przyznaj_punkty(blad, thresholds):
    excellent, good, ok = thresholds

    if blad <= excellent:
        return 3
    if blad <= good:
        return 2
    if blad <= ok:
        return 1
    return 0


def analyze_source_metric(ground_truth, source_config, metric_config):
    source = read_metric(
        source_config["dir"] / metric_config["file"],
        metric_config["key"],
        source_config["key"],
    )
    
    if source.empty or ground_truth.empty:
         return {
            "source": source,
            "merged": pd.DataFrame(),
            "has_data": False,
        }
        
    merged = ground_truth.merge(source, on="time")

    if merged.empty:
        return {
            "source": source,
            "merged": merged,
            "has_data": False,
        }

    ground_truth_column = f"{metric_config['key']}_ground_truth"
    source_column = f"{metric_config['key']}_{source_config['key']}"

    merged["blad"] = (merged[ground_truth_column] - merged[source_column]).abs()
    merged["punkty"] = merged["blad"].apply(
        lambda blad: przyznaj_punkty(blad, metric_config["thresholds"])
    )

    points = merged["punkty"].sum()
    max_points = len(merged) * 3
    rmse = np.sqrt((merged["blad"] ** 2).mean())

    return {
        "source": source,
        "merged": merged,
        "has_data": True,
        "points": points,
        "max_points": max_points,
        "score_percent": points / max_points * 100,
        "rmse": rmse,
        "hours": len(merged),
    }


def analyze_metric(metric_config):
    ground_truth = read_metric(
        GROUND_TRUTH_DIR / metric_config["file"],
        metric_config["key"],
        "ground_truth",
        metric_config.get("ground_truth_multiplier", 1.0),
    )

    results = {}
    for source_config in MODEL_SOURCES:
        results[source_config["key"]] = analyze_source_metric(
            ground_truth,
            source_config,
            metric_config,
        )

    return {
        "metric": metric_config,
        "ground_truth": ground_truth,
        "results": results,
    }


def sorted_metric_results(metric_analysis):
    available = [
        (source_config, metric_analysis["results"][source_config["key"]])
        for source_config in MODEL_SOURCES
        if metric_analysis["results"][source_config["key"]]["has_data"]
    ]
    available.sort(key=lambda item: (-item[1]["score_percent"], item[1]["rmse"]))
    return available


def print_metric_results(metric_analysis):
    metric = metric_analysis["metric"]
    ground_truth = metric_analysis["ground_truth"]

    print("\n" + "=" * 64)
    print(f"RANKING: {metric['name'].upper()}")
    print("=" * 64)
    print("Ground truth: default_model")
    print(f"Zakres default_model: {date_range_text(ground_truth)}")
    if "ground_truth_note" in metric:
        print(f"Uwaga: {metric['ground_truth_note']}")
    print(
        "Progi punktacji: "
        f"3 pkt <= {metric['thresholds'][0]} {metric['unit']}, "
        f"2 pkt <= {metric['thresholds'][1]} {metric['unit']}, "
        f"1 pkt <= {metric['thresholds'][2]} {metric['unit']}\n"
    )

    for index, (source_config, result) in enumerate(
        sorted_metric_results(metric_analysis),
        start=1,
    ):
        print(f"{index}. {source_config['name']}:")
        print(
            f"   -> Punkty: {result['points']} / {result['max_points']} "
            f"({result['score_percent']:.1f}%)"
        )
        print(
            f"   -> RMSE: {result['rmse']:.2f} {metric['unit']} "
            "(im mniej, tym lepiej)"
        )
        print(f"   -> Wspolne godziny: {result['hours']}")
        print(f"   -> Zakres modelu: {date_range_text(result['source'])}")
        print("-" * 64)

    print("=" * 64)


def build_total_ranking(metric_analyses):
    ranking = {}

    for source_config in MODEL_SOURCES:
        ranking[source_config["key"]] = {
            "name": source_config["name"],
            "points": 0,
            "max_points": 0,
            "hours": 0,
            "details": [],
        }

    for metric_analysis in metric_analyses:
        metric = metric_analysis["metric"]
        for source_config in MODEL_SOURCES:
            result = metric_analysis["results"][source_config["key"]]
            if not result["has_data"]:
                continue

            entry = ranking[source_config["key"]]
            entry["points"] += result["points"]
            entry["max_points"] += result["max_points"]
            entry["hours"] += result["hours"]
            entry["details"].append(
                f"{metric['name']}: {result['points']}/{result['max_points']} "
                f"({result['score_percent']:.1f}%)"
            )

    ranked = [entry for entry in ranking.values() if entry["max_points"] > 0]
    for entry in ranked:
        entry["score_percent"] = entry["points"] / entry["max_points"] * 100

    ranked.sort(key=lambda entry: (-entry["score_percent"], -entry["points"]))
    return ranked


def print_total_ranking(metric_analyses):
    print("\n" + "=" * 64)
    print("RANKING LACZNY: WIATR + OPADY")
    print("=" * 64)

    for index, entry in enumerate(build_total_ranking(metric_analyses), start=1):
        print(f"{index}. {entry['name']}:")
        print(
            f"   -> Punkty lacznie: {entry['points']} / {entry['max_points']} "
            f"({entry['score_percent']:.1f}%)"
        )
        print(f"   -> Suma wspolnych godzin po metrykach: {entry['hours']}")
        print("   -> Szczegoly: " + "; ".join(entry["details"]))
        print("-" * 64)

    print("=" * 64)


def plot_metric(metric_analysis):
    metric = metric_analysis["metric"]
    ground_truth = metric_analysis["ground_truth"]
    
    if ground_truth.empty:
        print(f"❌ Brak danych Ground Truth dla {metric['name']}, pomijam wykres.")
        return
        
    ground_truth_column = f"{metric['key']}_ground_truth"

    print(f"Generowanie wykresu: {metric['name']}...")

    plt.figure(figsize=(14, 7))
    plt.plot(
        ground_truth["time"],
        ground_truth[ground_truth_column],
        label="Ground truth (default_model)",
        color="black",
        linewidth=3,
    )

    for source_config in MODEL_SOURCES:
        result = metric_analysis["results"][source_config["key"]]
        if not result["has_data"]:
            continue

        source_column = f"{metric['key']}_{source_config['key']}"
        plt.plot(
            result["merged"]["time"],
            result["merged"][source_column],
            label=source_config["name"],
            color=source_config["color"],
            linestyle=source_config["linestyle"],
        )

    plt.xticks(ground_truth["time"][::24], rotation=45)
    plt.title(f"{metric['name']}: porownanie prognoz dla Warszawy", fontsize=16)
    plt.xlabel("Czas", fontsize=12)
    plt.ylabel(f"{metric['name']} ({metric['unit']})", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(metric["output"], dpi=300)
    plt.close()

    print(f"Wykres zapisano jako '{metric['output'].name}'")


def main():
    print("Wczytywanie danych dla wiatru i opadow...")

    metric_analyses = [analyze_metric(metric_config) for metric_config in METRICS]

    for metric_analysis in metric_analyses:
        print_metric_results(metric_analysis)

    print_total_ranking(metric_analyses)

    for metric_analysis in metric_analyses:
        plot_metric(metric_analysis)


if __name__ == "__main__":
    main()