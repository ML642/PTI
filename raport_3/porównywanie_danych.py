from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

GROUND_TRUTH_FILE = BASE_DIR / "default_model" / "temp.csv"
OUTPUT_FILE = BASE_DIR / "wykres_temperatura.png"

MODEL_SOURCES = [
    {
        "name": "AIFS",
        "key": "aifs",
        "path": BASE_DIR / "aifs" / "temp.csv",
        "color": "blue",
        "linestyle": "--",
    },
    {
        "name": "GraphCast",
        "key": "graphcast",
        "path": BASE_DIR / "graphcast" / "temp.csv",
        "color": "red",
        "linestyle": ":",
    },
    {
        "name": "Open-Meteo",
        "key": "open_meteo",
        "path": BASE_DIR / "open_meteo" / "temp.csv",
        "color": "green",
        "linestyle": "-.",
    },
    {
        "name": "Yr.no",
        "key": "yr_no",
        "path": BASE_DIR / "yr_no" / "temp.csv",
        "color": "purple",
        "linestyle": (0, (3, 1, 1, 1)),
    },
]


def read_temperature(path, source_key):
    df = pd.read_csv(path)
    expected_columns = {"time", "temp"}

    if not expected_columns.issubset(df.columns):
        columns = ", ".join(df.columns)
        raise ValueError(
            f"{source_key}: expected columns time,temp in {path}, got: {columns}"
        )

    df = df[["time", "temp"]].rename(columns={"temp": f"temp_{source_key}"})
    df["time"] = pd.to_datetime(df["time"])
    return df


def date_range_text(df):
    return f"{df['time'].min():%Y-%m-%d %H:%M} - {df['time'].max():%Y-%m-%d %H:%M}"


def przyznaj_punkty(blad):
    """Im mniejszy blad, tym wiecej punktow otrzymuje model."""
    if blad <= 0.5:
        return 3
    if blad <= 1.5:
        return 2
    if blad <= 3.0:
        return 1
    return 0


def analyze_source(ground_truth, source_config):
    source = read_temperature(source_config["path"], source_config["key"])
    merged = ground_truth.merge(source, on="time")

    if merged.empty:
        return {
            "config": source_config,
            "source": source,
            "merged": merged,
            "has_data": False,
        }

    temp_source_column = f"temp_{source_config['key']}"
    merged["blad"] = (merged["temp_ground_truth"] - merged[temp_source_column]).abs()
    merged["punkty"] = merged["blad"].apply(przyznaj_punkty)

    points = merged["punkty"].sum()
    max_points = len(merged) * 3
    rmse = np.sqrt((merged["blad"] ** 2).mean())

    return {
        "config": source_config,
        "source": source,
        "merged": merged,
        "has_data": True,
        "points": points,
        "max_points": max_points,
        "score_percent": points / max_points * 100,
        "rmse": rmse,
        "hours": len(merged),
    }


def print_results(ground_truth, results):
    print("\n" + "=" * 52)
    print("RANKING MODELI (TEMPERATURA)")
    print("=" * 52)
    print("Ground truth: default_model")
    print(f"Zakres default_model: {date_range_text(ground_truth)}\n")

    available_results = [result for result in results if result["has_data"]]
    missing_results = [result for result in results if not result["has_data"]]

    available_results.sort(key=lambda result: (-result["score_percent"], result["rmse"]))

    for index, result in enumerate(available_results, start=1):
        print(f"{index}. {result['config']['name']}:")
        print(
            f"   -> Punkty: {result['points']} / {result['max_points']} "
            f"({result['score_percent']:.1f}%)"
        )
        print(f"   -> RMSE: {result['rmse']:.2f} C (im mniej, tym lepiej)")
        print(f"   -> Wspolne godziny: {result['hours']}")
        print(f"   -> Zakres modelu: {date_range_text(result['source'])}")
        print("-" * 52)

    if missing_results:
        print("Brak wspolnych godzin dla:")
        for result in missing_results:
            print(
                f"   -> {result['config']['name']}: "
                f"{date_range_text(result['source'])}"
            )
        print("-" * 52)

    print("=" * 52)


def plot_results(ground_truth, results):
    print("Generowanie wykresu...")

    plt.figure(figsize=(14, 7))
    plt.plot(
        ground_truth["time"],
        ground_truth["temp_ground_truth"],
        label="Ground truth (default_model)",
        color="black",
        linewidth=3,
    )

    for result in results:
        if not result["has_data"]:
            continue

        config = result["config"]
        source_column = f"temp_{config['key']}"
        plt.plot(
            result["merged"]["time"],
            result["merged"][source_column],
            label=config["name"],
            color=config["color"],
            linestyle=config["linestyle"],
        )

    plt.xticks(ground_truth["time"][::24], rotation=45)
    plt.title("Porownanie prognoz temperatury dla Warszawy", fontsize=16)
    plt.xlabel("Czas", fontsize=12)
    plt.ylabel("Temperatura (C)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    plt.close()

    print(f"Wykres zapisano pomyslnie jako '{OUTPUT_FILE.name}'")


def main():
    print("Wczytywanie i laczenie danych...")

    ground_truth = read_temperature(GROUND_TRUTH_FILE, "ground_truth")
    results = [
        analyze_source(ground_truth, source_config)
        for source_config in MODEL_SOURCES
    ]

    print_results(ground_truth, results)
    plot_results(ground_truth, results)


if __name__ == "__main__":
    main()
