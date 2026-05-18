from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

GROUND_TRUTH = {
    "name": "Ground truth",
    "key": "ground_truth",
    "path": BASE_DIR / "default_model" / "temp.csv",
    "color": "black",
    "linestyle": "-",
    "linewidth": 3,
}

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
    {
        "name": "GFS",
        "key": "gfs",
        "path": BASE_DIR / "gfs" / "temp.csv",
        "color": "orange",
        "linestyle": "-",
    },
]

OUTPUT_FILE = BASE_DIR / "wykres_temperatura.png"


def read_temperature(path, source_key):
    if not path.exists():
        raise ValueError(f"brak pliku: {path}")

    df = pd.read_csv(path)
    expected_columns = {"time", "temp"}

    if not expected_columns.issubset(df.columns):
        columns = ", ".join(df.columns)
        raise ValueError(f"brak kolumn time,temp w {path}; sa: {columns}")

    value_column = f"temp_{source_key}"
    df = df[["time", "temp"]].rename(columns={"temp": value_column})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    df = df.dropna(subset=["time", value_column])
    df = df.drop_duplicates(subset=["time"]).sort_values("time")

    if df.empty:
        raise ValueError(f"brak poprawnych wierszy w {path}")

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
    try:
        source = read_temperature(source_config["path"], source_config["key"])
    except Exception as error:
        return {
            "config": source_config,
            "has_data": False,
            "reason": str(error),
        }

    merged = ground_truth.merge(source, on="time")
    if merged.empty:
        return {
            "config": source_config,
            "source": source,
            "has_data": False,
            "reason": (
                "brak wspolnych godzin "
                f"(model: {date_range_text(source)}, "
                f"ground truth: {date_range_text(ground_truth)})"
            ),
        }

    source_column = f"temp_{source_config['key']}"
    merged["blad"] = (merged["temp_ground_truth"] - merged[source_column]).abs()
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
    available_results = [result for result in results if result["has_data"]]
    outside_ranking = [
        result
        for result in results
        if not result["has_data"] and "source" in result
    ]
    invalid_results = [
        result
        for result in results
        if not result["has_data"] and "source" not in result
    ]
    available_results.sort(key=lambda result: (-result["score_percent"], result["rmse"]))

    print("\n" + "=" * 52)
    print("RANKING MODELI (TEMPERATURA)")
    print("=" * 52)
    print("Ground truth: default_model")
    print(f"Zakres default_model: {date_range_text(ground_truth)}\n")

    if available_results:
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
    else:
        print("Ranking punktowy: brak wspolnych godzin z default_model.")
        print("-" * 52)

    if outside_ranking:
        print("Na wykresie, poza rankingiem punktowym:")
        for result in outside_ranking:
            print(
                f"   -> {result['config']['name']}: "
                f"{date_range_text(result['source'])}"
            )
        print("-" * 52)

    if invalid_results:
        print("Nie wczytano:")
        for result in invalid_results:
            print(f"   -> {result['config']['name']}: {result['reason']}")
        print("-" * 52)

    print("=" * 52)


def plot_results(ground_truth, results):
    print("Generowanie wykresu...")

    plt.figure(figsize=(14, 7))
    plt.plot(
        ground_truth["time"],
        ground_truth["temp_ground_truth"],
        label="Ground truth (default_model)",
        color=GROUND_TRUTH["color"],
        linestyle=GROUND_TRUTH["linestyle"],
        linewidth=GROUND_TRUTH["linewidth"],
    )

    for result in results:
        if "source" not in result:
            continue

        config = result["config"]
        source_column = f"temp_{config['key']}"
        plt.plot(
            result["source"]["time"],
            result["source"][source_column],
            label=config["name"],
            color=config["color"],
            linestyle=config["linestyle"],
        )

    all_times = [ground_truth["time"]]
    all_times.extend(
        result["source"]["time"]
        for result in results
        if "source" in result
    )
    time_range = pd.concat(all_times)
    ticks = pd.date_range(time_range.min(), time_range.max(), freq="24h")
    plt.xticks(ticks, rotation=45)
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

    try:
        ground_truth = read_temperature(GROUND_TRUTH["path"], "ground_truth")
    except Exception as error:
        print(f"Nie mozna wczytac ground truth: {error}")
        return

    results = [
        analyze_source(ground_truth, source_config)
        for source_config in MODEL_SOURCES
    ]

    print_results(ground_truth, results)
    plot_results(ground_truth, results)


if __name__ == "__main__":
    main()