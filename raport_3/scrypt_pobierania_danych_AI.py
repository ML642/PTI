from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
REFERENCE_FILE = BASE_DIR / "default_model" / "temp.csv"

LAT = 52.23
LON = 21.01

MODELS = {
    "graphcast": {
        "url": "https://api.open-meteo.com/v1/gfs",
        "model": "gfs_graphcast025",
    },
    "aifs": {
        "url": "https://api.open-meteo.com/v1/ecmwf",
        "model": "ecmwf_aifs025_single",
    },
}


def main():
    reference_times = read_reference_times()
    start_date = reference_times.min().strftime("%Y-%m-%d")
    end_date = reference_times.max().strftime("%Y-%m-%d")
    allowed_times = set(reference_times.dt.strftime("%Y-%m-%dT%H:%M"))

    for source_name, model_config in MODELS.items():
        df = fetch_model(model_config, start_date, end_date)
        df = filter_to_reference_hours(df, allowed_times)

        if df.empty:
            print(f"Brak wspolnych godzin dla: {source_name}")
            continue

        write_source(source_name, df)
        print(f"Zapisano dane dla: {source_name} ({date_range_text(df)})")


def read_reference_times():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Brak {REFERENCE_FILE}. Najpierw uruchom: python scrypty.py"
        )

    df = pd.read_csv(REFERENCE_FILE)
    if "time" not in df.columns or df.empty:
        raise ValueError(f"Brak poprawnej kolumny time w {REFERENCE_FILE}")

    times = pd.to_datetime(df["time"], errors="coerce").dropna()
    if times.empty:
        raise ValueError(f"Brak poprawnych godzin w {REFERENCE_FILE}")

    return times


def fetch_model(model_config, start_date, end_date):
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "wind_speed_unit": "ms",
        "models": model_config["model"],
    }

    response = requests.get(model_config["url"], params=params, timeout=45)
    if not response.ok:
        raise requests.HTTPError(
            f"{response.status_code} dla {model_config['model']}: {response.text}",
            response=response,
        )

    data = response.json()
    if "hourly" not in data:
        raise ValueError(f"Brak danych hourly dla modelu {model_config['model']}: {data}")

    return pd.DataFrame(
        {
            "time": data["hourly"]["time"],
            "temp": data["hourly"]["temperature_2m"],
            "opady": data["hourly"]["precipitation"],
            "wiatr": data["hourly"]["wind_speed_10m"],
        }
    )


def filter_to_reference_hours(df, allowed_times):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M")
    df = df[df["time"].isin(allowed_times)]
    return df.drop_duplicates(subset=["time"]).sort_values("time")


def write_source(source_name, df):
    out = BASE_DIR / source_name
    out.mkdir(exist_ok=True)

    df[["time", "temp"]].to_csv(out / "temp.csv", index=False)
    df[["time", "opady"]].to_csv(out / "opady.csv", index=False)
    df[["time", "wiatr"]].to_csv(out / "wiatr.csv", index=False)


def date_range_text(df):
    times = pd.to_datetime(df["time"])
    return f"{times.min():%Y-%m-%d %H:%M} - {times.max():%Y-%m-%d %H:%M}"


if __name__ == "__main__":
    main()