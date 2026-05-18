from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import argparse
from datetime import date, timedelta
import json

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, default=52.23)
    parser.add_argument("--lon", type=float, default=21.01)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--source",
        choices=["ground_truth", "open_meteo", "yr_no", "all"],
        default="all",
    )
    parser.add_argument("--out-dir", default=BASE_DIR)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frames = {}

    if args.source in ("ground_truth", "all"):
        if args.start_date or args.end_date:
            start_date, end_date = get_ground_truth_dates(
                out_dir,
                args.start_date,
                args.end_date,
                args.days,
            )
            df = read_ground_truth(args.lat, args.lon, start_date, end_date)
        else:
            df = read_open_meteo_forecast(args.lat, args.lon, args.days)
        frames["default_model"] = df

    if args.source in ("open_meteo", "all"):
        frames["open_meteo"] = read_open_meteo_forecast(args.lat, args.lon, args.days)

    if args.source in ("yr_no", "all"):
        frames["yr_no"] = read_yr_no(args.lat, args.lon)

    if len(frames) > 1:
        frames = align_to_common_hours(frames)

    for source_name, df in frames.items():
        write_source(out_dir / source_name, df)
        print(f"Zapisano {source_name} w: {out_dir / source_name}")
        print(f"Zakres {source_name}: {date_range_text(df)}")

    print(f"Zapisano dane w katalogu: {out_dir}")


def get_ground_truth_dates(out_dir, start_date, end_date, days):
    if start_date and end_date:
        return start_date, end_date

    if start_date or end_date:
        raise ValueError("Podaj oba argumenty: --start-date oraz --end-date.")

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    return start.isoformat(), end.isoformat()


def read_ground_truth(lat, lon, start_date, end_date):
    data = fetch_historical_open_meteo(lat, lon, start_date, end_date)
    return hourly_to_frame(data)


def read_open_meteo_forecast(lat, lon, days):
    data = fetch_open_meteo_forecast(lat, lon, days)
    return hourly_to_frame(data)


def hourly_to_frame(data):
    if "hourly" not in data:
        raise ValueError(f"Brak danych hourly w odpowiedzi API: {data}")

    return pd.DataFrame(
        {
            "time": data["hourly"]["time"],
            "temp": data["hourly"]["temperature_2m"],
            "opady": data["hourly"]["precipitation"],
            "wiatr": data["hourly"]["wind_speed_10m"],
        }
    )


def read_yr_no(lat, lon):
    data = fetch_yr_no(lat, lon)
    return pd.DataFrame(
        [
            {
                "time": normalize_time(item["time"]),
                "temp": item["data"]["instant"]["details"]["air_temperature"],
                "opady": item["data"]
                .get("next_1_hours", {})
                .get("details", {})
                .get("precipitation_amount", 0),
                "wiatr": item["data"]["instant"]["details"]["wind_speed"],
            }
            for item in data["properties"]["timeseries"]
        ]
    )


def write_source(source_dir, df):
    source_dir.mkdir(parents=True, exist_ok=True)
    df[["time", "temp"]].to_csv(source_dir / "temp.csv", index=False)
    df[["time", "opady"]].to_csv(source_dir / "opady.csv", index=False)
    df[["time", "wiatr"]].to_csv(source_dir / "wiatr.csv", index=False)


def align_to_common_hours(frames):
    normalized = {
        source_name: normalize_frame_time(df)
        for source_name, df in frames.items()
    }

    common_times = None
    for df in normalized.values():
        times = set(df["time"])
        common_times = times if common_times is None else common_times & times

    if not common_times:
        ranges = "; ".join(
            f"{source_name}: {date_range_text(df)}"
            for source_name, df in normalized.items()
        )
        raise ValueError(f"Brak wspolnych godzin do zapisania. Zakresy: {ranges}")

    common_times = sorted(common_times)
    return {
        source_name: df[df["time"].isin(common_times)].reset_index(drop=True)
        for source_name, df in normalized.items()
    }


def normalize_frame_time(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M")
    return df.drop_duplicates(subset=["time"]).sort_values("time")


def date_range_text(df):
    times = pd.to_datetime(df["time"])
    return f"{times.min():%Y-%m-%d %H:%M} - {times.max():%Y-%m-%d %H:%M}"


def fetch_historical_open_meteo(lat, lon, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "wind_speed_unit": "ms",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    with urlopen(url, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_open_meteo_forecast(lat, lon, days):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "forecast_days": days,
        "wind_speed_unit": "ms",
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)
    with urlopen(url, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_yr_no(lat, lon):
    params = {"lat": lat, "lon": lon}
    url = "https://api.met.no/weatherapi/locationforecast/2.0/compact?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "pti-lab/1.0"})
    with urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_time(value):
    text = str(value).replace("Z", "")
    if "+" in text:
        text = text.split("+", 1)[0]
    return text[:16]


if __name__ == "__main__":
    main()
