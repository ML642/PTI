from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import argparse
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
        default="ground_truth",
    )
    parser.add_argument("--out-dir", default=BASE_DIR)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.source in ("ground_truth", "all"):
        start_date, end_date = get_ground_truth_dates(
            out_dir,
            args.start_date,
            args.end_date,
        )
        df = read_ground_truth(args.lat, args.lon, start_date, end_date)
        write_source(out_dir / "default_model", df)
        print(f"Zapisano ground truth w: {out_dir / 'default_model'}")
        print(f"Zakres ground truth: {start_date} - {end_date}")

    if args.source in ("open_meteo", "all"):
        write_source(
            out_dir / "open_meteo",
            read_open_meteo_forecast(args.lat, args.lon, args.days),
        )

    if args.source in ("yr_no", "all"):
        write_source(out_dir / "yr_no", read_yr_no(args.lat, args.lon))

    print(f"Zapisano dane w katalogu: {out_dir}")


def get_ground_truth_dates(out_dir, start_date, end_date):
    if start_date and end_date:
        return start_date, end_date

    detected_start, detected_end = detect_common_model_date_range(out_dir)
    return start_date or detected_start, end_date or detected_end


def detect_common_model_date_range(out_dir):
    model_files = [
        out_dir / "aifs" / "temp.csv",
        out_dir / "graphcast" / "temp.csv",
    ]

    ranges = []
    for path in model_files:
        if not path.exists():
            continue

        df = pd.read_csv(path)
        if "time" not in df.columns or df.empty:
            continue

        times = pd.to_datetime(df["time"])
        ranges.append((times.min(), times.max()))

    if not ranges:
        raise ValueError(
            "Nie podano --start-date/--end-date i nie znaleziono danych modeli "
            "w aifs/temp.csv oraz graphcast/temp.csv."
        )

    start = max(item[0] for item in ranges)
    end = min(item[1] for item in ranges)

    if start > end:
        raise ValueError(
            "Pliki modeli nie maja wspolnego zakresu dat. Podaj recznie "
            "--start-date i --end-date."
        )

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


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
