from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import argparse
import json

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, default=52.23)
    parser.add_argument("--lon", type=float, default=21.01)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--source", choices=["all", "open_meteo", "yr_no"], default="all")
    parser.add_argument("--out-dir", default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.source in ("all", "open_meteo"):
        write_source(out_dir / "open_meteo", read_open_meteo(args.lat, args.lon, args.days))

    if args.source in ("all", "yr_no"):
        write_source(out_dir / "yr_no", read_yr_no(args.lat, args.lon))

    print(f"Zapisano dane w katalogu: {out_dir}")


def read_open_meteo(lat, lon, days):
    data = fetch_open_meteo(lat, lon, days)
    return pd.DataFrame({
        "time": data["hourly"]["time"],
        "temp": data["hourly"]["temperature_2m"],
        "opady": data["hourly"]["precipitation"],
        "wiatr": data["hourly"]["wind_speed_10m"],
    })


def read_yr_no(lat, lon):
    data = fetch_yr_no(lat, lon)
    return pd.DataFrame([{
        "time": normalize_time(item["time"]),
        "temp": item["data"]["instant"]["details"]["air_temperature"],
        "opady": item["data"].get("next_1_hours", {}).get("details", {}).get("precipitation_amount", 0),
        "wiatr": item["data"]["instant"]["details"]["wind_speed"],
    } for item in data["properties"]["timeseries"]])


def write_source(source_dir, df):
    source_dir.mkdir(parents=True, exist_ok=True)
    df[["time", "temp"]].to_csv(source_dir / "temp.csv", index=False)
    df[["time", "opady"]].to_csv(source_dir / "opady.csv", index=False)
    df[["time", "wiatr"]].to_csv(source_dir / "wiatr.csv", index=False)


def fetch_open_meteo(lat, lon, days):
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
