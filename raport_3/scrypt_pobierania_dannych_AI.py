import requests
import pandas as pd
from pathlib import Path

LAT = 52.23
LON = 21.01

DATA_START = "2026-05-10"
DATA_END = "2026-05-13"

MODELS = {
    "graphcast": "gfs_graphcast025",
    "aifs": "ecmwf_aifs025_single"

}

for source_name, model_name in MODELS.items():

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": DATA_START,
        "end_date": DATA_END,
        "hourly": (
            "temperature_2m,"
            "precipitation,"
            "wind_speed_10m"
        ),
        "wind_speed_unit": "ms"
    }

    if model_name is not None:
        params["models"] = model_name

    response = requests.get(
        url,
        params=params
    )

    data = response.json()

    if "hourly" not in data:

        print(f"Błąd dla {source_name}")
        print(data)
        continue

    df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "temp": data["hourly"]["temperature_2m"],
        "opady": data["hourly"]["precipitation"],
        "wiatr": data["hourly"]["wind_speed_10m"]
    })

    out = Path(source_name)
    out.mkdir(exist_ok=True)

    df[["time", "temp"]].to_csv(
        out / "temp.csv",
        index=False
    )

    df[["time", "opady"]].to_csv(
        out / "opady.csv",
        index=False
    )

    df[["time", "wiatr"]].to_csv(
        out / "wiatr.csv",
        index=False
    )

    print(f"Zapisano dane dla: {source_name}")