from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "raport_3"
GROUND_TRUTH_DIR = DATA_DIR / "default_model"


@dataclass(frozen=True)
class Source:
    name: str
    key: str
    directory: Path


SOURCES = [
    Source("AIFS", "aifs", DATA_DIR / "aifs"),
    Source("GraphCast", "graphcast", DATA_DIR / "graphcast"),
    Source("Open-Meteo", "open_meteo", DATA_DIR / "open_meteo"),
    Source("Yr.no", "yr_no", DATA_DIR / "yr_no"),
    Source("GFS", "gfs", DATA_DIR / "gfs"),
    Source("Nasz model", "own_model", BASE_DIR / "own_model"),
]


METRIC_FILES = {
    "temp": "temp.csv",
    "opady": "opady.csv",
    "wiatr": "wiatr.csv",
}


TEMP_LABELS = ["mroz", "zimno", "chlodno", "komfort", "cieplo", "upal"]
RAIN_LABELS = ["brak_opadu", "slaby_opad", "umiarkowany_opad", "silny_opad"]
WIND_LABELS = ["cisza", "lekki_wiatr", "wietrznie", "silny_wiatr"]
DECISION_LABELS = ["zle", "ostroznie", "dobre"]

CRITICAL_EVENTS = {
    "freezing_boundary": "przejscie przez 0 C",
    "snow_or_ice_risk": "opad przy temperaturze bliskiej 0 C",
    "significant_precipitation": "opad istotny",
    "heavy_precipitation": "opad silny",
    "strong_wind": "wiatr silny",
    "thermal_extreme": "temperatura skrajna",
}


def read_metric(path: Path, metric: str, source_key: str) -> pd.DataFrame:
    """Read a CSV metric file and ignore unresolved git conflict markers."""
    if not path.exists():
        raise FileNotFoundError(path)

    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    clean_lines = [
        line
        for line in raw_lines
        if not line.strip().startswith(("<<<<<<<", "=======", ">>>>>>>"))
    ]
    df = pd.read_csv(StringIO("\n".join(clean_lines)))
    expected = {"time", metric}
    if not expected.issubset(df.columns):
        raise ValueError(f"{path}: expected columns time,{metric}; got {list(df.columns)}")

    value_col = f"{metric}_{source_key}"
    df = df[["time", metric]].rename(columns={metric: value_col})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["time", value_col])
    df = df.drop_duplicates(subset=["time"]).sort_values("time")

    if df.empty:
        raise ValueError(f"{path}: no valid rows")
    return df


def read_source(source_dir: Path, source_key: str) -> pd.DataFrame:
    frames = []
    for metric, file_name in METRIC_FILES.items():
        frames.append(read_metric(source_dir / file_name, metric, source_key))

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="time", how="inner")

    if merged.empty:
        raise ValueError(f"{source_dir}: no common timestamps for temp/opady/wiatr")
    return merged


def temp_label(value: float) -> str:
    if value < 0:
        return "mroz"
    if value < 10:
        return "zimno"
    if value < 18:
        return "chlodno"
    if value < 25:
        return "komfort"
    if value < 30:
        return "cieplo"
    return "upal"


def rain_label(value: float) -> str:
    if value <= 0.1:
        return "brak_opadu"
    if value <= 0.5:
        return "slaby_opad"
    if value <= 2.0:
        return "umiarkowany_opad"
    return "silny_opad"


def wind_label(value: float) -> str:
    if value < 2.0:
        return "cisza"
    if value < 5.0:
        return "lekki_wiatr"
    if value < 8.0:
        return "wietrznie"
    return "silny_wiatr"


def decision_label(temp: str, rain: str, wind: str) -> str:
    if temp in {"mroz", "upal"} or rain in {"umiarkowany_opad", "silny_opad"}:
        return "zle"
    if wind == "silny_wiatr":
        return "zle"
    if temp in {"zimno"} or rain == "slaby_opad" or wind == "wietrznie":
        return "ostroznie"
    return "dobre"


def add_semantic_columns(df: pd.DataFrame, source_key: str) -> pd.DataFrame:
    result = df.copy()
    temp_col = f"temp_{source_key}"
    rain_col = f"opady_{source_key}"
    wind_col = f"wiatr_{source_key}"

    result[f"temp_label_{source_key}"] = result[temp_col].apply(temp_label)
    result[f"rain_label_{source_key}"] = result[rain_col].apply(rain_label)
    result[f"wind_label_{source_key}"] = result[wind_col].apply(wind_label)
    result[f"decision_{source_key}"] = result.apply(
        lambda row: decision_label(
            row[f"temp_label_{source_key}"],
            row[f"rain_label_{source_key}"],
            row[f"wind_label_{source_key}"],
        ),
        axis=1,
    )
    return result


def ordinal_distance(value_a: str, value_b: str, labels: list[str]) -> float:
    if len(labels) <= 1:
        return 0.0
    return abs(labels.index(value_a) - labels.index(value_b)) / (len(labels) - 1)


def semantic_similarity(row: pd.Series, source_key: str) -> float:
    temp_dist = ordinal_distance(
        row[f"temp_label_ground_truth"],
        row[f"temp_label_{source_key}"],
        TEMP_LABELS,
    )
    rain_dist = ordinal_distance(
        row[f"rain_label_ground_truth"],
        row[f"rain_label_{source_key}"],
        RAIN_LABELS,
    )
    wind_dist = ordinal_distance(
        row[f"wind_label_ground_truth"],
        row[f"wind_label_{source_key}"],
        WIND_LABELS,
    )
    weighted_distance = 0.25 * temp_dist + 0.45 * rain_dist + 0.30 * wind_dist
    return 1.0 - weighted_distance


def critical_flags(df: pd.DataFrame, source_key: str) -> pd.DataFrame:
    temp = df[f"temp_{source_key}"]
    rain = df[f"opady_{source_key}"]
    wind = df[f"wiatr_{source_key}"]

    return pd.DataFrame(
        {
            "freezing_boundary": temp <= 0.0,
            "snow_or_ice_risk": (temp <= 0.5) & (rain > 0.1),
            "significant_precipitation": rain > 0.5,
            "heavy_precipitation": rain > 2.0,
            "strong_wind": wind >= 8.0,
            "thermal_extreme": (temp <= -5.0) | (temp >= 30.0),
        },
        index=df.index,
    )


def critical_mismatch_summary(df: pd.DataFrame, source_key: str) -> dict[str, int | float]:
    actual_flags = critical_flags(df, "ground_truth")
    predicted_flags = critical_flags(df, source_key)
    mismatches = actual_flags != predicted_flags

    any_actual = actual_flags.any(axis=1)
    any_predicted = predicted_flags.any(axis=1)
    any_mismatch = mismatches.any(axis=1)

    summary: dict[str, int | float] = {
        "critical_reference_cases": int(any_actual.sum()),
        "critical_predicted_cases": int(any_predicted.sum()),
        "critical_mismatch_cases": int(any_mismatch.sum()),
        "critical_mismatch_percent": float(any_mismatch.mean() * 100),
    }

    for event_key in CRITICAL_EVENTS:
        summary[f"{event_key}_mismatches"] = int(mismatches[event_key].sum())

    return summary


def critical_mismatch_rows(df: pd.DataFrame, source_key: str) -> pd.DataFrame:
    actual_flags = critical_flags(df, "ground_truth")
    predicted_flags = critical_flags(df, source_key)
    mismatches = actual_flags != predicted_flags
    rows = []

    for event_key, event_name in CRITICAL_EVENTS.items():
        event_rows = df.loc[mismatches[event_key]].copy()
        for index, row in event_rows.iterrows():
            rows.append(
                {
                    "time": row["time"],
                    "event": event_key,
                    "event_name": event_name,
                    "model": source_key,
                    "actual_event": bool(actual_flags.loc[index, event_key]),
                    "predicted_event": bool(predicted_flags.loc[index, event_key]),
                    "temp_ground_truth": row["temp_ground_truth"],
                    f"temp_{source_key}": row[f"temp_{source_key}"],
                    "opady_ground_truth": row["opady_ground_truth"],
                    f"opady_{source_key}": row[f"opady_{source_key}"],
                    "wiatr_ground_truth": row["wiatr_ground_truth"],
                    f"wiatr_{source_key}": row[f"wiatr_{source_key}"],
                }
            )

    return pd.DataFrame(rows)


def binary_metrics(predicted: pd.Series, actual: pd.Series) -> dict[str, float]:
    pred = predicted.astype(bool)
    act = actual.astype(bool)
    tp = int((pred & act).sum())
    fp = int((pred & ~act).sum())
    fn = int((~pred & act).sum())
    tn = int((~pred & ~act).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    success_rate = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "success_rate": success_rate,
    }


QueryFn = Callable[[pd.DataFrame, str], pd.Series]


def query_good_outdoor(df: pd.DataFrame, key: str) -> pd.Series:
    return df[f"decision_{key}"] == "dobre"


def query_rain_risk(df: pd.DataFrame, key: str) -> pd.Series:
    return df[f"rain_label_{key}"].isin(["umiarkowany_opad", "silny_opad"])


def query_wind_risk(df: pd.DataFrame, key: str) -> pd.Series:
    return df[f"wind_label_{key}"].isin(["wietrznie", "silny_wiatr"])


def query_bad_conditions(df: pd.DataFrame, key: str) -> pd.Series:
    return df[f"decision_{key}"] == "zle"


def query_thermal_comfort(df: pd.DataFrame, key: str) -> pd.Series:
    return df[f"temp_label_{key}"].isin(["chlodno", "komfort", "cieplo"])


SEARCH_QUERIES: list[tuple[str, str, QueryFn]] = [
    ("Q1", "dobre okno na aktywnosc zewnetrzna", query_good_outdoor),
    ("Q2", "ryzyko opadow istotnych", query_rain_risk),
    ("Q3", "ryzyko wiatru dla uzytkownika", query_wind_risk),
    ("Q4", "warunki niekorzystne", query_bad_conditions),
    ("Q5", "komfort termiczny", query_thermal_comfort),
]
