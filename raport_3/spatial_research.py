from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

BASE_DIR = Path(__file__).resolve().parent

# ==========================================================
# 1. KONFIGURACJA ETAPU II (REALNE SATELLITE TOWNS DLA AIFS)
# ==========================================================
MODEL_KEY = "ecmwf_aifs025"
TARGET_DIR = BASE_DIR / "default_model"  
TARGET_VAR = "temp"

# Rozsuwamy sieć na większe miasta Mazowsza, aby trafić w różne rastery AI (0.25°)
SPATIAL_POINTS = {
    "aifs_warszawa_centrum": {"lat": 52.23, "lon": 21.01},
    "aifs_plock_zachod":     {"lat": 52.54, "lon": 19.70},
    "aifs_ostrolenka_polnoc": {"lat": 53.08, "lon": 21.57},
    "aifs_radom_poludnie":   {"lat": 51.40, "lon": 21.14},
    "aifs_sedlce_wschod":    {"lat": 52.17, "lon": 22.28}
}

def fetch_spatial_point(nazwa, lat, lon, start_date, end_date):
    """Pobiera dane z API Open-Meteo dla konkretnego punktu geograficznego."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "wind_speed_unit": "ms",
        "models": MODEL_KEY,
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)
    
    with urlopen(url, timeout=45) as response:
        data = json.loads(response.read().decode("utf-8"))
        
    df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "temp": data["hourly"]["temperature_2m"],
    })
    
    folder = BASE_DIR / nazwa
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f"{TARGET_VAR}.csv", index=False)


def main():
    print("=" * 60)
    print("🌲 ETAP II: SPATIAL & TEMPORAL FEATURE ENGINEERING")
    print("=" * 60)
    
    if not (TARGET_DIR / f"{TARGET_VAR}.csv").exists():
        print(f"❌ Blad: Brak pliku etalonu w {TARGET_DIR}.")
        return
        
    df_target = pd.read_csv(TARGET_DIR / f"{TARGET_VAR}.csv")
    df_target["time"] = pd.to_datetime(df_target["time"])
    
    start_date = df_target["time"].min().strftime("%Y-%m-%d")
    end_date = df_target["time"].max().strftime("%Y-%m-%d")
    
    print(f"🎯 Synchronizacja czasu z Etapem I: {start_date} do {end_date}")
    
    # 2. POBIERANIE NOWEJ SIATKI GEOGRAFICZNEJ
    for nazwa, coords in SPATIAL_POINTS.items():
        print(f"🔄 Pobieranie zróżnicowanych danych dla: {nazwa}...")
        try:
            fetch_spatial_point(nazwa, coords["lat"], coords["lon"], start_date, end_date)
        except Exception as e:
            print(f"❌ Blad pobierania: {e}")
            return

    # 3. KONSOLIDACJA I INŻYNIERIA CECH (FEATURE ENGINEERING)
    print("-" * 60)
    print("🔄 Budowanie zaawansowanej macierzy cech...")
    
    df_target_renamed = df_target[["time", "temp"]].rename(columns={"temp": "target"})
    df_features = df_target_renamed[["time"]].copy()
    
    for punkt in SPATIAL_POINTS.keys():
        df_p = pd.read_csv(BASE_DIR / punkt / f"{TARGET_VAR}.csv")
        df_p["time"] = pd.to_datetime(df_p["time"])
        df_p = df_p[["time", "temp"]].rename(columns={"temp": punkt})
        df_features = df_features.merge(df_p, on="time", how="inner")
        
    df_ml = df_features.merge(df_target_renamed, on="time", how="inner")
    
    # --- INŻYNIERIA CECH CZASOWYCH (Dodajemy dynamikę dobową) ---
    df_ml["hour"] = df_ml["time"].dt.hour
    df_ml["day_of_week"] = df_ml["time"].dt.dayofweek
    
    # 4. TRENING MODELU RANDOM FOREST
    print("🧠 Trening przestrzenno-czasowego modelu Random Forest...")
    spatial_cols = list(SPATIAL_POINTS.keys())
    feature_columns = spatial_cols + ["hour", "day_of_week"]
    
    X = df_ml[feature_columns]
    y = df_ml["target"]
    
    # Zwiększamy parametry, wyłączamy ograniczenia
    rf = RandomForestRegressor(
        n_estimators=200, 
        max_depth=12,           # Duża głębokość pozwoli idealnie powtórzyć krzywą dobową
        min_samples_split=2,
        random_state=42
    )
    rf.fit(X, y)
    
    # Obliczamy dopasowanie na zbiorze treningowym (Residua)
    predictions = rf.predict(X)
    rmse_train = root_mean_squared_error(y, predictions)
    print(f"✅ Model wytrenowany dynamicznie! Wyjściowe RMSE: {rmse_train:.2f} °C")
    
    # 5. ANALIZA ISTOTNOŚCI CECH GEOGRAFICZNYCH
    print("\n📊 WPŁYW CECH NA SUPRE-PROGNOZĘ:")
    importances = rf.feature_importances_
    for nazwa, waga in zip(feature_columns, importances):
        print(f"   -> {nazwa:25}: {waga*100:.1f}%")
        
    # 6. GENEROWANIE I ZAPIS PROGNOZY
    df_ml["temp"] = predictions
    
    output_dir = BASE_DIR / "random_forest_spatial"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_ml[["time", "temp"]].to_csv(output_dir / "temp.csv", index=False)
    
    print("-" * 60)
    print(f"🚀 Sukces! Dynamiczny model ML został zapisany!")
    print("=" * 60)


if __name__ == "__main__":
    main()