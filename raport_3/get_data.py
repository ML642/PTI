import os
import requests
import pandas as pd

def pobierz_i_zapisz(url, nazwa_folderu):
    print(f"🔄 Pobieranie danych dla: {nazwa_folderu}...")
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"❌ Błąd pobierania: {response.status_code}")
        return
        
    dane_json = response.json()
    df = pd.DataFrame(dane_json['hourly'])
    
    df = df.rename(columns={
        'time': 'time', # Залишаємо time, бо так чекає скрипт порівняння
        'temperature_2m': 'temp',
        'precipitation': 'opady',
        'wind_speed_10m': 'wiatr'
    })

    if df['temp'].isnull().all():
        print(f"⚠️ UWAGA: Model {nazwa_folderu} zwrócił tylko puste dane!")
        return
    
    os.makedirs(nazwa_folderu, exist_ok=True)
    
    df[['time', 'temp']].to_csv(os.path.join(nazwa_folderu, 'temp.csv'), index=False)
    df[['time', 'opady']].to_csv(os.path.join(nazwa_folderu, 'opady.csv'), index=False)
    df[['time', 'wiatr']].to_csv(os.path.join(nazwa_folderu, 'wiatr.csv'), index=False)
    
    print(f"✅ Gotowe! Zapisano w '{nazwa_folderu}/'\n")

if __name__ == "__main__":
    LAT, LON = 52.23, 21.01
    
    # Беремо дати, які точно перетинаються з твоїм Open-Meteo
    DATA_START = "2026-05-18"
    DATA_END = "2026-05-24"

    # URL для ECMWF AIFS
    url_aifs = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&start_date={DATA_START}&end_date={DATA_END}&hourly=temperature_2m,precipitation,wind_speed_10m&models=ecmwf_aifs025"
    
    # URL dla Google GraphCast
    url_graphcast = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&start_date={DATA_START}&end_date={DATA_END}&hourly=temperature_2m,precipitation,wind_speed_10m&models=gfs_graphcast025"

    print("="*40)
    print("🤖 POBIERANIE DANYCH AI (WARSZAWA)")
    print("="*40)
    
    pobierz_i_zapisz(url_aifs, "aifs")
    pobierz_i_zapisz(url_graphcast, "graphcast")