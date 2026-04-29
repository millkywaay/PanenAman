import os
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

START_DATE = "2018-01-01"
END_DATE   = "2025-12-31"
CACHE_DIR  = "cache_cuaca"
OUTPUT_HARIAN  = "cuaca_harian_jatim_raw.csv"
OUTPUT_TAHUNAN = "cuaca_tahunan_jatim.csv"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "et0_fao_evapotranspiration",
    "windspeed_10m_max",
    "shortwave_radiation_sum",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
]

KABUPATEN = [
    {"nama": "Pacitan",          "lat": -8.1845,  "lon": 111.1014},
    {"nama": "Ponorogo",         "lat": -7.8665,  "lon": 111.4639},
    {"nama": "Trenggalek",       "lat": -8.0490,  "lon": 111.7087},
    {"nama": "Tulungagung",      "lat": -8.0653,  "lon": 111.9029},
    {"nama": "Blitar",           "lat": -8.0980,  "lon": 112.1615},
    {"nama": "Kediri",           "lat": -7.8167,  "lon": 111.9667},
    {"nama": "Malang",           "lat": -8.1583,  "lon": 112.6033},
    {"nama": "Lumajang",         "lat": -8.1325,  "lon": 113.2231},
    {"nama": "Jember",           "lat": -8.1724,  "lon": 113.7020},
    {"nama": "Banyuwangi",       "lat": -8.2192,  "lon": 114.3691},
    {"nama": "Bondowoso",        "lat": -7.9107,  "lon": 113.8226},
    {"nama": "Situbondo",        "lat": -7.7059,  "lon": 114.0082},
    {"nama": "Probolinggo",      "lat": -7.7543,  "lon": 113.2159},
    {"nama": "Pasuruan",         "lat": -7.6453,  "lon": 112.8990},
    {"nama": "Sidoarjo",         "lat": -7.4459,  "lon": 112.7184},
    {"nama": "Mojokerto",        "lat": -7.5271,  "lon": 112.4346},
    {"nama": "Jombang",          "lat": -7.5500,  "lon": 112.2333},
    {"nama": "Nganjuk",          "lat": -7.6044,  "lon": 111.9013},
    {"nama": "Madiun",           "lat": -7.6298,  "lon": 111.5239},
    {"nama": "Magetan",          "lat": -7.6551,  "lon": 111.3288},
    {"nama": "Ngawi",            "lat": -7.4008,  "lon": 111.4492},
    {"nama": "Bojonegoro",       "lat": -7.1507,  "lon": 111.8815},
    {"nama": "Tuban",            "lat": -6.8990,  "lon": 112.0508},
    {"nama": "Lamongan",         "lat": -7.1175,  "lon": 112.4154},
    {"nama": "Gresik",           "lat": -7.1569,  "lon": 112.6552},
    {"nama": "Bangkalan",        "lat": -6.9046,  "lon": 112.9834},
    {"nama": "Sampang",          "lat": -7.1936,  "lon": 113.2489},
    {"nama": "Pamekasan",        "lat": -7.1575,  "lon": 113.4742},
    {"nama": "Sumenep",          "lat": -6.9924,  "lon": 113.8607},
    {"nama": "Kota Kediri",      "lat": -7.8480,  "lon": 112.0172},
    {"nama": "Kota Blitar",      "lat": -8.0953,  "lon": 112.1608},
    {"nama": "Kota Malang",      "lat": -7.9797,  "lon": 112.6304},
    {"nama": "Kota Probolinggo", "lat": -7.7543,  "lon": 113.2159},
    {"nama": "Kota Pasuruan",    "lat": -7.6453,  "lon": 112.8990},
    {"nama": "Kota Mojokerto",   "lat": -7.4706,  "lon": 112.4344},
    {"nama": "Kota Madiun",      "lat": -7.6298,  "lon": 111.5239},
    {"nama": "Kota Surabaya",    "lat": -7.2575,  "lon": 112.7521},
    {"nama": "Kota Batu",        "lat": -7.8680,  "lon": 112.5239},
]

def fetch_satu_kabupaten(nama, lat, lon, retries=5, base_delay=10):
    """
    Fetch data harian dari Open-Meteo Archive API untuk satu kabupaten.
    Pakai endpoint /v1/archive (historical data, gratis tanpa API key).
    Exponential backoff untuk handle 429 rate limit.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "daily":      ",".join(DAILY_VARS),
        "timezone":   "Asia/Jakarta",
    }

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)

            # Handle 429 secara eksplisit
            if resp.status_code == 429:
                wait = base_delay * (2 ** attempt)  # 10, 20, 40, 80, 160 detik
                tqdm.write(f"    Rate limit (429) — tunggu {wait}s lalu retry...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            df = pd.DataFrame(data["daily"])
            df.rename(columns={"time": "tanggal"}, inplace=True)
            df["tanggal"]   = pd.to_datetime(df["tanggal"])
            df["kabupaten"] = nama
            df["lat"]       = lat
            df["lon"]       = lon
            return df

        except requests.exceptions.HTTPError as e:
            tqdm.write(f"    HTTP error attempt {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                wait = base_delay * (2 ** attempt)
                time.sleep(wait)
        except Exception as e:
            tqdm.write(f"    Error attempt {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(base_delay)

    tqdm.write(f"  ❌ GAGAL fetch {nama} setelah {retries} percobaan")
    return None


def hitung_run_konsekutif(series, threshold, min_run=2):
    """
    Hitung berapa kali ada run >= min_run hari berturut-turut di atas threshold.
    Contoh: curah hujan > 50mm selama 3 hari berturut = 1 kejadian.
    """
    s = (series > threshold).astype(int)
    count = 0
    run = 0
    for v in s:
        if v:
            run += 1
            if run == min_run:
                count += 1
        else:
            run = 0
    return count

def hitung_maks_run(series, threshold):
    """Panjang run berturut-turut terpanjang di atas threshold."""
    s = (series > threshold).astype(int)
    max_run = 0
    run = 0
    for v in s:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run

def agregat_tahunan(df_harian):
    """
    Dari data harian, hasilkan 1 baris per kabupaten per tahun
    dengan fitur-fitur yang relevan untuk prediksi produktivitas padi.
    """
    df = df_harian.copy()
    df["tahun"] = df["tanggal"].dt.year

    # Threshold ekstrim
    HUJAN_EKSTRIM   = 50   # mm/hari (BMKG: sangat lebat)
    HUJAN_KERING    = 1    # mm/hari (hari kering)
    SUHU_PANAS      = 35   # °C
    SUHU_DINGIN     = 20   # °C

    hasil = []
    for (kab, thn), g in df.groupby(["kabupaten", "tahun"]):
        row = {"kabupaten": kab, "tahun": thn}

        # -- Suhu --
        row["suhu_mean"]     = g["temperature_2m_mean"].mean()
        row["suhu_max_mean"] = g["temperature_2m_max"].mean()
        row["suhu_min_mean"] = g["temperature_2m_min"].mean()
        row["n_hari_panas"]  = (g["temperature_2m_max"] > SUHU_PANAS).sum()
        row["n_hari_dingin"] = (g["temperature_2m_min"] < SUHU_DINGIN).sum()

        # -- Curah hujan --
        row["hujan_total"]       = g["precipitation_sum"].sum()
        row["hujan_mean_harian"] = g["precipitation_sum"].mean()
        row["n_hari_hujan"]      = (g["precipitation_sum"] > HUJAN_KERING).sum()
        row["n_hari_kering"]     = (g["precipitation_sum"] <= HUJAN_KERING).sum()
        row["n_hari_ekstrim"]    = (g["precipitation_sum"] > HUJAN_EKSTRIM).sum()
        row["hujan_max_harian"]  = g["precipitation_sum"].max()

        # -- Run berturut-turut (fitur kunci dari diskusi tim) --
        row["n_run2_hujan_ekstrim"] = hitung_run_konsekutif(g["precipitation_sum"], HUJAN_EKSTRIM, 2)
        row["n_run3_hujan_ekstrim"] = hitung_run_konsekutif(g["precipitation_sum"], HUJAN_EKSTRIM, 3)
        row["maks_run_hujan_ekstrim"] = hitung_maks_run(g["precipitation_sum"], HUJAN_EKSTRIM)

        row["n_run3_kering"]      = hitung_run_konsekutif(g["precipitation_sum"], HUJAN_KERING, 3)
        row["n_run7_kering"]      = hitung_run_konsekutif(g["precipitation_sum"], HUJAN_KERING, 7)
        row["maks_run_kering"]    = hitung_maks_run(g["precipitation_sum"], HUJAN_KERING)

        # -- Evapotranspirasi (stress air) --
        row["et0_total"]  = g["et0_fao_evapotranspiration"].sum()
        row["et0_mean"]   = g["et0_fao_evapotranspiration"].mean()
        # Neraca air: hujan - ET (positif = surplus, negatif = defisit)
        row["neraca_air"] = g["precipitation_sum"].sum() - g["et0_fao_evapotranspiration"].sum()

        # -- Radiasi matahari --
        row["radiasi_mean"] = g["shortwave_radiation_sum"].mean()

        # -- Kelembaban --
        row["rh_max_mean"] = g["relative_humidity_2m_max"].mean()
        row["rh_min_mean"] = g["relative_humidity_2m_min"].mean()

        # -- Angin --
        row["angin_max_mean"] = g["windspeed_10m_max"].mean()
        row["n_hari_angin_kencang"] = (g["windspeed_10m_max"] > 40).sum()

        hasil.append(row)

    return pd.DataFrame(hasil)

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 60)
    print("FETCH CUACA HARIAN OPEN-METEO — JAWA TIMUR 2018–2025")
    print("=" * 60)
    print(f"Jumlah kabupaten/kota : {len(KABUPATEN)}")
    print(f"Periode               : {START_DATE} s/d {END_DATE}")
    print(f"Cache folder          : {CACHE_DIR}/")
    print()

    semua_harian = []
    gagal = []

    for kab in tqdm(KABUPATEN, desc="Fetching"):
        nama = kab["nama"]
        cache_path = os.path.join(CACHE_DIR, f"{nama.replace(' ', '_')}.csv")

        # Cek cache dulu
        if os.path.exists(cache_path):
            df_kab = pd.read_csv(cache_path, parse_dates=["tanggal"])
            tqdm.write(f"  [cache] {nama}")
        else:
            tqdm.write(f"  [fetch] {nama} ...")
            df_kab = fetch_satu_kabupaten(nama, kab["lat"], kab["lon"])

            if df_kab is not None:
                df_kab.to_csv(cache_path, index=False)
                tqdm.write(f"         ✅ {len(df_kab)} baris disimpan")
                time.sleep(3) 
            else:
                gagal.append(nama)
                continue

        semua_harian.append(df_kab)

    if not semua_harian:
        print("\n❌ Tidak ada data berhasil di-fetch.")
        return

    print("\nMenggabungkan data harian...")
    df_harian = pd.concat(semua_harian, ignore_index=True)
    df_harian = df_harian.sort_values(["kabupaten", "tanggal"]).reset_index(drop=True)
    df_harian.to_csv(OUTPUT_HARIAN, index=False)
    print(f"✅ Data harian: {len(df_harian):,} baris → {OUTPUT_HARIAN}")

    print("\nMenghitung fitur tahunan...")
    df_tahunan = agregat_tahunan(df_harian)
    df_tahunan = df_tahunan.sort_values(["kabupaten", "tahun"]).reset_index(drop=True)
    df_tahunan.to_csv(OUTPUT_TAHUNAN, index=False)
    print(f"✅ Data tahunan: {len(df_tahunan):,} baris → {OUTPUT_TAHUNAN}")

    print("\n" + "=" * 60)
    print("SELESAI!")
    print(f"  Data harian : {OUTPUT_HARIAN}  ({len(df_harian):,} baris)")
    print(f"  Data tahunan: {OUTPUT_TAHUNAN}  ({len(df_tahunan):,} baris)")
    if gagal:
        print(f"\n  ⚠ Gagal fetch: {', '.join(gagal)}")
        print("    → Jalankan script lagi, kabupaten yang gagal akan di-retry.")
    print("=" * 60)

    print("\nPreview cuaca_tahunan_jatim.csv (5 baris pertama):")
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(df_tahunan.head())


if __name__ == "__main__":
    main()
