import pandas as pd
import numpy as np
import os
import re

KOTA_LUAS_RUSAK = ["Kota Blitar", "Kota Mojokerto", "Kota Batu"]

def parse_angka_indonesia(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if "," in val:
        val = val.replace(".", "").replace(",", ".")
    return float(val)

def clean_bps_file(filepath, tahun=None):
    if tahun is None:
        match = re.search(r"20\d{2}", os.path.basename(filepath))
        tahun = int(match.group()) if match else np.nan

    df = pd.read_csv(filepath, dtype=str)
    col_map = {
        df.columns[0]: "kabupaten",
        df.columns[1]: "luas_panen_raw",
        df.columns[2]: "produktivitas_raw",
        df.columns[3]: "produksi_raw",
    }
    df = df.rename(columns=col_map)

    df = df[~df["kabupaten"].str.strip().str.lower().eq("jawa timur")].copy()
    df = df.dropna(subset=["kabupaten"])
    df = df[df["kabupaten"].str.strip() != ""]
    df["kabupaten"] = df["kabupaten"].str.strip()

    df["luas_panen_raw"]    = df["luas_panen_raw"].apply(parse_angka_indonesia)
    df["produktivitas_raw"] = df["produktivitas_raw"].apply(parse_angka_indonesia)
    df["produksi_raw"]      = df["produksi_raw"].apply(parse_angka_indonesia)

    # Fix kota yang hilang desimalnya
    mask_rusak = df["kabupaten"].isin(KOTA_LUAS_RUSAK) & (df["luas_panen_raw"] > 10000)
    if mask_rusak.any():
        print(f"  Fix luas hilang desimal: {df.loc[mask_rusak, 'kabupaten'].tolist()}")
        df.loc[mask_rusak, "luas_panen_raw"] = df.loc[mask_rusak, "luas_panen_raw"] / 100_000

    mask_kota_umum = (
        df["kabupaten"].str.startswith("Kota") &
        (df["luas_panen_raw"] > 10000) & (df["produksi_raw"] < 100)
    )
    if mask_kota_umum.any():
        print(f"  Fix heuristik kota lain: {df.loc[mask_kota_umum, 'kabupaten'].tolist()}")
        df.loc[mask_kota_umum, "luas_panen_raw"] = df.loc[mask_kota_umum, "luas_panen_raw"] / 100_000

    # Konversi ke satuan riil
    df["luas_panen_ha"]        = df["luas_panen_raw"] * 1000
    df["produktivitas_ku_ha"]  = df["produktivitas_raw"] / 100
    df["produktivitas_ton_ha"] = df["produktivitas_ku_ha"] / 10
    df["produksi_ton"]         = df["produksi_raw"] * 1000
    df["tahun"]                = tahun

    # Validasi
    df["produksi_hitung"] = df["luas_panen_ha"] * df["produktivitas_ku_ha"] / 10
    df["selisih_pct"] = (abs(df["produksi_hitung"] - df["produksi_ton"]) / df["produksi_ton"] * 100).round(2)
    n_outlier = (df["selisih_pct"] > 5).sum()
    if n_outlier > 0:
        print(f"  ⚠ {n_outlier} baris selisih > 5%:")
        print(df.loc[df["selisih_pct"] > 5, ["kabupaten","luas_panen_ha","produktivitas_ku_ha","produksi_ton","produksi_hitung","selisih_pct"]])
    else:
        print(f"  ✅ Semua {len(df)} baris lulus validasi")

    df = df.drop(columns=["luas_panen_raw","produktivitas_raw","produksi_raw","produksi_hitung","selisih_pct"])
    df = df[["kabupaten","tahun","luas_panen_ha","produktivitas_ku_ha","produktivitas_ton_ha","produksi_ton"]].reset_index(drop=True)
    return df


def clean_semua_file(folder=".", output_path="bps_padi_jatim_gabungan.csv"):
    files = []
    for f in os.listdir(folder):
        if not f.endswith(".csv"):
            continue
        m = re.search(r"20\d{2}", f)
        if m and "2018" <= m.group() <= "2025":
            files.append(f)
    files = sorted(files)

    if not files:
        print("Tidak ada file CSV dengan tahun 2018-2025 ditemukan.")
        return None

    semua = []
    for fname in files:
        fpath = os.path.join(folder, fname)
        print(f"\nMembersihkan: {fname}")
        df = clean_bps_file(fpath)
        semua.append(df)
        print(f"  → {len(df)} baris, tahun {df['tahun'].iloc[0]}")

    gabungan = pd.concat(semua, ignore_index=True)
    gabungan.to_csv(output_path, index=False)
    print(f"\n✅ Selesai! Total {len(gabungan)} baris → {output_path}")
    return gabungan


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            df_all = clean_semua_file(folder=path)
        else:
            df = clean_bps_file(path)
            print(df.to_string())
    else:
        filepath = "/mnt/user-data/uploads/Luas_Panen__Produktivitas__dan_Produksi_Padi_Menurut_Kabupaten_Kota_di_Provinsi_Jawa_Timur__2018.csv"
        print("=== TEST CLEANING DATA 2018 ===\n")
        df = clean_bps_file(filepath, tahun=2018)
        print("\n--- Hasil ---")
        print(df.to_string())
        print(f"\nProduktivitas (ku/ha):\n{df['produktivitas_ku_ha'].describe().round(2)}")
        print(f"\nLuas Panen (ha):\n{df['luas_panen_ha'].describe().round(0)}")
        print(f"\nProduksi (ton):\n{df['produksi_ton'].describe().round(0)}")
