
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, requests, os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

st.set_page_config(
    page_title="PanenAman — Ketahanan Pangan Jawa Timur",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────────────────────────────────────────
KOORDINAT = {
    "Pacitan":(-8.1845,111.1051),"Ponorogo":(-7.8656,111.4638),
    "Trenggalek":(-8.049,111.7091),"Tulungagung":(-8.0657,111.9021),
    "Blitar":(-8.0984,112.1615),"Kediri":(-7.8167,112.0167),
    "Malang":(-7.9797,112.6304),"Lumajang":(-8.1349,113.2237),
    "Jember":(-8.1725,113.7001),"Banyuwangi":(-8.2194,114.3691),
    "Bondowoso":(-7.91,113.82),"Situbondo":(-7.7058,114.0022),
    "Probolinggo":(-7.7543,113.2159),"Pasuruan":(-7.6456,112.9062),
    "Sidoarjo":(-7.4478,112.7183),"Mojokerto":(-7.4709,112.434),
    "Jombang":(-7.5494,112.2242),"Nganjuk":(-7.6042,111.9019),
    "Madiun":(-7.6298,111.5236),"Magetan":(-7.65,111.3294),
    "Ngawi":(-7.4033,111.4462),"Bojonegoro":(-7.1478,111.8817),
    "Tuban":(-6.8996,112.0498),"Lamongan":(-7.1173,112.416),
    "Gresik":(-7.1573,112.654),"Bangkalan":(-6.9029,112.9835),
    "Sampang":(-7.1982,113.2479),"Pamekasan":(-7.157,113.474),
    "Sumenep":(-6.9767,113.8606),"Kota Kediri":(-7.8167,112.0167),
    "Kota Blitar":(-8.0984,112.1615),"Kota Malang":(-7.9797,112.6304),
    "Kota Probolinggo":(-7.7543,113.2159),"Kota Pasuruan":(-7.6456,112.9062),
    "Kota Mojokerto":(-7.4709,112.434),"Kota Madiun":(-7.6298,111.5236),
    "Kota Surabaya":(-7.2575,112.7521),"Kota Batu":(-7.8669,112.5248),
}

# Label baru: Bahaya / Waspada / Normal
COLOR_MAP  = {"Bahaya": "#f85149", "Waspada": "#d29922", "Normal": "#3fb950"}
# Mapping dari label lama di CSV ke label baru di tampilan
STATUS_DISPLAY = {"Merah": "Bahaya", "Kuning": "Waspada", "Hijau": "Normal"}
ICON_MAP   = {"Bahaya": "🔴", "Waspada": "🟡", "Normal": "🟢"}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = joblib.load("output/model_rf_panenaman.pkl")
    fitur_x = joblib.load("output/fitur_x.pkl")
    return model, fitur_x

@st.cache_data
def load_data():
    df = pd.read_csv("df_hist.csv")

    # Buat kolom 'status' dari 'label_hist' kalau belum ada
    if 'status' not in df.columns:
        if 'label_hist' in df.columns:
            df['status'] = df['label_hist'].map({0:'Hijau', 1:'Kuning', 2:'Merah'})
        else:
            df['status'] = 'Kuning'

    # Hitung threshold kalau belum ada
    if 'baseline' not in df.columns:
        df['baseline']    = df.groupby('kabupaten')['produktivitas_ku_ha'].transform('mean')
        df['batas_merah'] = df['baseline'] * 0.70
        df['q1'] = df.groupby('kabupaten')['produktivitas_ku_ha'].transform(lambda x: x.quantile(0.25))
        df['q3'] = df.groupby('kabupaten')['produktivitas_ku_ha'].transform(lambda x: x.quantile(0.75))

    # Kolom tampilan (Bahaya/Waspada/Normal)
    df['status_display'] = df['status'].map(STATUS_DISPLAY).fillna('Waspada')

    # Koordinat
    df['lat'] = df['kabupaten'].map(lambda k: KOORDINAT.get(k, (None,None))[0])
    df['lon'] = df['kabupaten'].map(lambda k: KOORDINAT.get(k, (None,None))[1])
    return df

model, FITUR_X = load_model()
df = load_data()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: FETCH CUACA OPEN-METEO
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_cuaca_forecast(lat: float, lon: float, days: int = 16) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ("temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                  "precipitation_sum,relative_humidity_2m_mean,wind_speed_10m_max"),
        "forecast_days": days,
        "timezone": "Asia/Jakarta",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: REKOMENDASI EARLY WARNING (rule-based)
# ─────────────────────────────────────────────────────────────────────────────
def buat_rekomendasi(level: str, jenis_dominan: str, max_run: int, kab: str) -> dict:
    """
    Kembalikan dict berisi:
      - judul
      - deskripsi kondisi
      - list langkah rekomendasi
      - warna
    """
    if level == "Bahaya":
        if "HUJAN" in jenis_dominan:
            return {
                "judul": f"🚨 BAHAYA — Hujan Ekstrim Berturut-turut ({max_run} hari)",
                "deskripsi": (
                    f"{kab} mengalami hujan ekstrim {max_run} hari berturut-turut. "
                    "Risiko banjir sawah, kerebahan tanaman, dan gagal panen sangat tinggi."
                ),
                "rekomendasi": [
                    "Periksa dan bersihkan saluran drainase sawah segera",
                    "Jika padi sudah fase matang (>80% kuning), percepat panen sebelum banjir",
                    "Pasang penahan agar tanaman tidak rebah akibat angin dan hujan",
                    "Tunda pemupukan nitrogen — hujan akan melarutkan pupuk sia-sia",
                    "Koordinasi dengan penyuluh pertanian setempat untuk bantuan darurat",
                ],
                "warna": "#f85149",
            }
        elif "PANAS" in jenis_dominan:
            return {
                "judul": f"🚨 BAHAYA — Panas Ekstrim Berturut-turut ({max_run} hari)",
                "deskripsi": (
                    f"{kab} mengalami suhu ekstrim {max_run} hari berturut-turut. "
                    "Risiko spikelet sterility (gabah kosong/peso) dan stres air tinggi."
                ),
                "rekomendasi": [
                    "Pastikan ketersediaan air irigasi — tingkatkan frekuensi pengairan",
                    "Hindari pemupukan saat ini, tunggu suhu normal kembali",
                    "Jika fase bunting/berbunga, pantau ketat — ini fase paling rentan panas",
                    "Lakukan pengairan pagi dan sore untuk menurunkan suhu kanopi",
                    "Laporkan ke dinas pertanian jika ada gejala gabah hampa massal",
                ],
                "warna": "#f85149",
            }
        else:
            return {
                "judul": f"🚨 BAHAYA — Cuaca Ekstrim Kombinasi ({max_run} hari)",
                "deskripsi": (
                    f"{kab} mengalami cuaca ekstrim kombinasi {max_run} hari berturut-turut. "
                    "Hujan lebat sekaligus suhu tinggi meningkatkan risiko penyakit dan gagal panen."
                ),
                "rekomendasi": [
                    "Monitor kondisi tanaman setiap hari",
                    "Siapkan fungisida — kelembapan tinggi memicu penyakit blast dan hawar",
                    "Periksa drainase sawah dan pastikan tidak ada genangan",
                    "Koordinasi dengan kelompok tani untuk respons bersama",
                    "Hubungi penyuluh pertanian untuk asesmen lapangan",
                ],
                "warna": "#f85149",
            }

    elif level == "Waspada":
        if "HUJAN" in jenis_dominan:
            return {
                "judul": f"⚠️ WASPADA — Potensi Hujan Ekstrim ({max_run} hari)",
                "deskripsi": (
                    f"{kab} menunjukkan pola hujan di atas normal selama {max_run} hari. "
                    "Belum kritis, tetapi perlu dipantau."
                ),
                "rekomendasi": [
                    "Periksa kondisi drainase sawah sekarang sebelum hujan bertambah",
                    "Tunda rencana pemupukan 3–5 hari ke depan",
                    "Pantau prakiraan cuaca harian dari BMKG",
                    "Siapkan pompa air jika sawah berisiko tergenang",
                ],
                "warna": "#d29922",
            }
        elif "PANAS" in jenis_dominan:
            return {
                "judul": f"⚠️ WASPADA — Suhu Di Atas Normal ({max_run} hari)",
                "deskripsi": (
                    f"{kab} mencatat suhu lebih tinggi dari biasanya selama {max_run} hari. "
                    "Perhatikan kebutuhan air tanaman."
                ),
                "rekomendasi": [
                    "Tambah frekuensi pengairan jika memungkinkan",
                    "Pantau tanda-tanda stres air: daun menggulung di pagi hari",
                    "Pastikan irigasi berjalan normal, cek tidak ada kebocoran saluran",
                    "Jika fase generatif, prioritaskan ketersediaan air",
                ],
                "warna": "#d29922",
            }
        else:
            return {
                "judul": f"⚠️ WASPADA — Kondisi Perlu Dipantau ({max_run} hari)",
                "deskripsi": (
                    f"Terdapat indikasi cuaca tidak normal di {kab} selama {max_run} hari."
                ),
                "rekomendasi": [
                    "Pantau kondisi tanaman dan cuaca setiap hari",
                    "Siapkan langkah antisipasi sesuai kondisi lapangan",
                    "Konsultasi dengan penyuluh pertanian terdekat",
                ],
                "warna": "#d29922",
            }

    else:  # Normal / Aman
        return {
            "judul": "✅ NORMAL — Kondisi Cuaca Mendukung",
            "deskripsi": (
                f"Tidak ada anomali cuaca signifikan terdeteksi di {kab} "
                "dalam 16 hari ke depan. Kondisi mendukung kegiatan pertanian."
            ),
            "rekomendasi": [
                "Lanjutkan jadwal tanam, pemupukan, dan perawatan sesuai rencana",
                "Manfaatkan kondisi cuaca baik untuk aplikasi pestisida/fungisida jika diperlukan",
                "Pantau prakiraan cuaca BMKG mingguan sebagai kebiasaan rutin",
                "Catat kondisi tanaman untuk dokumentasi musim tanam",
            ],
            "warna": "#3fb950",
        }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — hanya navigasi, sisanya kosong untuk AI Chat nanti
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌾 PanenAman")
    st.caption("Monitoring Ketahanan Pangan Jawa Timur")
    st.divider()

    halaman = st.radio(
        "Navigasi",
        ["🗺️ Peta Status", "📊 EDA & Tren", "🤖 Prediksi AI", "📡 Early Warning", "💬 Chatbot AI"],
        label_visibility="collapsed"
    )
    st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# HALAMAN 1 — PETA STATUS
# ─────────────────────────────────────────────────────────────────────────────
if halaman == "🗺️ Peta Status":
    st.title("🗺️ Peta Status Ketahanan Pangan")

    # ── Filter tahun DI DALAM halaman ────────────────────────────────────────
    col_f1, col_f2 = st.columns([2, 5])
    with col_f1:
        tahun_pilih = st.selectbox(
            "Pilih tahun", sorted(df['tahun'].unique(), reverse=True),
            key="peta_tahun"
        )
    st.caption(f"Menampilkan data tahun {tahun_pilih} — threshold baseline −30% per kabupaten")

    df_yr = df[df['tahun'] == tahun_pilih].dropna(subset=['lat','lon']).copy()

    # ── Metric cards ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Wilayah", len(df_yr))
    c2.metric("🔴 Bahaya",  (df_yr['status_display']=='Bahaya').sum())
    c3.metric("🟡 Waspada", (df_yr['status_display']=='Waspada').sum())
    c4.metric("🟢 Normal",  (df_yr['status_display']=='Normal').sum())

    # ── Peta ─────────────────────────────────────────────────────────────────
    fig_map = px.scatter_mapbox(
        df_yr,
        lat="lat", lon="lon",
        color="status_display",
        size="produktivitas_ku_ha",
        size_max=22,
        color_discrete_map=COLOR_MAP,
        hover_name="kabupaten",
        hover_data={
            "produktivitas_ku_ha": ":.2f",
            "produksi_ton":        ":.0f",
            "status_display":      True,
            "lat": False, "lon": False,
        },
        mapbox_style="open-street-map",
        zoom=7, center={"lat": -7.5, "lon": 112.5},
        height=520,
        title=f"Status Pangan Jawa Timur {tahun_pilih}",
        labels={"status_display": "Status"},
    )
    fig_map.update_layout(margin=dict(t=40, r=0, l=0, b=0))

    # Tangkap klik marker
    clicked = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun")

    # ── Tabel di bawah peta — filter berdasarkan klik ────────────────────────
    st.divider()

    # Ambil nama kabupaten yang diklik dari plotly event
    kab_klik = None
    if clicked and clicked.get("selection") and clicked["selection"].get("points"):
        pt = clicked["selection"]["points"][0]
        kab_klik = pt.get("hovertext") or pt.get("customdata", [None])[0]

    # Session state untuk toggle: klik 1x → filter, klik 2x → semua
    if "peta_kab_filter" not in st.session_state:
        st.session_state.peta_kab_filter = None

    if kab_klik:
        if st.session_state.peta_kab_filter == kab_klik:
            # Klik kedua → reset ke semua
            st.session_state.peta_kab_filter = None
        else:
            # Klik pertama → filter ke kabupaten ini
            st.session_state.peta_kab_filter = kab_klik

    filter_aktif = st.session_state.peta_kab_filter

    if filter_aktif:
        st.subheader(f"📌 Detail: {filter_aktif}")
        st.caption("Klik marker yang sama lagi untuk kembali ke semua kabupaten")
        tbl_data = df_yr[df_yr['kabupaten'] == filter_aktif]
    else:
        st.subheader("📋 Semua Kabupaten / Kota")
        if kab_klik:
            st.caption("Klik satu kabupaten di peta untuk melihat detailnya saja")
        tbl_data = df_yr

    # Susun kolom tabel
    tbl_cols = ['kabupaten', 'produktivitas_ku_ha', 'produksi_ton',
                'luas_panen_ha', 'status_display']
    if 'baseline' in tbl_data.columns:
        tbl_cols += ['baseline', 'batas_merah']

    tbl = tbl_data[tbl_cols].copy().sort_values('produktivitas_ku_ha')
    tbl.columns = (
        ['Kabupaten/Kota', 'Produktivitas (ku/ha)', 'Produksi (ton)',
         'Luas Panen (ha)', 'Status']
        + (['Baseline (ku/ha)', 'Batas Bahaya (ku/ha)'] if 'baseline' in tbl_data.columns else [])
    )
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# HALAMAN 2 — EDA & TREN
# ─────────────────────────────────────────────────────────────────────────────
elif halaman == "📊 EDA & Tren":
    st.title("📊 EDA & Analisis Tren")

    # ── Filter kabupaten & tahun DI DALAM halaman ────────────────────────────
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        kab_pilih = st.selectbox(
            "Pilih kabupaten/kota", sorted(df['kabupaten'].unique()),
            key="eda_kab"
        )
    with col_f2:
        tahun_pilih = st.selectbox(
            "Pilih tahun (untuk tab ranking)",
            sorted(df['tahun'].unique(), reverse=True),
            key="eda_tahun"
        )

    tab1, tab2, tab3 = st.tabs(["📈 Tren Kabupaten", "🔗 Korelasi Cuaca", "🏆 Ranking"])

    # ── Tab 1: Tren ───────────────────────────────────────────────────────────
    with tab1:
        sub = df[df['kabupaten'] == kab_pilih].sort_values('tahun')
        if sub.empty:
            st.warning("Data tidak ditemukan.")
        else:
            bl = sub['baseline'].iloc[0] if 'baseline' in sub.columns else None
            bm = sub['batas_merah'].iloc[0] if 'batas_merah' in sub.columns else None

            fig_tren = go.Figure()
            fig_tren.add_trace(go.Scatter(
                x=sub['tahun'], y=sub['produktivitas_ku_ha'],
                mode='lines+markers', name='Produktivitas',
                line=dict(color='#58a6ff', width=2.5),
                marker=dict(
                    size=10,
                    color=[COLOR_MAP.get(s, '#58a6ff') for s in sub['status_display']],
                    line=dict(color='white', width=1.5)
                )
            ))
            if bl:
                fig_tren.add_hline(y=bl, line_dash="solid", line_color="#bc8cff", line_width=1.5,
                                   annotation_text=f"Baseline {bl:.1f}",
                                   annotation_position="bottom right")
            if bm:
                fig_tren.add_hline(y=bm, line_dash="dash", line_color="#f85149", line_width=1.5,
                                   annotation_text=f"Batas Bahaya {bm:.1f}",
                                   annotation_position="bottom right")
            fig_tren.update_layout(
                title=f"Tren Produktivitas — {kab_pilih}",
                xaxis_title="Tahun", yaxis_title="Produktivitas (ku/ha)",
                height=400, template="plotly_dark",
                xaxis=dict(dtick=1),
            )
            st.plotly_chart(fig_tren, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rata-rata",  f"{sub['produktivitas_ku_ha'].mean():.2f} ku/ha")
            c2.metric("Tertinggi",  f"{sub['produktivitas_ku_ha'].max():.2f} ku/ha")
            c3.metric("Terendah",   f"{sub['produktivitas_ku_ha'].min():.2f} ku/ha")
            c4.metric("Std Deviasi",f"{sub['produktivitas_ku_ha'].std():.2f}")

    # ── Tab 2: Korelasi ───────────────────────────────────────────────────────
    with tab2:
        FITUR_VIS = [c for c in [
            'hujan_total', 'neraca_air', 'suhu_mean', 'suhu_max_mean',
            'n_run2_hujan_ekstrim', 'n_run3_hujan_ekstrim', 'maks_run_hujan_ekstrim',
            'maks_run_kering', 'n_run7_kering', 'n_hari_panas', 'n_hari_ekstrim',
            'et0_total', 'rh_max_mean', 'rh_min_mean', 'n_hari_angin_kencang',
            # nama alternatif kalau ada
            'curah_hujan_total', 'n_run2_hujan', 'n_run3_hujan', 'maks_run_hujan',
        ] if c in df.columns]

        if not FITUR_VIS:
            st.warning("Tidak ada kolom fitur cuaca yang ditemukan di df_hist.csv.")
        else:
            fitur_sel = st.selectbox("Pilih fitur cuaca", FITUR_VIS, key="fitur_scatter")
            fig_scatter = px.scatter(
                df, x=fitur_sel, y='produktivitas_ku_ha',
                color='status_display',
                color_discrete_map=COLOR_MAP,
                hover_name='kabupaten',
                hover_data={'tahun': True},
                trendline='ols',
                title=f"Korelasi: {fitur_sel.replace('_',' ')} vs Produktivitas",
                labels={
                    fitur_sel: fitur_sel.replace('_', ' '),
                    'produktivitas_ku_ha': 'Produktivitas (ku/ha)',
                    'status_display': 'Status',
                },
                template='plotly_dark', height=450,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            corr_val = df[[fitur_sel, 'produktivitas_ku_ha']].corr().iloc[0, 1]
            arah = "positif 📈" if corr_val > 0 else "negatif 📉"
            kekuatan = "kuat" if abs(corr_val) > 0.5 else "sedang" if abs(corr_val) > 0.3 else "lemah"
            st.info(f"Korelasi Pearson: **{corr_val:.3f}** — hubungan {arah} dan {kekuatan}")

    # ── Tab 3: Ranking ────────────────────────────────────────────────────────
    with tab3:
        df_rank = df[df['tahun'] == tahun_pilih].sort_values('produktivitas_ku_ha')
        fig_rank = px.bar(
            df_rank, x='produktivitas_ku_ha', y='kabupaten',
            color='status_display',
            color_discrete_map=COLOR_MAP,
            orientation='h',
            title=f"Ranking Produktivitas Tahun {tahun_pilih}",
            labels={
                'produktivitas_ku_ha': 'Produktivitas (ku/ha)',
                'kabupaten': '',
                'status_display': 'Status',
            },
            template='plotly_dark', height=780,
        )
        fig_rank.update_layout(showlegend=True)
        st.plotly_chart(fig_rank, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# HALAMAN 3 — PREDIKSI AI
# ─────────────────────────────────────────────────────────────────────────────
elif halaman == "🤖 Prediksi AI":
    st.title("🤖 Prediksi Status Pangan")
    st.caption("Random Forest Classifier — input fitur cuaca → prediksi Bahaya / Waspada / Normal")

    # ── Filter kabupaten & tahun DI DALAM halaman ────────────────────────────
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        kab_pilih = st.selectbox(
            "Kabupaten / Kota", sorted(df['kabupaten'].unique()),
            key="pred_kab"
        )
    with col_f2:
        tahun_pilih = st.selectbox(
            "Tahun (untuk mode historis)",
            sorted(df['tahun'].unique(), reverse=True),
            key="pred_tahun"
        )

    st.divider()
    # Hanya dua mode: Manual dan Dari data historis
    mode = st.radio(
        "Mode input fitur",
        ["Dari data historis", "Input manual"],
        horizontal=True
    )

    fitur_input = {}

    if mode == "Dari data historis":
        row = df[(df['kabupaten'] == kab_pilih) & (df['tahun'] == tahun_pilih)]
        if row.empty:
            st.warning("Data tidak ditemukan untuk kombinasi ini.")
            st.stop()
        row = row.iloc[0]
        fitur_input = {f: float(row[f]) for f in FITUR_X if f in row.index}
        st.success(f"Data historis **{kab_pilih} {tahun_pilih}** berhasil dimuat — {len(fitur_input)} fitur.")
        with st.expander("Lihat nilai fitur"):
            st.dataframe(
                pd.DataFrame.from_dict(
                    fitur_input, orient='index', columns=['Nilai']
                ).rename_axis('Fitur'),
                use_container_width=True
            )

    else:  # Input manual
        st.info("Isi nilai fitur cuaca yang ingin diprediksi.")
        col1, col2, col3 = st.columns(3)
        with col1:
            fitur_input['suhu_mean']          = st.number_input("Suhu rata-rata (°C)",       20.0, 35.0, 27.0, 0.1)
            fitur_input['suhu_max_mean']       = st.number_input("Suhu max rata-rata (°C)",   25.0, 40.0, 31.0, 0.1)
            fitur_input['suhu_min_mean']       = st.number_input("Suhu min rata-rata (°C)",   18.0, 28.0, 23.0, 0.1)
            fitur_input['n_hari_panas']        = st.number_input("Hari panas ekstrim/tahun",  0, 150, 20)
            fitur_input['luas_panen_ha']       = st.number_input("Luas panen (ha)",           100.0, 200000.0, 50000.0, 1000.0)
        with col2:
            fitur_input['n_run7_kering']       = st.number_input("Run 7+ hari kering",        0, 30, 5)
            fitur_input['maks_run_kering']     = st.number_input("Run kering terpanjang (hari)", 0, 120, 20)
            fitur_input['neraca_air']          = st.number_input("Neraca air (mm)",            -500.0, 2000.0, 200.0, 10.0)
            fitur_input['et0_total']           = st.number_input("ET0 total/tahun (mm)",       800.0, 2000.0, 1400.0, 10.0)
            fitur_input['et0_mean']            = st.number_input("ET0 harian rata-rata",       2.0, 6.0, 3.8, 0.1)
        with col3:
            fitur_input['radiasi_mean']        = st.number_input("Radiasi matahari rata-rata", 10.0, 25.0, 18.0, 0.5)
            fitur_input['rh_max_mean']         = st.number_input("RH max rata-rata (%)",       70.0, 100.0, 90.0, 0.5)
            fitur_input['rh_min_mean']         = st.number_input("RH min rata-rata (%)",       40.0, 80.0, 60.0, 0.5)
            fitur_input['angin_max_mean']      = st.number_input("Kecepatan angin max (km/h)", 5.0, 50.0, 17.0, 0.5)
            fitur_input['n_hari_angin_kencang']= st.number_input("Hari angin kencang/tahun",   0, 60, 3)

    # ── Tombol prediksi ───────────────────────────────────────────────────────
    st.divider()
    if st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True):
        if not fitur_input:
            st.warning("Isi fitur terlebih dahulu.")
            st.stop()

        x_vec   = np.array([[fitur_input.get(f, 0) for f in FITUR_X]])
        pred_raw = model.predict(x_vec)[0]           # Merah/Kuning/Hijau (dari model)
        pred     = STATUS_DISPLAY.get(pred_raw, pred_raw)  # → Bahaya/Waspada/Normal
        proba    = model.predict_proba(x_vec)[0]
        classes  = [STATUS_DISPLAY.get(c, c) for c in model.classes_]

        warna = COLOR_MAP.get(pred, '#58a6ff')
        ikon  = ICON_MAP.get(pred, '❓')
        deskripsi = {
            "Bahaya":  "⚠️ Risiko gagal panen tinggi — produktivitas diprediksi di bawah 70% baseline",
            "Waspada": "📋 Kondisi di bawah rata-rata — perlu perhatian dan pemantauan rutin",
            "Normal":  "🎉 Kondisi baik — produktivitas diprediksi di atas rata-rata historis",
        }.get(pred, "")

        st.markdown(f"""
        <div style="background:{warna}22;border:2px solid {warna};border-radius:12px;
                    padding:24px;text-align:center;margin:16px 0">
            <div style="font-size:3.5rem">{ikon}</div>
            <div style="font-size:2rem;font-weight:700;color:{warna};margin:8px 0">
                {pred.upper()}
            </div>
            <div style="color:#8b949e;font-size:0.95rem">{deskripsi}</div>
        </div>
        """, unsafe_allow_html=True)

        # Probabilitas
        fig_prob = go.Figure(go.Bar(
            x=classes,
            y=proba,
            marker_color=[COLOR_MAP.get(c, 'gray') for c in classes],
            text=[f"{p*100:.1f}%" for p in proba],
            textposition='outside',
        ))
        fig_prob.update_layout(
            title="Distribusi probabilitas prediksi",
            yaxis=dict(range=[0, 1.15], tickformat='.0%'),
            template='plotly_dark', height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_prob, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# HALAMAN 4 — EARLY WARNING
# ─────────────────────────────────────────────────────────────────────────────
elif halaman == "📡 Early Warning":
    st.title("📡 Early Warning Cuaca")
    st.caption("Deteksi anomali cuaca 16 hari ke depan — realtime dari Open-Meteo Forecast API")

    # ── Filter kabupaten DI DALAM halaman (tidak butuh tahun) ────────────────
    col_f1, col_f2 = st.columns([2, 3])
    with col_f1:
        kab_ew = st.selectbox(
            "Pilih kabupaten / kota",
            sorted(df['kabupaten'].unique()),
            key="ew_kab"
        )
    with col_f2:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        cek = st.button("🌤️ Cek kondisi cuaca sekarang", type="primary", use_container_width=True)

    if not cek:
        st.info("Pilih kabupaten lalu klik tombol di atas untuk mengambil data cuaca realtime.")
        st.stop()

    lat, lon = KOORDINAT.get(kab_ew, (-7.5, 112.5))
    with st.spinner(f"Mengambil data cuaca {kab_ew} dari Open-Meteo..."):
        raw = fetch_cuaca_forecast(lat, lon, 16)

    if "error" in raw:
        st.error(f"Gagal mengambil data: {raw['error']}")
        st.stop()

    daily   = raw["daily"]
    tanggal = daily["time"]
    hujan   = [v or 0 for v in daily.get("precipitation_sum", [0]*16)]
    suhu_mx = [v or 0 for v in daily.get("temperature_2m_max", [0]*16)]
    suhu_mn = [v or 0 for v in daily.get("temperature_2m_min", [0]*16)]
    suhu_rt = [v or 0 for v in daily.get("temperature_2m_mean", [0]*16)]
    rh      = [v or 0 for v in daily.get("relative_humidity_2m_mean", [0]*16)]
    angin   = [v or 0 for v in daily.get("wind_speed_10m_max", [0]*16)]

    # Ambil Q3 historis kabupaten dari df kalau ada, fallback ke Q3 forecast
    kab_hist = df[df['kabupaten'] == kab_ew]
    if not kab_hist.empty and 'q3' in kab_hist.columns:
        # Pakai threshold historis kabupaten
        q3_h = kab_hist['q3'].iloc[0]          # Q3 produktivitas sebagai proxy
        q3_s = float(kab_hist['suhu_max_mean'].mean()) if 'suhu_max_mean' in kab_hist.columns else np.percentile(suhu_mx, 75)
    else:
        q3_h = np.percentile(hujan, 75) if max(hujan) > 0 else 10
        q3_s = np.percentile(suhu_mx, 75)

    # Threshold cuaca harian
    q3_hujan_hari = np.percentile(hujan, 75) if max(hujan) > 0 else 15
    q3_suhu_hari  = np.percentile(suhu_mx, 75)
    q3_angin_hari = np.percentile(angin, 75)

    # Flag tiap hari
    flags, jenis = [], []
    for h, s, a in zip(hujan, suhu_mx, angin):
        eks_h = h > q3_hujan_hari
        eks_s = s > q3_suhu_hari
        eks_a = a > q3_angin_hari
        if eks_h and eks_s:
            flags.append(1); jenis.append("HUJAN+PANAS")
        elif eks_h:
            flags.append(1); jenis.append("HUJAN LEBAT")
        elif eks_s:
            flags.append(1); jenis.append("PANAS")
        elif eks_a:
            flags.append(1); jenis.append("ANGIN KENCANG")
        else:
            flags.append(0); jenis.append("NORMAL")

    # Hitung run maksimum
    max_run = 0; cur_run = 0
    for f in flags:
        cur_run = cur_run + 1 if f else 0
        max_run = max(max_run, cur_run)

    # Jenis dominan (selain NORMAL)
    from collections import Counter
    non_normal = [j for j in jenis if j != "NORMAL"]
    jenis_dominan = Counter(non_normal).most_common(1)[0][0] if non_normal else "NORMAL"

    # Tentukan level
    if max_run >= 3:
        level = "Bahaya"
    elif max_run >= 2:
        level = "Waspada"
    else:
        level = "Normal"

    # ── Tampilkan status utama ────────────────────────────────────────────────
    st.divider()
    rek = buat_rekomendasi(level, jenis_dominan, max_run, kab_ew)
    warna = rek["warna"]

    st.markdown(f"""
    <div style="background:{warna}18;border:2px solid {warna};border-radius:14px;
                padding:20px 24px;margin-bottom:16px">
        <div style="font-size:1.4rem;font-weight:700;color:{warna};margin-bottom:8px">
            {rek['judul']}
        </div>
        <div style="color:#c9d1d9;font-size:0.95rem;line-height:1.6">
            {rek['deskripsi']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Rekomendasi tindakan ──────────────────────────────────────────────────
    st.subheader("📋 Rekomendasi Tindakan")
    for i, r in enumerate(rek["rekomendasi"], 1):
        st.markdown(f"**{i}.** {r}")

    st.divider()

    # ── Metrik ringkasan 16 hari ──────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Suhu rata-rata", f"{np.mean(suhu_rt):.1f} °C")
    c2.metric("Suhu max rata", f"{np.mean(suhu_mx):.1f} °C")
    c3.metric("Total hujan",   f"{sum(hujan):.0f} mm")
    c4.metric("Hari ekstrim",  sum(flags))
    c5.metric("Run terpanjang",f"{max_run} hari")

    st.divider()

    # ── Chart curah hujan 16 hari ─────────────────────────────────────────────
    color_ew = {
        "NORMAL":       "#3fb950",
        "HUJAN LEBAT":  "#58a6ff",
        "PANAS":        "#d29922",
        "ANGIN KENCANG":"#bc8cff",
        "HUJAN+PANAS":  "#f85149",
    }
    df_ew = pd.DataFrame({
        "Tanggal":      tanggal,
        "Curah Hujan (mm)": hujan,
        "Suhu Max (°C)":    suhu_mx,
        "Kelembapan (%)":   rh,
        "Kecepatan Angin":  angin,
        "Kondisi":          jenis,
    })

    tab_c1, tab_c2 = st.tabs(["🌧️ Curah Hujan", "🌡️ Suhu Max"])

    with tab_c1:
        fig_hujan = px.bar(
            df_ew, x="Tanggal", y="Curah Hujan (mm)",
            color="Kondisi",
            color_discrete_map=color_ew,
            title=f"Curah Hujan Harian 16 Hari ke Depan — {kab_ew}",
            template="plotly_dark", height=360,
        )
        fig_hujan.add_hline(
            y=q3_hujan_hari, line_dash="dash", line_color="#f85149",
            annotation_text=f"Q3 threshold ({q3_hujan_hari:.1f} mm)",
            annotation_position="top left"
        )
        st.plotly_chart(fig_hujan, use_container_width=True)

    with tab_c2:
        fig_suhu = px.line(
            df_ew, x="Tanggal", y="Suhu Max (°C)",
            title=f"Suhu Maksimum Harian — {kab_ew}",
            markers=True,
            template="plotly_dark", height=360,
            color_discrete_sequence=["#d29922"],
        )
        fig_suhu.add_hline(
            y=q3_suhu_hari, line_dash="dash", line_color="#f85149",
            annotation_text=f"Q3 threshold ({q3_suhu_hari:.1f} °C)",
            annotation_position="top left"
        )
        st.plotly_chart(fig_suhu, use_container_width=True)

    # ── Tabel lengkap 16 hari ─────────────────────────────────────────────────
    with st.expander("📊 Lihat data lengkap 16 hari"):
        st.dataframe(df_ew, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# HALAMAN 5 — CHATBOT AI (DEDICATED PAGE)
# ─────────────────────────────────────────────────────────────────────────────
elif halaman == "💬 Chatbot AI":
    st.title("💬 Asisten AI PanenAman")
    st.caption("Tanyakan apa saja seputar anomali cuaca, mitigasi pertanian, atau hasil prediksi sistem.")

    # CSS untuk mempercantik tampilan chat
    st.markdown("""
        <style>
        .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
        .stChatInputContainer { padding-bottom: 20px; }
        </style>
    """, unsafe_allow_html=True)

    # Inisialisasi Session State
    if "messages" not in st.session_state:
        # Prompt sistem dengan instruksi agar jawaban tetap singkat/to-the-point
        system_instructions = (
            "Kamu adalah Asisten AI PanenAman (Sistem Prediksi & Ketahanan Pangan Padi Jawa Timur). "
            "Tugasmu: membantu analisis cuaca, mitigasi gagal panen, dan menjelaskan data Random Forest/BPS/Open-Meteo. "
            "Aturan Penting: Jawab dengan ramah, teknis akurat, namun SANGAT RINGKAS DAN PADAT (maksimal 2-3 paragraf). "
            "Gunakan poin-poin jika perlu agar mudah dibaca petani."
        )
        
        # Logika "Akali" pesan pertama (Hidden Handshake)
        st.session_state.messages = [
            {"role": "user", "content": f"Tugasmu: {system_instructions}\n\nApakah kamu mengerti?"},
            {"role": "assistant", "content": "Mengerti! Saya siap menjadi Asisten AI PanenAman. Saya akan memberikan jawaban yang ringkas dan solutif."}
        ]

    # Tampilkan chat (Skip index 0 dan 1 agar instruksi awal tidak terlihat)
    for msg in st.session_state.messages[2:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input User
    if prompt := st.chat_input("Contoh: Apa saran mitigasi jika hujan lebat 5 hari di Malang?"):
        # Tampilkan pesan user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Respon dari OpenRouter
        with st.chat_message("assistant"):
            try:
                # Menggunakan Laguna XS.2 sebagai default model
                model_name = os.environ.get("OPENROUTER_MODEL", "poolside/laguna-xs.2:free")
                
                response = client.chat.completions.create(
                    model=model_name,
                    messages=st.session_state.messages,
                    stream=True,
                    max_tokens=800 # Laguna XS.2 support output panjang, kita set ke 800 agar pas
                )
                
                full_res = st.write_stream(response)
                st.session_state.messages.append({"role": "assistant", "content": full_res})
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate-limited" in error_msg:
                    st.error("⚠️ Server sedang ramai. Mohon tunggu beberapa detik dan coba lagi.")
                elif "400" in error_msg:
                    st.error("⚠️ Format prompt tidak didukung oleh model ini. Mencoba memuat ulang konteks...")
                else:
                    st.error(f"Gagal memuat AI. Error detail: {error_msg}")
    # Tombol Reset Chat (Opsional, diletakkan di atas agar mudah diakses)
    if len(st.session_state.messages) > 2:
        if st.button("🗑️ Bersihkan Percakapan"):
            del st.session_state.messages
            st.rerun()