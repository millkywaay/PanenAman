# 🌾 PanenAman
### Sistem Prediksi & Ketahanan Pangan Nasional

PanenAman adalah platform analitik cerdas berbasis Machine Learning yang dirancang untuk memperkuat ketahanan pangan nasional. Dengan mengintegrasikan data historis BPS dan prakiraan cuaca real-time, platform ini mampu mendeteksi risiko gagal panen dan memberikan rekomendasi mitigasi yang presisi.

## 🔗 Akses Aplikasi

👉 **[https://panen-aman-app.azurewebsites.net/](https://panen-aman-app.azurewebsites.net/)**

---

## 🚀 Fitur Utama

- **🗺️ Peta Status Ketahanan Pangan** — Visualisasi interaktif status daerah (Bahaya, Waspada, Normal) berbasis data spasial.
- **📊 Analisis Tren & Korelasi (EDA)** — Eksplorasi mendalam hubungan antara anomali cuaca terhadap produktivitas hasil panen.
- **🤖 Prediksi AI (Random Forest)** — Estimasi potensi hasil panen dan risiko kegagalan menggunakan model Random Forest yang telah dilatih.
- **⚡ Early Warning Cuaca** — Integrasi Open-Meteo API untuk memantau prakiraan cuaca ekstrem hingga 16 hari ke depan.
- **💬 Asisten AI Terintegrasi** — Chatbot cerdas (via OpenRouter) yang memberikan saran teknis budidaya dan strategi mitigasi risiko.

---

## 🛠️ Teknologi yang Digunakan

| Kategori | Teknologi |
|---|---|
| Frontend/UI | Streamlit |
| Data & ML | Pandas, NumPy, Scikit-Learn |
| Visualisasi | Plotly Express & Plotly Graph Objects |
| External API | Open-Meteo API (Cuaca), OpenRouter API (AI Chatbot) |
| Deployment | Docker & Microsoft Azure App Service |

---

## 👥 Tim & Peran

| Nama | Peran |
|---|---|
| **Moh Alwi Fuad** | Data Researcher & Validator — Riset threshold cuaca & validasi data |
| **Khoirunnisa** | Machine Learning Engineer — Pengembangan & optimasi model AI |
| **Miftah Al Ghifari** | Cloud & DevOps Engineer — Deployment Azure, Docker, & CI/CD |

---

## 💻 Panduan Instalasi Lokal

**1. Clone repository:**
```bash
git clone https://github.com/millkywaay/PanenAman.git
cd PanenAman
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Konfigurasi Environment Variable:**

Buat file `.env` di direktori utama, lalu tambahkan API Key:
```env
OPENROUTER_API_KEY="sk-or-v1-kunci-api-anda"
OPENROUTER_MODEL="poolside/laguna-xs.2:free"
```

**4. Jalankan aplikasi:**
```bash
streamlit run app.py
```

---

## ☁️ Deployment (Docker & Azure)

Aplikasi ini di-deploy menggunakan **Azure App Service (Web App for Containers)**. 
