# 🌾 PanenAman (Sistem Prediksi & Ketahanan Pangan)

PanenAman adalah platform analitik cerdas berbasis Machine Learning yang dirancang untuk memantau ketahanan pangan padi, mendeteksi anomali cuaca ekstrem, dan memberikan rekomendasi mitigasi risiko gagal panen secara real-time.

## 🚀 Fitur Utama
- Peta Status Ketahanan Pangan: Visualisasi interaktif status daerah (Bahaya, Waspada, Normal).
- Analisis Tren & Korelasi (EDA): Mengeksplorasi hubungan antara faktor cuaca terhadap hasil panen.
- Prediksi AI (Random Forest): Memprediksi potensi gagal panen menggunakan model Machine Learning.
- Early Warning Cuaca: Terintegrasi dengan Open-Meteo API untuk memantau prakiraan cuaca 16 hari ke depan.
- Asisten AI Terintegrasi: Chatbot cerdas yang ditenagai oleh model AI (via OpenRouter) untuk saran mitigasi.

## 🛠️ Teknologi yang Digunakan
- Frontend/UI: Streamlit
- Data & Machine Learning: Pandas, NumPy, Scikit-Learn
- Visualisasi: Plotly Express & Plotly Graph Objects
- External API: Open-Meteo API (Cuaca), OpenRouter API (AI Chatbot)
- Deployment: Docker & Microsoft Azure App Service

## 💻 Panduan Instalasi Lokal

1. Clone repository ini:
   git clone https://github.com/millkywaay/PanenAman.git
   cd PanenAman

2. Install dependencies:
   pip install -r requirements.txt

3. Konfigurasi Environment Variable:
   Buat file .env di direktori utama, lalu tambahkan API Key:
   OPENROUTER_API_KEY="sk-or-v1-kunci-api-anda"
   OPENROUTER_MODEL="poolside/laguna-xs.2:free"

4. Jalankan Aplikasi:
   streamlit run app.py

## ☁️ Deployment (Docker & Azure)
Aplikasi ini di-deploy menggunakan Azure App Service (Web App for Containers). Images dibangun menggunakan Dockerfile yang tersedia di repositori ini. Rahasia API (API Keys) dikonfigurasi secara aman melalui menu Environment Variables di Azure Portal.

---
Dibuat untuk mendukung ketahanan pangan dan inovasi agrikultur cerdas di Jawa Timur.