import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Konfigurasi Halaman Utama & Tema Gelap
st.set_page_config(page_title="AI Prediksi & Analisis Saham", layout="wide")

# --- MENGAKTIFKAN MEMORI APLIKASI (SESSION STATE) ---
if "hasil_scan" not in st.session_state:
    st.session_state.hasil_scan = None

# Custom CSS untuk tampilan modern (Glassmorphism & Neon)
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(0, 255, 255, 0.2); }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Navigasi
st.sidebar.title("🤖 AI Trading Assistant")
menu = st.sidebar.radio("Pilih Mode Analisis:", 
                        ["📈 AI Prediksi (Spesifik)", 
                         "🔍 Scanner Saham Naik (Global & ID)"])

# --- FITUR BARU: INDIKATOR SUMBER DATA DI SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 Status Sistem")
st.sidebar.success("✅ **Terhubung (Real-Time)**\n\nSumber: Yahoo Finance API")
st.sidebar.caption("Model AI: Polynomial Regression (Degree 3)")
# ----------------------------------------------------

# Daftar Saham untuk Scanner Otomatis (SUDAH DIPERBANYAK)
DAFTAR_SAHAM = [
    # --- SAHAM INDONESIA (LQ45 & Unggulan) ---
    "BBCA.JK", "BBRI.JK", "TLKM.JK", "GOTO.JK", "BRIS.JK", "BMRI.JK", "AMRT.JK",
    "ASII.JK", "UNTR.JK", "ICBP.JK", "INDF.JK", "ADRO.JK", "PGAS.JK", "ANTM.JK", 
    "PTBA.JK", "KLBF.JK", "CPIN.JK", "BBNI.JK", "ITMG.JK", "AKRA.JK",
    
    # --- SAHAM AMERIKA / GLOBAL ---
    "AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "META", "AMZN", "NFLX", "AMD", "INTC"
]

# Fungsi Core AI 
def hitung_prediksi_ai(ticker):
    try:
        data = yf.download(ticker, period="1y", progress=False)
        if len(data) < 50: return None, None, None
        
        # Proses Machine Learning (Polynomial)
        data['Hari_Ke'] = np.arange(len(data))
        X = data[['Hari_Ke']].values
        y = data['Close'].squeeze().values
        
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Prediksi 7 hari ke depan
        hari_terakhir = data['Hari_Ke'].iloc[-1]
        hari_depan = np.array([[hari_terakhir + i] for i in range(1, 8)])
        prediksi_harga = model.predict(poly.transform(hari_depan))
        
        return data, prediksi_harga, data['Close'].iloc[-1].item()
    except:
        return None, None, None

# ---------------------------------------------------------
# MENU 1: AI PREDIKSI (SPESIFIK)
# ---------------------------------------------------------
if menu == "📈 AI Prediksi (Spesifik)":
    st.title("Mesin Prediksi Harga Saham (AI Lanjutan)")
    st.write("Modul ini menggunakan **Polynomial Regression** untuk membaca gelombang tren masa lalu dan memprediksi fluktuasi (naik/turun) harga 7 hari ke depan.")
    
    ticker = st.text_input("Masukkan Kode Saham (Contoh: BBCA.JK atau NVDA):", "BBCA.JK").upper()
    
    if st.button("Jalankan Mesin AI"):
        with st.spinner("AI sedang mempelajari gelombang grafik..."):
            data, prediksi_harga, harga_skrg = hitung_prediksi_ai(ticker)
            
            if data is not None:
                tanggal_depan = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)
                
                # Menghitung Detail Tabel
                status_harian = []
                selisih_harian = []
                harga_kemarin = harga_skrg 
                
                for p in prediksi_harga:
                    selisih = p - harga_kemarin
                    if selisih > 0:
                        status_harian.append("🟢 Naik")
                    elif selisih < 0:
                        status_harian.append("🔴 Turun")
                    else:
                        status_harian.append("⚪ Tetap")
                        
                    selisih_harian.append(f"{int(abs(selisih)):,}")
                    harga_kemarin = p 
                
                # Membuat Grafik
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'].squeeze(), 
                                high=data['High'].squeeze(),
                                low=data['Low'].squeeze(), 
                                close=data['Close'].squeeze(),
                                name="Harga Saham Asli"))
                
                tanggal_sambungan = [data.index[-1]] + list(tanggal_depan)
                harga_sambungan = [harga_skrg] + list(prediksi_harga)
                
                fig.add_trace(go.Scatter(x=tanggal_sambungan, y=harga_sambungan, 
                                         mode='lines+markers', name='🤖 Prediksi AI (7 Hari)', 
                                         line=dict(color='#00FFFF', width=3, dash='dot')))
                
                fig.update_layout(title=f"Prediksi Polynomial AI: {ticker}", xaxis_title="Tanggal",
                                  yaxis_title="Harga", template="plotly_dark", height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Menampilkan Tabel Detail
                st.markdown("### 🔮 Detail Fluktuasi 7 Hari Kedepan")
                df_prediksi = pd.DataFrame({
                    "Tanggal": [t.strftime('%d %B %Y') for t in tanggal_depan],
                    "Status": status_harian,
                    "Prediksi Harga": [f"{int(p):,}" for p in prediksi_harga],
                    "Perubahan": selisih_harian
                })
                st.table(df_prediksi)
            else:
                st.error("Data saham tidak ditemukan atau koneksi gagal.")

# ---------------------------------------------------------
# MENU 2: SCANNER SAHAM NAIK (GLOBAL & ID)
# ---------------------------------------------------------
elif menu == "🔍 Scanner Saham Naik (Global & ID)":
    st.title("🚀 Top Recommendations (Potensi Naik)")
    st.write("Sistem memindai pasar untuk mencari tren positif 7 hari ke depan. Klik menu detail di bawah tiap saham untuk melihat prediksi harian.")
    
    # Tombol Scan
    if st.button("Mulai Pemindaian Pasar (Perbarui Data)"):
        rekomendasi_sementara = []
        progress_bar = st.progress(0)
        
        for i, t in enumerate(DAFTAR_SAHAM):
            progress_bar.progress((i + 1) / len(DAFTAR_SAHAM))
            data, prediksi, harga_skrg = hitung_prediksi_ai(t)
            
            if data is not None and prediksi[-1] > harga_skrg:
                persen = ((prediksi[-1] - harga_skrg) / harga_skrg) * 100
                
                # --- MENYIAPKAN DATA TABEL UNTUK MENU KLIK ---
                tanggal_depan = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)
                status_harian = []
                harga_kemarin = harga_skrg 
                for p in prediksi:
                    if p - harga_kemarin > 0: status_harian.append("🟢")
                    else: status_harian.append("🔴")
                    harga_kemarin = p 
                    
                df_detail = pd.DataFrame({
                    "Tanggal": [tgl.strftime('%d %b') for tgl in tanggal_depan],
                    "Arah": status_harian,
                    "Estimasi": [f"{int(p):,}" for p in prediksi]
                })
                
                rekomendasi_sementara.append({
                    "ticker": t, "harga": harga_skrg, "prediksi": prediksi[-1], 
                    "persen": persen, "data_hist": data['Close'].tail(20),
                    "tabel_detail": df_detail 
                })
        
        progress_bar.empty()
        st.session_state.hasil_scan = rekomendasi_sementara
        
    # --- MENAMPILKAN HASIL DARI MEMORI ---
    if st.session_state.hasil_scan is not None:
        rekomendasi = st.session_state.hasil_scan
        
        if len(rekomendasi) > 0:
            rekomendasi = sorted(rekomendasi, key=lambda x: x['persen'], reverse=True)
            
            cols = st.columns(3)
            for idx, item in enumerate(rekomendasi):
                with cols[idx % 3]:
                    st.subheader(f"📈 {item['ticker']}")
                    st.write(f"Harga Skrg: **{item['harga']:,.2f}**")
                    st.write(f"Target (7H): **{item['prediksi']:,.2f}**")
                    
                    # Grafik Mini
                    mini_fig = go.Figure()
                    mini_fig.add_trace(go.Scatter(y=item['data_hist'], mode='lines', line=dict(color='#39FF14', width=3)))
                    mini_fig.update_layout(xaxis_visible=False, yaxis_visible=False, height=80, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(mini_fig, use_container_width=True, config={'displayModeBar': False})
                    
                    st.write(f"Potensi: :green[+{item['persen']:.2f}%]")
                    
                    # Fitur Dropdown
                    with st.expander("📅 Lihat Tren & Tanggal"):
                        st.dataframe(item['tabel_detail'], hide_index=True, use_container_width=True)
                        
                    st.markdown("---")
        else:
            st.warning("Belum ditemukan saham dengan sinyal kenaikan kuat.")