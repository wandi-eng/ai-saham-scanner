import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Konfigurasi Halaman Utama & Tema Gelap
st.set_page_config(page_title="Prediksi & Analyzer Saham", layout="wide")

if "hasil_scan" not in st.session_state:
    st.session_state.hasil_scan = None

# Custom CSS untuk tampilan modern
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(0, 255, 255, 0.2); }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("🤖 AI Trading Assistant")
menu = st.sidebar.radio("Pilih Mode Analisis:", 
                        ["📈 Prediksi Berbasis MA (Spesifik)", 
                         "🔍 Scanner Saham Naik (Global & ID)"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 Status Sistem")
st.sidebar.success("✅ **Terhubung (Real-Time)**\n\nSumber: Yahoo Finance API")
st.sidebar.caption("Kalender: Hari Bursa Aktif (Skip Weekend)\nModel AI: MA20/50 + Short-Term Linear")

# Daftar Saham untuk Scanner Otomatis
DAFTAR_SAHAM = [
    # Saham Indonesia
    "BBCA.JK", "BBRI.JK", "TLKM.JK", "GOTO.JK", "BRIS.JK", "BMRI.JK", "AMRT.JK",
    "ASII.JK", "UNTR.JK", "ICBP.JK", "INDF.JK", "ADRO.JK", "PGAS.JK", "ANTM.JK", 
    # Saham Global
    "AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "META", "AMZN"
]

# Fungsi Core AI (Berbasis MA20 & 20-Day Trend)
def hitung_prediksi_ai(ticker):
    try:
        data = yf.download(ticker, period="1y", progress=False)
        if len(data) < 50: return None, None, None, None, None
        
        # Menghitung MA20 dan MA50
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        # Prediksi Masa Depan: Regresi Linear pada 20 Hari Terakhir
        data_20 = data.tail(20).copy()
        data_20['Hari_Ke'] = np.arange(len(data_20))
        
        X = data_20[['Hari_Ke']].values
        y = data_20['Close'].squeeze().values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Prediksi 7 hari BUKA BURSA ke depan
        hari_terakhir = data_20['Hari_Ke'].iloc[-1]
        hari_depan = np.array([[hari_terakhir + i] for i in range(1, 8)])
        prediksi_harga = model.predict(hari_depan)
        
        return data, prediksi_harga, data['Close'].iloc[-1].item(), data['MA20'], data['MA50']
    except:
        return None, None, None, None, None

# ---------------------------------------------------------
# MENU 1: AI PREDIKSI (SPESIFIK)
# ---------------------------------------------------------
if menu == "📈 Prediksi Berbasis MA (Spesifik)":
    st.title("Analisis Tren MA20/MA50 & Prediksi AI")
    st.write("Modul ini menampilkan tren **MA20 & MA50** historis, dan memprediksi masa depan berdasarkan momentum harga 20 hari kerja terakhir.")
    
    ticker = st.text_input("Masukkan Kode Saham (Contoh: ASII.JK atau NVDA):", "ASII.JK").upper()
    
    if st.button("Jalankan Analisis"):
        with st.spinner("Menghitung rata-rata pergerakan..."):
            data, prediksi_harga, harga_skrg, ma20, ma50 = hitung_prediksi_ai(ticker)
            
            if data is not None:
                # PERBAIKAN KALENDER: MENGGUNAKAN BDATE_RANGE (BUSINESS DAYS)
                tanggal_depan = pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)
                
                status_harian = []
                selisih_harian = []
                harga_kemarin = harga_skrg 
                
                for p in prediksi_harga:
                    selisih = p - harga_kemarin
                    if selisih > 0: status_harian.append("🟢 Naik")
                    elif selisih < 0: status_harian.append("🔴 Turun")
                    else: status_harian.append("⚪ Tetap")
                        
                    selisih_harian.append(f"{int(abs(selisih)):,}")
                    harga_kemarin = p 
                
                # Membuat Grafik
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'].squeeze(), high=data['High'].squeeze(),
                                low=data['Low'].squeeze(), close=data['Close'].squeeze(),
                                name="Harga Saham"))
                
                fig.add_trace(go.Scatter(x=data.index, y=ma20.squeeze(), mode='lines', name='MA20 (Tren Cepat)', line=dict(color='#FFD700', width=2)))
                fig.add_trace(go.Scatter(x=data.index, y=ma50.squeeze(), mode='lines', name='MA50 (Tren Menengah)', line=dict(color='#9370DB', width=2)))
                
                tanggal_sambungan = [data.index[-1]] + list(tanggal_depan)
                harga_sambungan = [harga_skrg] + list(prediksi_harga)
                
                fig.add_trace(go.Scatter(x=tanggal_sambungan, y=harga_sambungan, 
                                         mode='lines+markers', name='🤖 Proyeksi Tren (7 Hari Bursa)', 
                                         line=dict(color='#00FFFF', width=3, dash='dot')))
                
                fig.update_layout(title=f"Analisis MA & Proyeksi: {ticker}", xaxis_title="Tanggal",
                                  yaxis_title="Harga", template="plotly_dark", height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 🔮 Detail Proyeksi 7 Hari Bursa Kedepan")
                df_prediksi = pd.DataFrame({
                    "Tanggal": [t.strftime('%d %B %Y') for t in tanggal_depan],
                    "Hari": [t.strftime('%A') for t in tanggal_depan], # Tambahan kolom nama hari
                    "Status": status_harian,
                    "Estimasi Harga": [f"{int(p):,}" for p in prediksi_harga],
                    "Perubahan": selisih_harian
                })
                st.table(df_prediksi)
            else:
                st.error("Data saham tidak ditemukan.")

# ---------------------------------------------------------
# MENU 2: SCANNER SAHAM NAIK (GLOBAL & ID)
# ---------------------------------------------------------
elif menu == "🔍 Scanner Saham Naik (Global & ID)":
    st.title("🚀 Top Recommendations (Bursa Aktif)")
    st.write("Sistem memindai pasar untuk mencari saham yang momentum MA20-nya diprediksi terus naik untuk 7 hari perdagangan ke depan.")
    
    if st.button("Mulai Pemindaian Pasar"):
        rekomendasi_sementara = []
        progress_bar = st.progress(0)
        
        for i, t in enumerate(DAFTAR_SAHAM):
            progress_bar.progress((i + 1) / len(DAFTAR_SAHAM))
            data, prediksi, harga_skrg, ma20, ma50 = hitung_prediksi_ai(t)
            
            if data is not None and prediksi[-1] > harga_skrg and harga_skrg > ma20.iloc[-1].item():
                persen = ((prediksi[-1] - harga_skrg) / harga_skrg) * 100
                
                # PERBAIKAN KALENDER DI MENU SCANNER
                tanggal_depan = pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)
                status_harian = []
                harga_kemarin = harga_skrg 
                for p in prediksi:
                    if p - harga_kemarin > 0: status_harian.append("🟢")
                    else: status_harian.append("🔴")
                    harga_kemarin = p 
                    
                df_detail = pd.DataFrame({
                    "Tgl/Hari": [f"{tgl.strftime('%d %b')} ({tgl.strftime('%a')})" for tgl in tanggal_depan],
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
                    
                    mini_fig = go.Figure()
                    mini_fig.add_trace(go.Scatter(y=item['data_hist'], mode='lines', line=dict(color='#39FF14', width=3)))
                    mini_fig.update_layout(xaxis_visible=False, yaxis_visible=False, height=80, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(mini_fig, use_container_width=True, config={'displayModeBar': False})
                    
                    st.write(f"Potensi: :green[+{item['persen']:.2f}%]")
                    
                    with st.expander("📅 Lihat Prediksi Hari Bursa"):
                        st.dataframe(item['tabel_detail'], hide_index=True, use_container_width=True)
                        
                    st.markdown("---")
        else:
            st.warning("Belum ditemukan saham yang momentum MA20-nya sedang kuat hari ini.")
