# app/services.py
import time
import yfinance as yf
import pandas as pd
import numpy as np
from app.config import (
    TARGET_DIVIDEND_YIELD, PE_WAJAR_BANK, PE_WAJAR_UMUM,
    MARKET_INDEX_TICKER, ADX_THRESHOLD_GORENGAN,
    ATR_MULTIPLIER_SL, ATR_MULTIPLIER_TP,
    TRANCHE_ALLOKASI, BATAS_TOLERANSI_PENURUNAN_DIVIDEN,
    RETRY_PERCOBAAN_MAKSIMAL, RETRY_JEDA_DETIK,
    MAX_ALOKASI_SWING_PERSEN, MAX_ALOKASI_GORENGAN_PERSEN, MAX_JUMLAH_GORENGAN_BERSAMAAN
)


# =========================================================================
# HELPER: BATCH FETCH & RETRY (mempercepat & menstabilkan pengambilan data)
# =========================================================================

def ambil_riwayat_batch(tickers: list, period: str = "1y", interval: str = "1d"):
    """
    Tarik data historis BANYAK ticker sekaligus dalam SATU request ke Yahoo Finance
    (yf.download), jauh lebih cepat & lebih hemat request dibanding loop
    yf.Ticker().history() satu per satu. Dipakai oleh endpoint screener.

    Return: dict {ticker: DataFrame}. Ticker yang gagal/data kosong tidak masuk dict.
    """
    if not tickers:
        return {}
    try:
        data = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by='ticker',
            auto_adjust=False,
            threads=True,
            progress=False
        )
    except Exception:
        return {}

    hasil = {}
    if data is None or data.empty:
        return hasil

    if len(tickers) == 1:
        # yf.download tidak membuat kolom multi-index kalau cuma 1 ticker
        t = tickers[0]
        df_bersih = data.dropna(how='all')
        if not df_bersih.empty:
            hasil[t] = df_bersih
        return hasil

    for t in tickers:
        try:
            df_t = data[t].dropna(how='all')
            if not df_t.empty:
                hasil[t] = df_t
        except Exception:
            continue
    return hasil


def _ambil_info_dengan_retry(saham):
    """
    Retry pengambilan .info sampai RETRY_PERCOBAAN_MAKSIMAL kali kalau Yahoo Finance
    gagal sesaat / balikin data kosong (transient error), sebelum benar-benar dianggap gagal.

    CATATAN: sebelumnya fungsi ini mewajibkan field 'trailingEps' ada, yang bikin saham
    kecil/kurang ter-cover Yahoo Finance (misal VAST) selalu gagal walau data harga &
    teknikalnya sebenarnya tersedia. Sekarang cukup ada SALAH SATU sumber harga acuan
    (previousClose / currentPrice / regularMarketPrice) — bagian fundamental yang hilang
    akan di-fallback ke nilai default di hitung_analisis_saham, bukan bikin seluruh
    analisis gagal total.
    """
    for percobaan in range(RETRY_PERCOBAAN_MAKSIMAL):
        try:
            info = saham.info
            if info and (info.get('previousClose') or info.get('currentPrice') or info.get('regularMarketPrice')):
                return info
        except Exception:
            pass
        if percobaan < RETRY_PERCOBAAN_MAKSIMAL - 1:
            time.sleep(RETRY_JEDA_DETIK)
    return None


# =========================================================================
# INDIKATOR TEKNIKAL & UTILITAS ANALISIS
# =========================================================================

def hitung_indikator_lengkap(df, period=14):
    """Menghitung RSI, Stochastic, MACD, ATR, ADX, dan DI+/DI- secara native"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    df['RSI14'] = 100 - (100 / (1 + rs))

    df['L14'] = df['Low'].rolling(window=period).min()
    df['H14'] = df['High'].rolling(window=period).max()
    df['Stoch_K'] = 100 * ((df['Close'] - df['L14']) / (df['H14'] - df['L14'] + 1e-10))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    df['UpMove'] = df['High'].diff()
    df['DownMove'] = df['Low'].diff()

    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)

    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

    df['ATR14'] = df['TR'].rolling(window=period).mean()

    plus_di = 100 * (df['+DM'].rolling(window=period).mean() / (df['ATR14'] + 1e-10))
    minus_di = 100 * (df['-DM'].rolling(window=period).mean() / (df['ATR14'] + 1e-10))

    df['+DI14'] = plus_di
    df['-DI14'] = minus_di

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    df['ADX14'] = dx.rolling(window=period).mean()

    return df


def cek_kekuatan_support_dan_resisten(df, harga_sekarang, window_hari=60, toleransi_persen=0.015):
    """Deteksi support memindai window_hari terakhir (default 60 hari), resisten pakai 120 hari."""
    df_window = df.tail(window_hari).copy()
    df_window['is_low'] = df_window['Low'] == df_window['Low'].rolling(window=10, center=True, min_periods=1).min()
    titik_terendah_historis = df_window[df_window['is_low']]['Low'].tolist()

    jumlah_sentuhan_support = 0
    area_support_kuat = 0

    for low_val in titik_terendah_historis:
        if low_val > 0 and abs(harga_sekarang - low_val) / low_val <= toleransi_persen:
            jumlah_sentuhan_support += 1
            area_support_kuat = int(low_val)

    if jumlah_sentuhan_support >= 3:
        klasifikasi_support = f"SANGAT KUAT 🔥 (Telah diuji {jumlah_sentuhan_support}x di area Rp{area_support_kuat})"
    elif jumlah_sentuhan_support == 2:
        klasifikasi_support = f"SEDANG 🛡️ (Telah diuji 2x di area Rp{area_support_kuat})"
    else:
        klasifikasi_support = "LEMAH / DINAMIS 💤 (Hanya mengandalkan garis EMA berjalan)"

    resisten_terdekat = int(df['High'].tail(120).max())
    return klasifikasi_support, resisten_terdekat, area_support_kuat


def cek_kondisi_market():
    """Filter makro: cek tren IHSG sebelum sinyal per-saham dipakai."""
    try:
        index = yf.Ticker(MARKET_INDEX_TICKER)
        df = index.history(period="6mo", auto_adjust=False)
        if df.empty or len(df) < 50:
            return {"status": "TIDAK DIKETAHUI", "market_bullish": True, "keterangan": "Data indeks tidak tersedia, filter market dilewati"}
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        harga_terakhir = float(df['Close'].iloc[-1])
        ema50_terakhir = float(df['EMA50'].iloc[-1])
        bullish = harga_terakhir > ema50_terakhir
        return {
            "status": "BULLISH 📈" if bullish else "BEARISH 📉",
            "market_bullish": bool(bullish),
            "ihsg_saat_ini": round(harga_terakhir, 2),
            "ihsg_ema50": round(ema50_terakhir, 2)
        }
    except Exception:
        return {"status": "TIDAK DIKETAHUI", "market_bullish": True, "keterangan": "Gagal mengambil data indeks, filter market dilewati"}


def cek_guardrail_fundamental(saham, info):
    """Circuit breaker fundamental sebagai pengganti stop-loss teknikal untuk average-down."""
    alasan = []
    aman = True
    try:
        divs = saham.dividends
        if not divs.empty:
            per_tahun = divs.resample('YE').sum()
            if len(per_tahun) >= 2:
                dividen_tahun_ini = float(per_tahun.iloc[-1])
                dividen_tahun_lalu = float(per_tahun.iloc[-2])
                if dividen_tahun_lalu > 0 and dividen_tahun_ini < dividen_tahun_lalu * (1 - BATAS_TOLERANSI_PENURUNAN_DIVIDEN):
                    aman = False
                    turun_persen = round((1 - (dividen_tahun_ini / dividen_tahun_lalu)) * 100, 1)
                    alasan.append(f"Dividen turun {turun_persen}% dari tahun sebelumnya")
    except Exception:
        pass

    eps = info.get('trailingEps', 0)
    if eps is not None and eps < 0:
        aman = False
        alasan.append("EPS negatif (perusahaan sedang mencatat rugi)")

    if not alasan:
        alasan.append("Tidak ada sinyal peringatan fundamental terdeteksi")

    return {"aman_untuk_average_down": aman, "alasan": alasan}


def hitung_zona_average_down(harga_sekarang, ema20, ema50, ema200, area_support_kuat):
    """Tiga tingkat area akumulasi untuk strategi dividen tanpa cut loss."""
    level_3 = area_support_kuat if area_support_kuat > 0 else ema200
    return {
        "tranche_1": {"area_harga": ema20, "alokasi_persen": TRANCHE_ALLOKASI[0], "keterangan": "Koreksi ringan, dekat EMA20"},
        "tranche_2": {"area_harga": ema50, "alokasi_persen": TRANCHE_ALLOKASI[1], "keterangan": "Koreksi sedang, dekat EMA50"},
        "tranche_3": {"area_harga": level_3, "alokasi_persen": TRANCHE_ALLOKASI[2], "keterangan": "Koreksi dalam, dekat area support kuat / EMA200"},
        "catatan": "Alokasi bertahap ini asumsi guardrail_fundamental.aman_untuk_average_down bernilai true. Jika false, evaluasi ulang sebelum menambah posisi."
    }


# =========================================================================
# STRATEGI 1: SWING-INVESTMENT DIVIDEN
# =========================================================================

def hitung_analisis_saham(ticker_symbol: str, kondisi_market: dict = None, df_riwayat: pd.DataFrame = None):
    """
    df_riwayat: opsional, DataFrame historis yang SUDAH ditarik sebelumnya (misal lewat
    ambil_riwayat_batch di screener). Kalau None, fungsi ini akan fetch sendiri
    (dipakai untuk endpoint analisis 1 ticker berdiri sendiri).
    """
    if not ticker_symbol.endswith(".JK"):
        ticker = f"{ticker_symbol.upper()}.JK"
    else:
        ticker = ticker_symbol.upper()

    saham = yf.Ticker(ticker)
    info = _ambil_info_dengan_retry(saham)

    if not info:
        return None

    # --- A. PROSES DATA FUNDAMENTAL ---
    eps = info.get('trailingEps', 0)
    pbv_ratio = info.get('priceToBook', 0)
    return_on_equity = info.get('returnOnEquity', 0)
    beta = info.get('beta', 1.0)

    total_dividen = info.get('dividendRate', 0)
    if total_dividen == 0 or total_dividen is None:
        divs = saham.dividends
        total_dividen = int(divs.resample('YE').sum().iloc[-1]) if not divs.empty else 0

    pe_acuan = PE_WAJAR_BANK if "Bank" in info.get('industry', '') else PE_WAJAR_UMUM
    # Fallback berjenjang untuk harga acuan (beberapa saham kecil tidak selalu punya
    # semua field ini terisi di Yahoo Finance)
    harga_acuan = info.get('previousClose') or info.get('currentPrice') or info.get('regularMarketPrice') or 0
    harga_wajar = int(eps * pe_acuan) if eps > 0 else int(harga_acuan)

    if total_dividen > 0:
        harga_maks_layak_beli = int(total_dividen / TARGET_DIVIDEND_YIELD)
        status_dividen = f"LAYAK ({round((total_dividen / (harga_acuan or 1)) * 100, 2)}% Yield)"
        is_dividend_stock = True
    else:
        harga_maks_layak_beli = int(harga_wajar * 0.85)
        status_dividen = "TIDAK ADA DIVIDEN ❌"
        is_dividend_stock = False

    # --- B. PROSES DATA TEKNIKAL ---
    if df_riwayat is not None:
        df = df_riwayat.copy()
    else:
        df = saham.history(period="1y", auto_adjust=False)

    if df.empty or len(df) < 200:
        return None

    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    df = hitung_indikator_lengkap(df)

    terakhir = df.iloc[-1]
    harga_sekarang = int(terakhir['Close'])
    ema20, ema50, ema200 = int(terakhir['EMA20']), int(terakhir['EMA50']), int(terakhir['EMA200'])
    macd, stoch_d, rsi = terakhir['MACD'], terakhir['Stoch_D'], terakhir['RSI14']
    adx, plus_di, minus_di = terakhir['ADX14'], terakhir['+DI14'], terakhir['-DI14']

    volume_terakhir = int(terakhir['Volume'])
    volume_rata_rata = int(df['Volume'].tail(20).mean())
    is_volume_strong = volume_terakhir > (volume_rata_rata * 1.5)

    klasifikasi_support, resisten_terdekat, area_support_kuat = cek_kekuatan_support_dan_resisten(df, harga_sekarang)
    jarak_ke_resisten = round(((resisten_terdekat - harga_sekarang) / harga_sekarang) * 100, 2)

    # --- C. DETEKSI TEKANAN JUAL INSTITUSI & FORUM MATCH ---
    is_panic_selling = (harga_sekarang < ema50) and is_volume_strong
    status_arus_modal = "PANIC SELLING / INSTITUSI KELUAR ⚠️" if is_panic_selling else "ARUS KAS STABIL / NORMAL 👍"

    f1_kondisi = harga_sekarang < ema20 and harga_sekarang > ema200 and stoch_d <= 20 and macd < 0

    # --- DETEKSI MACD BULLISH CROSSOVER DI BAWAH GARIS NOL ("Early Rebound / Bottoming Signal") ---
    # Beda dengan f1_kondisi (yang cuma cek MACD < 0 secara statis), ini mendeteksi EVENT
    # garis MACD baru saja cross ke ATAS garis Signal-nya, sementara nilai MACD-nya sendiri
    # masih di bawah 0 - dianggap sinyal awal pembalikan sebelum tren beneran naik.
    histogram_sekarang = terakhir['MACD_Histogram']
    histogram_kemarin = df['MACD_Histogram'].iloc[-2] if len(df) >= 2 else histogram_sekarang
    macd_crossover_bullish = (histogram_kemarin <= 0) and (histogram_sekarang > 0)
    sinyal_macd_early_rebound = bool(macd_crossover_bullish and macd < 0)

    if f1_kondisi and sinyal_macd_early_rebound:
        status_forum_swing = "SANGAT AKTIF 🔥🔄 (Oversold + MACD Bullish Crossover)"
    elif f1_kondisi:
        status_forum_swing = "AKTIF 🔥"
    else:
        status_forum_swing = "TIDAK AKTIF 💤"

    f2_kondisi = (ema20 > ema50) and (rsi >= 50) and (adx > 20) and (plus_di > minus_di) and (harga_sekarang > ema200) and is_volume_strong
    status_forum_day = "TREN SANGAT KUAT 🚀" if f2_kondisi else "TREN LEMAH / SIDEWAYS 💤"

    # --- D. LOGIKA GUARDRAIL & STRATEGI ---
    wajib_stop_loss = beta > 1.3 or not is_dividend_stock

    if wajib_stop_loss:
        kategori_risiko = f"TINGGI (Beta: {round(beta, 2)}) 🔥"
        status_proteksi = "MURNI TRADING CEPAT (Wajib Stop Loss)"
        status_tren = "UPTREND SPEKULATIF 📈" if harga_sekarang > ema20 else "DOWNTREND SPEKULATIF 📉"
        rekomendasi = "WAIT/TRADING CEPAT - SET STOP LOSS DI GROWIN KETAT 3-5%!"
    else:
        kategori_risiko = f"RENDAH/AMAN (Beta: {round(beta, 2)}) 🛡️"
        status_proteksi = "AMAN UNTUK STRATEGI GABUNGAN (Bisa Tanpa Cut Loss)"

        if is_panic_selling:
            status_tren = "DOWNTREND DISKONTINU 📉"
            rekomendasi = "ANTRE BELI SUPER PASIF (Institusi sedang jualan, tunggu reda)"
        elif harga_sekarang > ema20 and ema20 > ema50:
            status_tren = "UPTREND 📈"
            rekomendasi = "BUY ON WEAKNESS (Antre Beli di GROWIN dekat EMA20)" if harga_sekarang <= (ema20 * 1.015) else "HOLD (Tunggu Koreksi Sehat)"
        elif harga_sekarang < ema20 and harga_sekarang > ema50:
            status_tren = "KOREKSI DALAM 📉"
            rekomendasi = "WAIT AND SEE (Tunggu Sentuh EMA50)"
        elif harga_sekarang < ema50:
            status_tren = "ZONA DISKON / BEARISH SEMANTARA 📉"
            rekomendasi = "ZONA SEROK / AKUMULASI (Harga Murah di Bawah EMA50)"
        else:
            status_tren = "KONSOLIDASI 📊"
            rekomendasi = "WAIT AND SEE"

    if f1_kondisi and not wajib_stop_loss and not is_panic_selling:
        rekomendasi = "BUY ON WEAKNESS ★★★ (Konfirmasi Oversold Forum Aktif!)"
    elif f2_kondisi and not wajib_stop_loss:
        rekomendasi = "STRONG BUY / MOMENTUM RIDE 🚀 (Konfirmasi Tren ADX Meledak!)"

    if kondisi_market is None:
        kondisi_market = cek_kondisi_market()
    if not kondisi_market.get("market_bullish", True) and "BUY" in rekomendasi and not wajib_stop_loss:
        rekomendasi = f"{rekomendasi} (⚠️ IHSG sedang BEARISH, pertimbangkan kurangi ukuran posisi)"

    # --- E. TEXT PENJELASAN OTOMATIS ---
    posisi_pos = "di atas" if harga_sekarang > ema20 else "di bawah"
    vol_text = "disertai volume tinggi" if is_volume_strong else "dengan volume cenderung rendah"
    penjelasan_chart = f"Harga {ticker_symbol.upper()} (Rp{harga_sekarang}) berada {posisi_pos} garis acuan EMA 20 (Rp{ema20}). Pergerakan harian berjalan {vol_text}. Grafik menunjukkan kondisi {status_tren}. Arus institusi saat ini terdeteksi {status_arus_modal}."
    panduan_saran_growin = f"1. Pasang Auto Order Beli pertama sedekat mungkin dengan lantai EMA 20 di area Rp{ema20}. 2. Jika Anda menerapkan investasi jangka panjang tanpa cut loss, siapkan peluru serok kedua di area benteng EMA 50 (Rp{ema50}). 3. Set jaring jual otomatis Take Profit GTC langsung di atap resisten Rp{resisten_terdekat}."

    # --- F. ZONA AVERAGE DOWN + GUARDRAIL FUNDAMENTAL ---
    zona_average_down = None
    guardrail_fundamental = None
    if not wajib_stop_loss:
        zona_average_down = hitung_zona_average_down(harga_sekarang, ema20, ema50, ema200, area_support_kuat)
        guardrail_fundamental = cek_guardrail_fundamental(saham, info)
        if not guardrail_fundamental["aman_untuk_average_down"]:
            rekomendasi = f"{rekomendasi} | ⚠️ GUARDRAIL FUNDAMENTAL AKTIF: pertimbangkan HENTIKAN average down, cek alasan di field guardrail_fundamental"

    # --- G. SARAN MANAJEMEN RISIKO POSISI (stateless, cuma saran batas, bukan tracking) ---
    manajemen_risiko = {
        "maks_alokasi_modal_persen": MAX_ALOKASI_SWING_PERSEN,
        "keterangan": "Saran batas alokasi modal ke SATU saham ini. Sistem tidak melacak posisi lain yang sudah kamu buka (stateless), jadi total across saham tetap perlu kamu catat manual."
    }

    return {
        "saham": ticker_symbol.upper(),
        "harga_saat_ini": harga_sekarang,
        "fundamental": {
            "harga_wajar": harga_wajar,
            "harga_maks_layak_beli": harga_maks_layak_beli,
            "pbv_ratio": round(pbv_ratio, 2) if pbv_ratio else "N/A",
            "return_on_equity": f"{round(return_on_equity * 100, 2)}%" if return_on_equity else "N/A",
            "status_dividen": status_dividen
        },
        "teknikal": {
            "status_tren": status_tren,
            "klasifikasi_lantai": klasifikasi_support,
            "target_atap_resisten": f"Rp{resisten_terdekat} (Potensi ruang kenaikan: +{jarak_ke_resisten}%)",
            "ema20": ema20, "ema50": ema50, "ema200": ema200,
            "rsi_14": round(rsi, 2), "stochastic_d": round(stoch_d, 2), "adx_strength": round(adx, 2),
            "status_arus_modal": status_arus_modal,
            "konfirmasi_oversold_swing": status_forum_swing,
            "oversold_swing_aktif": bool(f1_kondisi),
            "macd_early_rebound_terdeteksi": sinyal_macd_early_rebound,
            "konfirmasi_daytrading_adx": status_forum_day,
            "penjelasan_chart": penjelasan_chart,
            "panduan_saran_growin": panduan_saran_growin
        },
        "guardrail_proteksi": {
            "kategori_risiko": kategori_risiko,
            "aturan_akun": status_proteksi,
            "wajib_stop_loss": wajib_stop_loss
        },
        "kondisi_market": kondisi_market,
        "zona_average_down": zona_average_down,
        "guardrail_fundamental": guardrail_fundamental,
        "manajemen_risiko": manajemen_risiko,
        "rekomendasi_akhir": rekomendasi
    }


# =========================================================================
# STRATEGI 2: SCREENER MOMENTUM GORENGAN
# =========================================================================

def hitung_momentum_gorengan(ticker_symbol: str, df_riwayat: pd.DataFrame = None):
    if not ticker_symbol.endswith(".JK"):
        ticker = f"{ticker_symbol.upper()}.JK"
    else:
        ticker = ticker_symbol.upper()

    saham = yf.Ticker(ticker)

    if df_riwayat is not None:
        df = df_riwayat.copy()
    else:
        df = saham.history(period="60d", interval="1h", auto_adjust=False)

    if df.empty or len(df) < 40:
        return None

    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df = hitung_indikator_lengkap(df)

    terakhir = df.iloc[-1]
    harga_sekarang = int(terakhir['Close'])
    ema5, ema10 = terakhir['EMA5'], terakhir['EMA10']
    rsi, adx, plus_di, minus_di = terakhir['RSI14'], terakhir['ADX14'], terakhir['+DI14'], terakhir['-DI14']
    atr = terakhir['ATR14']

    volume_terakhir = terakhir['Volume']
    volume_rata_rata = df['Volume'].iloc[:-1].tail(35).mean()

    is_volume_spike = volume_rata_rata > 0 and volume_terakhir > (volume_rata_rata * 2.5)
    is_bullish_momentum = harga_sekarang > ema5 and ema5 > ema10
    is_trend_explosive = adx > ADX_THRESHOLD_GORENGAN and plus_di > minus_di

    if pd.notna(atr) and atr > 0:
        cl_level = int(harga_sekarang - (ATR_MULTIPLIER_SL * atr))
        tp_level = int(harga_sekarang + (ATR_MULTIPLIER_TP * atr))
        metode_tp_sl = "ATR-based"
    else:
        cl_level = int(harga_sekarang * 0.965)
        tp_level = int(harga_sekarang * 1.07)
        metode_tp_sl = "Fixed % (fallback)"

    trailing_stop_saran = int(ema5) if harga_sekarang > ema5 else cl_level

    try:
        info = saham.info
        beta = info.get('beta', 1.8) if info else 1.8
    except Exception:
        beta = 1.8

    status_filter = "GAGAL 💤"
    if is_volume_spike and is_bullish_momentum and is_trend_explosive:
        status_filter = "LOLOS SCREENING 🔥 (Ledakan ADX + Bandar Masuk!)"

    manajemen_risiko = {
        "maks_alokasi_modal_persen": MAX_ALOKASI_GORENGAN_PERSEN,
        "maks_posisi_bersamaan_disarankan": MAX_JUMLAH_GORENGAN_BERSAMAAN,
        "keterangan": "Saran batas per posisi gorengan. Sistem tidak melacak berapa posisi gorengan yang sudah kamu buka bersamaan (stateless) — kamu perlu jaga disiplin ini manual."
    }

    return {
        "saham": ticker_symbol.upper(),
        "harga_saat_ini": harga_sekarang,
        "status_filter": status_filter,
        "indikator": {
            "lonjakan_volume": f"{round(volume_terakhir / volume_rata_rata, 1)}x Lipat" if volume_rata_rata > 0 else "N/A",
            "rsi_momentum": round(rsi, 2),
            "adx_power": round(adx, 2),
            "atr_volatilitas": round(float(atr), 2) if pd.notna(atr) else "N/A",
            "tingkat_volatilitas_beta": round(beta, 2)
        },
        "bracket_order_growin": {
            "target_take_profit": tp_level,
            "batas_cut_loss": cl_level,
            "trailing_stop_saran": trailing_stop_saran,
            "metode": metode_tp_sl
        },
        "manajemen_risiko": manajemen_risiko,
        "peringatan_keamanan": "RESIKO EKSTREM! Pergerakan harga murni ledakan tren momentum intraday.",
        "rekomendasi_aksi": "DAY TRADING CEPAT - WAJIB LANGSUNG SET AUTO ORDER STOP LOSS DI GROWIN!"
    }
