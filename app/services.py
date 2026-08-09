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
    MAX_ALOKASI_SWING_PERSEN, MAX_ALOKASI_GORENGAN_PERSEN, MAX_JUMLAH_GORENGAN_BERSAMAAN,
    YIELD_MINIMAL_UNTUK_VALUASI_DIVIDEN,
    ADX_TREN_MODERAT, ADX_TREN_KUAT, ADX_TREN_EKSTREM,
    FEE_TRANSAKSI_TOTAL_PERSEN, MIN_PROFIT_BERSIH_DAYTRADING_PERSEN
)
import datetime


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
    """
    Menghitung RSI, Stochastic, MACD, ATR, ADX, dan DI+/DI- secara native.

    RSI, ATR, dan ADX memakai Wilder's smoothing (ewm alpha=1/period), BUKAN rata-rata
    rolling biasa. Ini formula standar industri — nilai yang dihasilkan konsisten dengan
    chart di TradingView/aplikasi sekuritas, sehingga ambang seperti ADX > 20 punya makna
    yang sama dengan literatur. Rolling mean biasa membuat ADX "loncat-loncat" (nilai lama
    keluar dari window sekaligus) dan umumnya lebih lambat mendeteksi tren baru.
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / period, adjust=False).mean()
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
    df['DownMove'] = -df['Low'].diff()

    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0.0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0.0)

    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

    tr_smooth = df['TR'].ewm(alpha=1 / period, adjust=False).mean()
    df['ATR14'] = tr_smooth

    plus_dm_smooth = df['+DM'].ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_smooth = df['-DM'].ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (plus_dm_smooth / (tr_smooth + 1e-10))
    minus_di = 100 * (minus_dm_smooth / (tr_smooth + 1e-10))

    df['+DI14'] = plus_di
    df['-DI14'] = minus_di

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    df['ADX14'] = dx.ewm(alpha=1 / period, adjust=False).mean()

    # Chaikin Money Flow 20-periode: proxy "arus bandar" dari data harga+volume.
    # Mengukur apakah volume lebih banyak terjadi saat harga ditutup dekat HIGH
    # (tekanan beli / akumulasi) atau dekat LOW (tekanan jual / distribusi).
    # Bukan pengganti broker summary asli, tapi menangkap fenomena yang sama
    # tanpa butuh sumber data berbayar.
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    df['CMF20'] = (mfm * df['Volume']).rolling(window=20).sum() / (df['Volume'].rolling(window=20).sum() + 1e-10)

    return df


def bulatkan_ke_tick_idx(harga, ke_bawah=True):
    """
    Bulatkan harga ke fraksi harga (tick size) resmi BEI supaya harga rekomendasi
    bisa langsung dipakai antre order tanpa ditolak sistem broker.
    ke_bawah=True untuk harga beli (konservatif), False untuk harga jual.
    """
    if harga < 200:
        tick = 1
    elif harga < 500:
        tick = 2
    elif harga < 2000:
        tick = 5
    elif harga < 5000:
        tick = 10
    else:
        tick = 25
    if ke_bawah:
        return int(harga // tick * tick)
    return int(-(-harga // tick) * tick)


def nilai_kualitas_tren_adx(adx, plus_di, minus_di):
    """Klasifikasi kualitas tren berdasarkan kekuatan ADX + arah dominan DI."""
    # bool() eksplisit: nilai dari pandas adalah numpy.bool_ yang gagal diserialisasi JSON
    arah_bullish = bool(plus_di > minus_di)
    if adx >= ADX_TREN_EKSTREM:
        label = "SANGAT KUAT / EKSTREM ⚡ (waspada tren sudah matang, rawan pembalikan)"
    elif adx >= ADX_TREN_KUAT:
        label = "KUAT 🚀 (zona ideal untuk momentum trading)"
    elif adx >= ADX_TREN_MODERAT:
        label = "MODERAT 📈 (tren baru terbentuk, butuh konfirmasi volume)"
    else:
        label = "LEMAH / SIDEWAYS 💤 (ADX di bawah 20, tidak ada tren jelas)"

    tren_bagus = bool(adx >= ADX_TREN_MODERAT) and arah_bullish
    return {
        "tren_bagus_untuk_daytrading": tren_bagus,
        "kekuatan": label,
        "arah": "BULLISH (DI+ dominan)" if arah_bullish else "BEARISH (DI- dominan)"
    }


def interpretasi_arus_bandar_cmf(cmf):
    """
    Terjemahkan nilai Chaikin Money Flow jadi status arus bandar yang mudah dibaca.
    Ambang +/-0.05 dan +/-0.15 adalah konvensi umum interpretasi CMF.
    """
    if cmf is None or (isinstance(cmf, float) and pd.isna(cmf)):
        return {"cmf_20": None, "status": "TIDAK TERSEDIA", "penjelasan": "Data belum cukup untuk menghitung CMF 20-periode."}
    cmf = float(cmf)
    if cmf >= 0.15:
        status = "AKUMULASI KUAT 🟢🟢"
        penjelasan = "Volume terkonsentrasi saat harga ditutup dekat high — indikasi kuat ada pihak besar mengakumulasi."
    elif cmf >= 0.05:
        status = "AKUMULASI 🟢"
        penjelasan = "Tekanan beli lebih dominan dari tekanan jual dalam 20 periode terakhir."
    elif cmf > -0.05:
        status = "NETRAL ⚪"
        penjelasan = "Tidak ada dominasi arus dana yang jelas — pasar seimbang."
    elif cmf > -0.15:
        status = "DISTRIBUSI 🔴"
        penjelasan = "Tekanan jual lebih dominan — hati-hati, ada indikasi pihak besar mengurangi posisi."
    else:
        status = "DISTRIBUSI KUAT 🔴🔴"
        penjelasan = "Volume terkonsentrasi saat harga ditutup dekat low — indikasi kuat distribusi besar sedang berlangsung."
    return {"cmf_20": round(cmf, 3), "status": status, "penjelasan": penjelasan}


def hitung_rekomendasi_entry_daytrading(harga_sekarang, ema_pullback, atr, target_jual,
                                        adx, plus_di, minus_di):
    """
    Rekomendasi harga masuk untuk daytrading saat tren ADX bagus.

    - harga_entry_terbaik: antre limit beli di area pullback sehat — sekitar EMA acuan
      atau setengah ATR di bawah harga sekarang (mana yang lebih tinggi), supaya tidak
      mengejar harga di pucuk.
    - harga_masuk_maksimal: harga TERTINGGI yang masih layak dibeli. Di atas harga ini,
      kalaupun target jual tercapai, profit bersih setelah fee broker sudah di bawah
      ambang minimal (MIN_PROFIT_BERSIH_DAYTRADING_PERSEN) — artinya risk/reward tidak
      lagi sepadan.
    """
    kualitas = nilai_kualitas_tren_adx(adx, plus_di, minus_di)
    if not kualitas["tren_bagus_untuk_daytrading"]:
        return {
            "aktif": False,
            "kualitas_tren_adx": kualitas,
            "keterangan": "Tren ADX belum memenuhi syarat daytrading (butuh ADX >= 20 dengan DI+ dominan). Tidak ada rekomendasi harga masuk."
        }

    if not (atr and atr > 0) or harga_sekarang <= 0 or target_jual <= harga_sekarang:
        return {
            "aktif": False,
            "kualitas_tren_adx": kualitas,
            "keterangan": "Tren ADX bagus, tapi target jual sudah di bawah/sama dengan harga sekarang (harga di puncak resisten) atau ATR tidak tersedia — ruang profit tidak cukup untuk entry baru."
        }

    entry_terbaik = min(harga_sekarang, max(ema_pullback, harga_sekarang - 0.5 * atr))
    entry_terbaik = bulatkan_ke_tick_idx(entry_terbaik, ke_bawah=True)

    faktor_biaya = 1 + (FEE_TRANSAKSI_TOTAL_PERSEN + MIN_PROFIT_BERSIH_DAYTRADING_PERSEN) / 100
    harga_masuk_maksimal = bulatkan_ke_tick_idx(target_jual / faktor_biaya, ke_bawah=True)

    stop_loss = bulatkan_ke_tick_idx(entry_terbaik - (ATR_MULTIPLIER_SL * atr), ke_bawah=True)
    potensi_profit_dari_entry = round(float((target_jual - entry_terbaik) / entry_terbaik) * 100 - FEE_TRANSAKSI_TOTAL_PERSEN, 2)

    harga_sudah_kemahalan = bool(harga_sekarang > harga_masuk_maksimal)

    return {
        "aktif": True,
        "kualitas_tren_adx": kualitas,
        "harga_entry_terbaik": entry_terbaik,
        "harga_masuk_maksimal": harga_masuk_maksimal,
        "target_jual": bulatkan_ke_tick_idx(target_jual, ke_bawah=True),
        "stop_loss_disarankan": stop_loss,
        "estimasi_profit_bersih_dari_entry_terbaik_persen": potensi_profit_dari_entry,
        "keterangan": (
            f"⚠️ Harga sekarang (Rp{harga_sekarang}) SUDAH DI ATAS harga masuk maksimal (Rp{harga_masuk_maksimal}). "
            f"Mengejar di harga ini membuat profit bersih ke target jual di bawah {MIN_PROFIT_BERSIH_DAYTRADING_PERSEN}% setelah fee — lebih bijak menunggu pullback ke area entry terbaik."
            if harga_sudah_kemahalan else
            f"Antre limit beli di area Rp{entry_terbaik} (entry terbaik). Masih boleh masuk sampai maksimal Rp{harga_masuk_maksimal} — "
            f"di atas itu profit bersih ke target Rp{bulatkan_ke_tick_idx(target_jual, ke_bawah=True)} tidak lagi menutup fee + ambang profit minimal {MIN_PROFIT_BERSIH_DAYTRADING_PERSEN}%."
        ),
        "asumsi": f"Fee transaksi bolak-balik {FEE_TRANSAKSI_TOTAL_PERSEN}%, profit bersih minimal {MIN_PROFIT_BERSIH_DAYTRADING_PERSEN}%. Harga sudah dibulatkan ke fraksi harga resmi BEI."
    }


def buat_penjelasan_teknikal(rsi, stoch_d, macd, macd_signal, adx, plus_di, minus_di,
                             harga_sekarang, ema20, ema50, ema200, is_volume_strong, cmf=None):
    """Penjelasan singkat per indikator dalam bahasa sederhana, dihasilkan dari nilai aktual."""
    if rsi >= 70:
        rsi_text = f"RSI {round(rsi, 1)} — jenuh beli (overbought), rawan koreksi jangka pendek."
    elif rsi <= 30:
        rsi_text = f"RSI {round(rsi, 1)} — jenuh jual (oversold), potensi mantul jika ada konfirmasi."
    elif rsi >= 50:
        rsi_text = f"RSI {round(rsi, 1)} — momentum cenderung bullish (di atas garis tengah 50)."
    else:
        rsi_text = f"RSI {round(rsi, 1)} — momentum cenderung bearish (di bawah garis tengah 50)."

    if stoch_d <= 20:
        stoch_text = f"Stochastic %D {round(stoch_d, 1)} — area oversold, harga tertekan dan berpotensi jenuh jual."
    elif stoch_d >= 80:
        stoch_text = f"Stochastic %D {round(stoch_d, 1)} — area overbought, hati-hati aksi ambil untung."
    else:
        stoch_text = f"Stochastic %D {round(stoch_d, 1)} — area netral, belum ada sinyal jenuh."

    if macd > macd_signal and macd > 0:
        macd_text = "MACD di atas garis sinyal dan di atas nol — momentum naik sedang berjalan."
    elif macd > macd_signal:
        macd_text = "MACD baru memotong ke atas garis sinyal tapi masih di bawah nol — indikasi awal pembalikan naik (early rebound)."
    elif macd < 0:
        macd_text = "MACD di bawah garis sinyal dan di bawah nol — tekanan turun masih dominan."
    else:
        macd_text = "MACD di atas nol tapi melemah ke bawah garis sinyal — momentum naik mulai mendingin."

    kualitas_adx = nilai_kualitas_tren_adx(adx, plus_di, minus_di)
    adx_text = f"ADX {round(adx, 1)} — kekuatan tren {kualitas_adx['kekuatan']}, arah {kualitas_adx['arah']}."

    if harga_sekarang > ema20 and ema20 > ema50 and harga_sekarang > ema200:
        ema_text = "Harga di atas EMA20 > EMA50 dan di atas EMA200 — struktur tren naik sehat di semua kerangka waktu."
    elif harga_sekarang > ema200:
        ema_text = "Harga masih di atas EMA200 (tren besar naik) tapi sedang koreksi terhadap EMA jangka pendek."
    else:
        ema_text = "Harga di bawah EMA200 — tren besar masih turun, sinyal beli apapun berisiko melawan arus."

    vol_text = ("Volume hari ini di atas 1.5x rata-rata 20 hari — pergerakan dikonfirmasi partisipasi pasar yang nyata."
                if is_volume_strong else
                "Volume di bawah ambang 1.5x rata-rata — pergerakan harga belum dikonfirmasi volume, rawan sinyal palsu.")

    hasil = {
        "rsi": rsi_text,
        "stochastic": stoch_text,
        "macd": macd_text,
        "adx": adx_text,
        "posisi_ema": ema_text,
        "volume": vol_text
    }
    if cmf is not None:
        arus = interpretasi_arus_bandar_cmf(cmf)
        hasil["arus_bandar"] = f"CMF {arus['cmf_20']} — {arus['status']}. {arus['penjelasan']}"
    return hasil


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

def ambil_info_tanggal_dividen(info):
    """
    Coba ambil info tanggal terkait dividen dari Yahoo Finance.

    PENTING - keterbatasan yang perlu disadari:
    - Yahoo Finance TIDAK selalu punya data ini untuk saham IDX (cakupannya jauh lebih
      lengkap untuk saham AS). Field kosong BUKAN berarti saham tidak bagi dividen,
      bisa jadi cuma datanya tidak ter-cover Yahoo.
    - 'exDividendDate' dari Yahoo itu EX-DATE (tanggal saham mulai diperdagangkan TANPA
      hak dividen), BUKAN cum-date. Cum-date = hari bursa terakhir SEBELUM ex-date (hari
      terakhir kamu masih dapat hak dividen kalau beli saat itu). Di sini cum-date
      dihitung sebagai ESTIMASI (ex-date dikurangi 1 hari), bukan tanggal resmi.
    - Untuk kepastian jadwal cum-date, recording date (tanggal pencatatan), dan payment
      date (tanggal pembayaran) yang akurat, sumber otoritatifnya adalah pengumuman
      aksi korporasi resmi di idx.co.id atau KSEI — bukan Yahoo Finance.
    """
    ex_date_epoch = info.get('exDividendDate')
    if not ex_date_epoch:
        return {
            "tersedia": False,
            "catatan": "Data tanggal dividen tidak tersedia di Yahoo Finance untuk saham ini. Cek jadwal resmi di idx.co.id atau KSEI."
        }
    try:
        ex_date = datetime.datetime.utcfromtimestamp(ex_date_epoch).date()
        cum_date_perkiraan = ex_date - datetime.timedelta(days=1)
        return {
            "tersedia": True,
            "estimasi_cum_date": str(cum_date_perkiraan),
            "ex_dividend_date": str(ex_date),
            "catatan": "Cum-date di sini ESTIMASI (ex-date dikurangi 1 hari), bukan tanggal resmi. Verifikasi ke idx.co.id/KSEI sebelum mengambil keputusan berdasarkan tanggal ini."
        }
    except Exception:
        return {
            "tersedia": False,
            "catatan": "Gagal memproses data tanggal dividen dari Yahoo Finance."
        }


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
    # `or 0/1.0`: Yahoo kadang mengisi field dengan None (bukan menghilangkan key),
    # sehingga .get(key, default) tetap mengembalikan None dan perbandingan angka crash
    eps = info.get('trailingEps') or 0
    pbv_ratio = info.get('priceToBook') or 0
    return_on_equity = info.get('returnOnEquity') or 0
    beta = info.get('beta') or 1.0

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
        dividend_yield_persen = round((total_dividen / (harga_acuan or 1)) * 100, 2)
        if (dividend_yield_persen / 100) >= YIELD_MINIMAL_UNTUK_VALUASI_DIVIDEN:
            # Yield cukup signifikan untuk dijadikan basis valuasi utama
            harga_maks_layak_beli = int(total_dividen / TARGET_DIVIDEND_YIELD)
            status_dividen = f"LAYAK ({dividend_yield_persen}% Yield)"
            is_dividend_stock = True
        else:
            # Ada dividen tapi nominalnya receh - rumus dividend-yield di sini akan
            # menghasilkan angka tidak masuk akal (jauh di bawah harga wajar fundamental),
            # jadi fallback ke valuasi PE-based, dan diperlakukan sebagai bukan-dividend-stock
            # untuk keperluan guardrail risiko (nggak ada 'bantalan dividen' yang berarti).
            harga_maks_layak_beli = int(harga_wajar * 0.85)
            status_dividen = f"ADA DIVIDEN TAPI KECIL ({dividend_yield_persen}% Yield, di bawah ambang {int(YIELD_MINIMAL_UNTUK_VALUASI_DIVIDEN * 100)}%) ⚠️"
            is_dividend_stock = False
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
    cmf_terakhir = terakhir['CMF20'] if 'CMF20' in df.columns else None
    arus_bandar_cmf = interpretasi_arus_bandar_cmf(cmf_terakhir)

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

    f2_kondisi = (ema20 > ema50) and (rsi >= 50) and (adx > ADX_TREN_MODERAT) and (plus_di > minus_di) and (harga_sekarang > ema200) and is_volume_strong
    status_forum_day = "TREN SANGAT KUAT 🚀" if f2_kondisi else "TREN LEMAH / SIDEWAYS 💤"

    # --- REKOMENDASI HARGA ENTRY DAYTRADING (saat tren ADX bagus) ---
    # Target jual memakai resisten terdekat kalau masih ada ruang di atas harga,
    # kalau tidak (harga sudah di puncak) fallback ke proyeksi ATR.
    atr_terakhir = float(terakhir['ATR14']) if pd.notna(terakhir['ATR14']) else 0.0
    target_jual_daytrading = float(resisten_terdekat) if resisten_terdekat > harga_sekarang else harga_sekarang + (ATR_MULTIPLIER_TP * atr_terakhir)
    rekomendasi_daytrading = hitung_rekomendasi_entry_daytrading(
        harga_sekarang=harga_sekarang,
        ema_pullback=ema20,
        atr=atr_terakhir,
        target_jual=target_jual_daytrading,
        adx=adx, plus_di=plus_di, minus_di=minus_di
    )

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

    # --- CROSS-CHECK VALUASI vs REKOMENDASI ---
    # Rekomendasi berbasis teknikal/momentum (f2_kondisi dkk) dan harga_wajar berbasis
    # fundamental itu 2 sistem yang independen - tanpa cross-check, sistem bisa dengan
    # pede bilang "STRONG BUY" sementara harga sebenarnya sudah jauh di atas harga wajar,
    # tanpa ada yang menandai kontradiksi ini. Field ini bikin kontradiksinya EKSPLISIT
    # alih-alih tersembunyi di 2 angka berbeda yang harus dibandingkan manual oleh user.
    premi_terhadap_wajar_persen = None
    peringatan_valuasi = None
    if harga_wajar > 0:
        premi_terhadap_wajar_persen = round(((harga_sekarang - harga_wajar) / harga_wajar) * 100, 2)
        if premi_terhadap_wajar_persen > 20 and ("BUY" in rekomendasi or "STRONG" in rekomendasi):
            peringatan_valuasi = (
                f"⚠️ Harga saat ini {premi_terhadap_wajar_persen}% DI ATAS estimasi harga wajar fundamental (Rp{harga_wajar}). "
                f"Sinyal beli di atas murni berbasis momentum teknikal, BUKAN valuasi murah — "
                f"risiko koreksi lebih besar kalau momentum berbalik arah."
            )

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
            "status_dividen": status_dividen,
            "info_tanggal_dividen": ambil_info_tanggal_dividen(info)
        },
        "teknikal": {
            "status_tren": status_tren,
            "klasifikasi_lantai": klasifikasi_support,
            "target_atap_resisten": f"Rp{resisten_terdekat} (Potensi ruang kenaikan: +{jarak_ke_resisten}%)",
            "ema20": ema20, "ema50": ema50, "ema200": ema200,
            "rsi_14": round(rsi, 2), "stochastic_d": round(stoch_d, 2), "adx_strength": round(adx, 2),
            "plus_di_14": round(float(plus_di), 2), "minus_di_14": round(float(minus_di), 2),
            "macd": round(float(macd), 2),
            "macd_signal": round(float(terakhir['MACD_Signal']), 2),
            "macd_histogram": round(float(histogram_sekarang), 2),
            "status_arus_modal": status_arus_modal,
            "arus_bandar_cmf": arus_bandar_cmf,
            "konfirmasi_oversold_swing": status_forum_swing,
            "oversold_swing_aktif": bool(f1_kondisi),
            "macd_early_rebound_terdeteksi": sinyal_macd_early_rebound,
            "konfirmasi_daytrading_adx": status_forum_day,
            "rekomendasi_daytrading": rekomendasi_daytrading,
            "penjelasan_indikator": buat_penjelasan_teknikal(
                rsi, stoch_d, macd, float(terakhir['MACD_Signal']), adx, plus_di, minus_di,
                harga_sekarang, ema20, ema50, ema200, is_volume_strong, cmf=cmf_terakhir
            ),
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
        "premi_terhadap_harga_wajar_persen": premi_terhadap_wajar_persen,
        "peringatan_valuasi": peringatan_valuasi,
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
    arus_bandar_cmf = interpretasi_arus_bandar_cmf(terakhir['CMF20'] if 'CMF20' in df.columns else None)

    volume_terakhir = terakhir['Volume']
    volume_rata_rata = df['Volume'].iloc[:-1].tail(35).mean()

    is_volume_spike = volume_rata_rata > 0 and volume_terakhir > (volume_rata_rata * 2.5)
    is_bullish_momentum = harga_sekarang > ema5 and ema5 > ema10
    is_trend_explosive = adx > ADX_THRESHOLD_GORENGAN and plus_di > minus_di

    if pd.notna(atr) and atr > 0:
        cl_level = bulatkan_ke_tick_idx(harga_sekarang - (ATR_MULTIPLIER_SL * atr), ke_bawah=True)
        tp_level = bulatkan_ke_tick_idx(harga_sekarang + (ATR_MULTIPLIER_TP * atr), ke_bawah=True)
        metode_tp_sl = "ATR-based"
    else:
        cl_level = bulatkan_ke_tick_idx(harga_sekarang * 0.965, ke_bawah=True)
        tp_level = bulatkan_ke_tick_idx(harga_sekarang * 1.07, ke_bawah=True)
        metode_tp_sl = "Fixed % (fallback)"

    trailing_stop_saran = bulatkan_ke_tick_idx(ema5, ke_bawah=True) if harga_sekarang > ema5 else cl_level

    # Rekomendasi harga masuk (entry terbaik + batas masuk maksimal yang masih profit)
    # saat tren ADX intraday bagus. EMA5 dipakai sebagai acuan pullback intraday.
    atr_val = float(atr) if pd.notna(atr) and atr > 0 else harga_sekarang * 0.02
    rekomendasi_entry = hitung_rekomendasi_entry_daytrading(
        harga_sekarang=harga_sekarang,
        ema_pullback=float(ema5),
        atr=atr_val,
        target_jual=float(tp_level),
        adx=float(adx), plus_di=float(plus_di), minus_di=float(minus_di)
    )

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
            "kualitas_tren_adx": nilai_kualitas_tren_adx(float(adx), float(plus_di), float(minus_di)),
            "arus_bandar_cmf": arus_bandar_cmf,
            "atr_volatilitas": round(float(atr), 2) if pd.notna(atr) else "N/A",
            "tingkat_volatilitas_beta": round(beta, 2)
        },
        "penjelasan_indikator": {
            "volume": (f"Volume jam terakhir {round(volume_terakhir / volume_rata_rata, 1)}x lipat rata-rata — indikasi ada pihak besar masuk."
                       if is_volume_spike else "Belum ada lonjakan volume berarti (butuh > 2.5x rata-rata)."),
            "momentum": ("Harga di atas EMA5 > EMA10 — momentum intraday bullish."
                         if is_bullish_momentum else "Struktur EMA intraday belum bullish (harga belum di atas EMA5>EMA10)."),
            "adx": (f"ADX {round(float(adx), 1)} dengan DI+ dominan — tren intraday sedang meledak."
                    if is_trend_explosive else f"ADX {round(float(adx), 1)} — tren intraday belum cukup kuat/arah belum bullish."),
            "arus_bandar": f"CMF {arus_bandar_cmf['cmf_20']} — {arus_bandar_cmf['status']}. {arus_bandar_cmf['penjelasan']}"
        },
        "rekomendasi_entry_daytrading": rekomendasi_entry,
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
