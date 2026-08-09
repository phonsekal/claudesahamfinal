# app/routers.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import APIRouter, HTTPException, Query
from app.services import (
    hitung_analisis_saham, hitung_momentum_gorengan, cek_kondisi_market,
    ambil_riwayat_batch
)
from app.backtest import (
    backtest_swing_dividen, backtest_gorengan_momentum,
    backtest_watchlist_swing, backtest_watchlist_gorengan
)
from app.config import INDEX_BLUECHIP_UTAMA, WATCHLIST_GORENGAN, SEMUA_SAHAM_IDX_STARTER

router = APIRouter(prefix="/v1")

# ThreadPoolExecutor dipakai untuk paralelisasi network I/O (yfinance). Angka worker
# dijaga moderat supaya tidak kena rate-limit Yahoo Finance.
MAX_WORKERS_SCREENER = 5

# Batas ukuran batch untuk endpoint "screener semua saham IDX". Vercel serverless
# punya batas waktu eksekusi (10-60 detik tergantung plan) - scan >900 saham IDX
# TIDAK MUAT dalam satu request. Karena itu endpoint ini dipaginasi: panggil
# berkali-kali dengan offset berbeda (atau jadwalkan lewat n8n/cron) untuk cover
# seluruh universe secara bertahap.
BATAS_LIMIT_MAKSIMAL_PER_PANGGILAN = 60


@router.get("/analisis/swing/{ticker}")
async def analisis_swing_saham(ticker: str):
    """Analisis lengkap satu saham (fundamental + teknikal) untuk strategi swing-dividen."""
    data = hitung_analisis_saham(ticker)
    if not data:
        raise HTTPException(status_code=404, detail=f"Data untuk {ticker.upper()} tidak ditemukan atau tidak lengkap.")
    return {"status": "success", "data": data}


@router.get("/analisis/gorengan/{ticker}")
async def analisis_gorengan_saham(ticker: str):
    """Analisis momentum satu saham untuk strategi day-trading ADX."""
    data = hitung_momentum_gorengan(ticker)
    if not data:
        raise HTTPException(status_code=404, detail=f"Data untuk {ticker.upper()} tidak ditemukan atau tidak lengkap.")
    return {"status": "success", "data": data}


@router.get("/market/status")
async def status_market():
    """Cek kondisi tren IHSG saat ini sebagai filter makro sebelum sinyal individual dipakai."""
    return {"status": "success", "data": cek_kondisi_market()}


def _jalankan_screener_swing(daftar_ticker: list):
    """
    Screener swing generik: fetch riwayat harga SEMUA ticker sekaligus lewat batch
    download (lebih cepat & lebih hemat request dibanding satu-satu), lalu proses
    tiap saham secara paralel (ThreadPoolExecutor) hanya untuk bagian .info yang
    memang tidak bisa di-batch oleh yfinance.
    """
    kondisi_market = cek_kondisi_market()
    tickers_jk = [t if t.endswith(".JK") else f"{t.upper()}.JK" for t in daftar_ticker]
    data_riwayat = ambil_riwayat_batch(tickers_jk, period="1y", interval="1d")

    saham_lolos = []
    saham_gagal = []

    def proses(ticker_jk):
        symbol = ticker_jk.replace(".JK", "")
        df_riwayat = data_riwayat.get(ticker_jk)
        try:
            return symbol, hitung_analisis_saham(symbol, kondisi_market=kondisi_market, df_riwayat=df_riwayat)
        except Exception:
            return symbol, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_SCREENER) as executor:
        futures = [executor.submit(proses, t) for t in tickers_jk]
        for future in as_completed(futures):
            symbol, data = future.result()
            if not data:
                saham_gagal.append(symbol)
                continue
            teknikal = data.get("teknikal", {})
            if teknikal.get("oversold_swing_aktif"):
                saham_lolos.append({
                    "saham": data.get("saham"),
                    "harga_saat_ini": data.get("harga_saat_ini"),
                    "status_tren": teknikal.get("status_tren"),
                    "konfirmasi_oversold_swing": teknikal.get("konfirmasi_oversold_swing"),
                    "status_dividen": data.get("fundamental", {}).get("status_dividen"),
                    "zona_average_down": data.get("zona_average_down"),
                    "guardrail_fundamental": data.get("guardrail_fundamental"),
                    "manajemen_risiko": data.get("manajemen_risiko"),
                    "rekomendasi": data.get("rekomendasi_akhir")
                })

    return kondisi_market, saham_lolos, saham_gagal


def _jalankan_screener_gorengan(daftar_ticker: list):
    """Screener gorengan generik dengan batch fetch data intraday."""
    tickers_jk = [t if t.endswith(".JK") else f"{t.upper()}.JK" for t in daftar_ticker]
    data_riwayat = ambil_riwayat_batch(tickers_jk, period="60d", interval="1h")

    saham_lolos = []
    saham_gagal = []

    def proses(ticker_jk):
        symbol = ticker_jk.replace(".JK", "")
        df_riwayat = data_riwayat.get(ticker_jk)
        try:
            return symbol, hitung_momentum_gorengan(symbol, df_riwayat=df_riwayat)
        except Exception:
            return symbol, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_SCREENER) as executor:
        futures = [executor.submit(proses, t) for t in tickers_jk]
        for future in as_completed(futures):
            symbol, data = future.result()
            if not data:
                saham_gagal.append(symbol)
                continue
            if "LOLOS" in data.get("status_filter", ""):
                saham_lolos.append({
                    "saham": data.get("saham"),
                    "status": data.get("status_filter"),
                    "rsi_momentum": data.get("indikator", {}).get("rsi_momentum"),
                    "adx_power": data.get("indikator", {}).get("adx_power"),
                    "kualitas_tren_adx": data.get("indikator", {}).get("kualitas_tren_adx"),
                    "rekomendasi_entry_daytrading": data.get("rekomendasi_entry_daytrading"),
                    "bracket_order_growin": data.get("bracket_order_growin"),
                    "manajemen_risiko": data.get("manajemen_risiko"),
                    "rekomendasi": data.get("rekomendasi_aksi")
                })

    return saham_lolos, saham_gagal


@router.get("/screener/swing-dividen")
async def run_screener_swing_dividen():
    kondisi_market, saham_lolos, saham_gagal = _jalankan_screener_swing(INDEX_BLUECHIP_UTAMA)
    return {
        "status": "success",
        "kondisi_market": kondisi_market,
        "data": saham_lolos,
        "saham_gagal_dianalisis": saham_gagal
    }


@router.get("/screener/gorengan-momentum")
async def run_screener_gorengan_momentum():
    saham_lolos, saham_gagal = _jalankan_screener_gorengan(WATCHLIST_GORENGAN)
    return {
        "status": "success",
        "radar_saham_gorengan_aktif": saham_lolos,
        "saham_gagal_dianalisis": saham_gagal
    }


@router.get("/screener/swing-dividen/semua-saham")
async def run_screener_swing_semua_saham(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=BATAS_LIMIT_MAKSIMAL_PER_PANGGILAN)):
    """
    Screener swing-dividen atas starter universe SEMUA_SAHAM_IDX_STARTER (bukan cuma
    watchlist bluechip). DIPAGINASI karena scan seluruh saham IDX tidak muat dalam
    1 request serverless — panggil berulang dengan offset berbeda untuk cover semua.
    Contoh: offset=0&limit=50, lalu offset=50&limit=50, dst.
    """
    total_saham = len(SEMUA_SAHAM_IDX_STARTER)
    slice_ticker = SEMUA_SAHAM_IDX_STARTER[offset: offset + limit]
    if not slice_ticker:
        raise HTTPException(status_code=404, detail=f"Offset {offset} di luar jangkauan. Total saham di starter list: {total_saham}.")

    kondisi_market, saham_lolos, saham_gagal = _jalankan_screener_swing(slice_ticker)
    offset_berikutnya = offset + limit if (offset + limit) < total_saham else None

    return {
        "status": "success",
        "kondisi_market": kondisi_market,
        "total_saham_di_starter_list": total_saham,
        "diproses_offset": offset,
        "diproses_limit": limit,
        "offset_berikutnya": offset_berikutnya,
        "data": saham_lolos,
        "saham_gagal_dianalisis": saham_gagal
    }


@router.get("/screener/gorengan-momentum/semua-saham")
async def run_screener_gorengan_semua_saham(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=BATAS_LIMIT_MAKSIMAL_PER_PANGGILAN)):
    """Sama seperti swing/semua-saham tapi untuk strategi momentum gorengan. Juga dipaginasi."""
    total_saham = len(SEMUA_SAHAM_IDX_STARTER)
    slice_ticker = SEMUA_SAHAM_IDX_STARTER[offset: offset + limit]
    if not slice_ticker:
        raise HTTPException(status_code=404, detail=f"Offset {offset} di luar jangkauan. Total saham di starter list: {total_saham}.")

    saham_lolos, saham_gagal = _jalankan_screener_gorengan(slice_ticker)
    offset_berikutnya = offset + limit if (offset + limit) < total_saham else None

    return {
        "status": "success",
        "total_saham_di_starter_list": total_saham,
        "diproses_offset": offset,
        "diproses_limit": limit,
        "offset_berikutnya": offset_berikutnya,
        "radar_saham_gorengan_aktif": saham_lolos,
        "saham_gagal_dianalisis": saham_gagal
    }


@router.get("/backtest/swing/{ticker}")
async def backtest_swing(ticker: str, tahun: int = 2):
    """Backtest strategi swing-oversold pada data historis harian 1 saham."""
    hasil = backtest_swing_dividen(ticker, tahun=tahun)
    if not hasil:
        raise HTTPException(status_code=404, detail=f"Data historis untuk {ticker.upper()} tidak cukup untuk backtest.")
    return {"status": "success", "data": hasil}


@router.get("/backtest/gorengan/{ticker}")
async def backtest_gorengan(ticker: str):
    """Backtest strategi momentum gorengan pada data historis 60 hari interval 1 jam."""
    hasil = backtest_gorengan_momentum(ticker)
    if not hasil:
        raise HTTPException(status_code=404, detail=f"Data historis untuk {ticker.upper()} tidak cukup untuk backtest.")
    return {"status": "success", "data": hasil}


@router.get("/backtest/swing/watchlist/gabungan")
async def backtest_swing_gabungan(tahun: int = 2):
    """Backtest strategi swing-oversold di SELURUH watchlist bluechip sekaligus, digabungkan (lebih valid secara statistik)."""
    hasil = backtest_watchlist_swing(tahun=tahun)
    return {"status": "success", "data": hasil}


@router.get("/backtest/gorengan/watchlist/gabungan")
async def backtest_gorengan_gabungan():
    """Backtest strategi momentum gorengan di SELURUH watchlist gorengan sekaligus, digabungkan."""
    hasil = backtest_watchlist_gorengan()
    return {"status": "success", "data": hasil}
