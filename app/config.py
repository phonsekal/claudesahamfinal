# app/config.py
TARGET_DIVIDEND_YIELD = 0.07
PE_WAJAR_BANK = 12.0
PE_WAJAR_UMUM = 15.0

# Ambang minimal yield dividen supaya dianggap layak jadi BASIS VALUASI utama.
# Saham yang secara teknis bagi dividen tapi nominalnya receh (di bawah ambang ini)
# diperlakukan seperti saham non-dividen untuk keperluan valuasi (pakai harga wajar
# PE-based sebagai acuan), bukan rumus dividend-yield yang bisa menghasilkan angka
# tidak masuk akal untuk yield yang sangat kecil.
YIELD_MINIMAL_UNTUK_VALUASI_DIVIDEN = 0.01  # 1%

# Watchlist awal untuk strategi Swing-Investment Dividen (Big & Medium Caps)
INDEX_BLUECHIP_UTAMA = ["BBRI.JK", "BMRI.JK", "BBNI.JK", "BBCA.JK", "ASII.JK", "TLKM.JK", "UNVR.JK", "PTBA.JK", "ADRO.JK", "ANTM.JK", "ICBP.JK", "INDF.JK", "AMRT.JK", "SIDO.JK"]

# Watchlist awal untuk strategi Saham Gorengan Spekulatif (High Volatility / Penny Stocks)
WATCHLIST_GORENGAN = ["JGLE.JK", "BUMI.JK", "BRMS.JK", "DEWA.JK", "DOOH.JK", "GOTO.JK", "WIFI.JK", "JKON.JK"]

# --- KONFIGURASI TAMBAHAN (STRATEGI YANG DIMAKSIMALKAN) ---

# Ticker indeks acuan untuk filter kondisi makro market sebelum sinyal individual dipakai
MARKET_INDEX_TICKER = "^JKSE"

# Ambang ADX untuk strategi gorengan (20 = default longgar, 25 = lebih ketat/selektif)
ADX_THRESHOLD_GORENGAN = 20.0

# Kelipatan ATR untuk menghitung TP/SL gorengan secara proporsional ke volatilitas saham,
# menggantikan angka persentase flat (7% / 3.5%) yang sama rata untuk semua saham
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 3.0

# Alokasi bertingkat (persen) untuk 3 tranche average-down di strategi dividen.
# Alokasi makin besar di level koreksi yang makin dalam.
TRANCHE_ALLOKASI = [20, 30, 50]

# Batas toleransi penurunan dividen tahun berjalan vs tahun lalu sebelum guardrail
# fundamental menyatakan "tidak aman untuk average down" (circuit breaker pengganti cut loss)
BATAS_TOLERANSI_PENURUNAN_DIVIDEN = 0.30  # 30%

# --- KONFIGURASI RETRY (ketahanan terhadap kegagalan sesaat Yahoo Finance) ---
RETRY_PERCOBAAN_MAKSIMAL = 2
RETRY_JEDA_DETIK = 1.5

# --- KONFIGURASI MANAJEMEN RISIKO PORTOFOLIO (saran ukuran posisi) ---
# CATATAN: API ini stateless (tidak ada database), jadi angka di bawah ini adalah
# SARAN batas per posisi, bukan pelacakan otomatis posisi yang sudah kamu buka.
# Kamu tetap perlu mencatat sendiri total alokasi aktifmu di luar sistem ini,
# kecuali nanti ditambahkan lapisan penyimpanan (database) terpisah.
MAX_ALOKASI_SWING_PERSEN = 15          # maks % modal per saham dividen/swing
MAX_ALOKASI_GORENGAN_PERSEN = 5        # maks % modal per saham gorengan (lebih kecil, risiko tinggi)
MAX_JUMLAH_GORENGAN_BERSAMAAN = 3      # saran jumlah maksimal posisi gorengan dibuka bersamaan

# --- UNIVERSE SAHAM UNTUK SCREENER "SEMUA SAHAM IDX" ---
# PENTING: Ini BUKAN daftar lengkap seluruh ~900 saham yang terdaftar di IDX.
# Ini starter list ~100 ticker yang relatif likuid/dikenal luas, disusun dari
# pengetahuan umum publik (bukan hasil scraping dataset pihak ketiga mana pun,
# untuk menghindari isu lisensi/hak cipta atas kompilasi data tersebut).
#
# Kalau kamu mau cakupan penuh seluruh saham IDX, lengkapi list ini sendiri dari
# sumber resmi: https://www.idx.co.id/id/perusahaan-tercatat/profil-perusahaan-tercatat
# (unduh daftar resmi, lalu tambahkan kode ticker + akhiran ".JK" ke list ini).
SEMUA_SAHAM_IDX_STARTER = [
    # Perbankan
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK", "ARTO.JK", "BJBR.JK", "BJTM.JK", "BTPS.JK", "BNGA.JK",
    # Consumer / Ritel
    "UNVR.JK", "ICBP.JK", "INDF.JK", "MYOR.JK", "ULTJ.JK", "SIDO.JK", "AMRT.JK", "MIDI.JK", "MAPI.JK", "ACES.JK",
    "ERAA.JK", "LPPF.JK", "RALS.JK", "CPIN.JK", "JPFA.JK", "GGRM.JK", "HMSP.JK", "KLBF.JK", "KAEF.JK", "TSPC.JK",
    # Telco / Infrastruktur Digital
    "TLKM.JK", "EXCL.JK", "ISAT.JK", "TOWR.JK", "TBIG.JK", "GOTO.JK", "BUKA.JK", "EMTK.JK", "WIFI.JK",
    # Otomotif / Manufaktur
    "ASII.JK", "UNTR.JK", "AUTO.JK", "SMSM.JK",
    # Pertambangan / Energi
    "PTBA.JK", "ADRO.JK", "ITMG.JK", "INDY.JK", "HRUM.JK", "ANTM.JK", "INCO.JK", "MDKA.JK", "TINS.JK", "BUMI.JK",
    "BRMS.JK", "MEDC.JK", "ELSA.JK", "PGAS.JK", "AKRA.JK", "BSSR.JK", "DSSA.JK", "PSAB.JK", "MBMA.JK", "AMMN.JK",
    "NCKL.JK",
    # Semen / Properti / Konstruksi
    "SMGR.JK", "INTP.JK", "WSKT.JK", "WIKA.JK", "PTPP.JK", "JSMR.JK", "PWON.JK", "BSDE.JK", "CTRA.JK", "SMRA.JK",
    "LPKR.JK", "ASRI.JK",
    # Perkebunan
    "AALI.JK", "LSIP.JK", "SMAR.JK", "TAPG.JK",
    # Media / Teknologi
    "SCMA.JK", "MNCN.JK",
    # Kimia / Petrokimia
    "BRPT.JK", "TPIA.JK",
    # Saham Gorengan / High Volatility
    "DEWA.JK", "DOOH.JK", "JGLE.JK", "JKON.JK", "BULL.JK", "RAJA.JK", "PANI.JK", "BREN.JK", "CUAN.JK",
]
