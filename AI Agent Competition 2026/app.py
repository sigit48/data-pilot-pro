import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import os
import json
import hashlib
import hmac
import base64
import unicodedata
import re
import warnings
warnings.filterwarnings('ignore')

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

# Gemini AI imports
try:
    import google.generativeai as genai
    _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False

# Anomaly Detection imports
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import LocalOutlierFactor
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

# Advanced Forecasting imports
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _STATSMODELS_OK = True
except ImportError:
    _STATSMODELS_OK = False

try:
    from scipy import stats as scipy_stats
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

try:
    from prophet import Prophet
    _PROPHET_OK = True
except ImportError:
    _PROPHET_OK = False

# Image processing for Vision OCR
try:
    from PIL import Image
    import io as _io
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# ═══════════════════════════════════════════════════════════════
# ── LICENSE ENGINE (EMBEDDED — tidak perlu file eksternal) ──
# ═══════════════════════════════════════════════════════════════
#
# PENTING: Ganti _SECRET_SALT dengan string acak milik Anda sendiri
# sebelum publish ke GitHub. Jalankan di terminal untuk generate:
#   python -c "import secrets; print(secrets.token_hex(32))"
#
_SECRET_SALT = "datapilot-pro-2024-ganti-dengan-token-acak-milik-anda-min32c"
_PRODUCT_ID  = "DATAPILOT-PRO-2024"
_LICENSE_FILE = Path(".datapilot_license")

TIER_LIMITS = {
    "TRIAL":      {"max_rows": 500,    "label": "Trial (7 Hari)"},
    "BASIC":      {"max_rows": 5_000,  "label": "Basic"},
    "PRO":        {"max_rows": 500_000,"label": "Professional"},
    "ENTERPRISE": {"max_rows": 999_999,"label": "Enterprise"},
}

def _lic_sign(payload_str: str) -> str:
    return hmac.new(
        _SECRET_SALT.encode("utf-8"),
        (payload_str + _PRODUCT_ID).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:24].upper()

def generate_license_key(owner: str, expiry_days: int, tier: str = "PRO") -> str:
    """Buat license key baru. Jalankan sekali dari terminal atau script terpisah."""
    tier = tier.upper()
    if tier not in TIER_LIMITS:
        raise ValueError(f"Tier tidak valid: {tier}")
    expiry = (
        "UNLIMITED" if expiry_days == 0
        else (datetime.utcnow() + timedelta(days=expiry_days)).strftime("%Y%m%d")
    )
    payload = {"owner": owner, "expiry": expiry, "tier": tier,
               "product": _PRODUCT_ID, "issued": datetime.utcnow().strftime("%Y%m%d")}
    payload_str = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    # Base64 TIDAK di-uppercase — harus case-preserving agar decode benar
    b64 = base64.urlsafe_b64encode(payload_str.encode("utf-8")).decode("ascii")
    sig  = _lic_sign(payload_str)          # signature saja yang uppercase
    return f"DPP-{b64}-{sig}"

def _decode_key(key: str):
    """Parse + verifikasi key. Return payload dict atau None."""
    try:
        key = key.strip()
        # Prefix DPP- case-insensitive
        if not key.upper().startswith("DPP-"):
            return None
        rest = key[4:]   # buang "DPP-" (4 karakter)
        # Signature = 24 karakter UPPERCASE di ujung, dipisah "-"
        dash_pos = rest.rfind("-")
        if dash_pos < 0:
            return None
        b64_part  = rest[:dash_pos]          # case-preserving
        sig_given = rest[dash_pos + 1:].upper()
        if len(sig_given) != 24 or not b64_part:
            return None
        # Padding base64
        pad = 4 - len(b64_part) % 4
        if pad != 4:
            b64_part += "=" * pad
        payload_str = base64.urlsafe_b64decode(b64_part.encode("ascii")).decode("utf-8")
        sig_expected = _lic_sign(payload_str)
        if not hmac.compare_digest(sig_given, sig_expected):
            return None
        return json.loads(payload_str)
    except Exception:
        return None

def validate_license(key: str | None = None) -> dict:
    """Validasi lisensi. Cari dari: argumen → env var → file .datapilot_license"""
    INVALID = {"valid": False, "tier": "NONE", "owner": "", "expiry": "",
               "days_left": 0, "limits": TIER_LIMITS["TRIAL"], "message": ""}

    raw_key = key
    if not raw_key:
        raw_key = os.environ.get("DATAPILOT_KEY", "").strip()
    if not raw_key and _LICENSE_FILE.exists():
        try:
            raw_key = _LICENSE_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    if not raw_key:
        INVALID["message"] = "Tidak ada lisensi ditemukan."
        return INVALID

    payload = _decode_key(raw_key)
    if not payload:
        INVALID["message"] = "⛔ License key tidak valid atau telah dimodifikasi."
        return INVALID
    if payload.get("product") != _PRODUCT_ID:
        INVALID["message"] = "⛔ Key ini bukan untuk produk Data Pilot Pro."
        return INVALID

    tier       = payload.get("tier", "TRIAL").upper()
    owner      = payload.get("owner", "Unknown")
    expiry_str = payload.get("expiry", "")

    if expiry_str == "UNLIMITED":
        days_left, expired = "UNLIMITED", False
    else:
        try:
            expiry_dt = datetime.strptime(expiry_str, "%Y%m%d")
            days_left = (expiry_dt - datetime.utcnow()).days
            expired   = days_left < 0
        except ValueError:
            INVALID["message"] = "⛔ Format tanggal lisensi korup."
            return INVALID

    if expired:
        INVALID["message"] = (f"⛔ Lisensi kadaluarsa sejak {expiry_str}. "
                               "Hubungi pemilik platform untuk perpanjangan.")
        return INVALID

    tier_info = TIER_LIMITS.get(tier, TIER_LIMITS["TRIAL"])
    label     = tier_info["label"]
    days_txt  = f"{days_left} hari lagi" if days_left != "UNLIMITED" else "Tidak terbatas"
    return {
        "valid":     True,
        "tier":      tier,
        "owner":     owner,
        "expiry":    expiry_str,
        "days_left": days_left,
        "limits":    tier_info,
        "message":   f"✅ Lisensi {label} aktif — {owner} | {days_txt}",
    }

# ═══════════════════════════════════════════════════════════════
# ── SMART DATA CLEANER (EMBEDDED) ──
# ═══════════════════════════════════════════════════════════════

NULL_SYNONYMS = {
    'null','none','na','n/a','nan','-','--','---','kosong','empty',
    'unknown','tidak ada','tidak diketahui','#n/a','#null!','#value!',
    '#ref!','#div/0!','.','missing','tidak tersedia','n.a','n.a.','?','??',
}
DATE_FORMATS = [
    "%d/%m/%Y","%m/%d/%Y","%Y-%m-%d","%d-%m-%Y",
    "%d %B %Y","%B %d, %Y","%d %b %Y","%b %d %Y",
    "%Y%m%d","%d%m%Y","%d/%m/%y","%m/%d/%y",
]

def _is_null_syn(val) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)): return True
    return str(val).strip().lower() in NULL_SYNONYMS

def _clean_col_name(name: str) -> str:
    name = unicodedata.normalize("NFC", str(name)).strip()
    name = name.encode("ascii", errors="ignore").decode()
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return name or "col_unnamed"

def _parse_num(s) -> float | None:
    if s is None: return None
    s = str(s).strip()
    # Hapus karakter noise: backtick, kutip, spasi lebar, simbol mata uang
    s = re.sub(r'[`\'"´]', '', s)
    s = s.replace('\u00a0',' ').replace('\u200b','').replace('\ufeff','')
    s = re.sub(r'[Rr][Pp]\.?\s*','', s)
    s = re.sub(r'[$€£¥₹]','', s)
    s = re.sub(r'\s*(units?|pcs|kg|ton|liter?|ltr|m²|m2|km|%)\s*$','',s,flags=re.IGNORECASE)
    negative = False
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1]; negative = True
    s = s.replace('%','').strip()
    # Format Eropa vs US
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'): s = s.replace('.','').replace(',','.')
        else: s = s.replace(',','')
    elif ',' in s:
        after = s.split(',')[-1]
        s = s.replace(',','') if (len(after)==3 and after.isdigit()) else s.replace(',','.')
    else:
        if s.count('.') > 1: s = s.replace('.','')
    s = s.replace(' ','')
    s = re.sub(r'[^\d.\-]','', s).strip('.')
    if not s or s in ('','-'): return None
    try:
        v = float(s); return -v if negative else v
    except ValueError:
        return None

def _try_date(val):
    if pd.isnull(val): return None
    s = str(val).strip()
    for fmt in DATE_FORMATS:
        try: return datetime.strptime(s, fmt)
        except ValueError: continue
    try: return pd.to_datetime(s, infer_datetime_format=True, dayfirst=True)
    except: return None

def _col_type(series: pd.Series) -> str:
    sample = series.dropna().astype(str).head(150)
    if len(sample) == 0: return 'categorical'
    n = len(sample)
    num_hit  = sum(1 for v in sample if _parse_num(v) is not None)
    date_hit = sum(1 for v in sample if _try_date(v) is not None)
    if series.nunique() == len(series) and series.dtype == object: return 'id'
    if num_hit/n  >= 0.70: return 'numeric'
    if date_hit/n >= 0.70 and num_hit/n < 0.50: return 'date'
    if 0.30 <= num_hit/n < 0.70: return 'mixed'
    return 'categorical'

def _flag_iqr(series: pd.Series, mult=3.0) -> pd.Series:
    q1,q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - mult*iqr) | (series > q3 + mult*iqr)

def smart_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    report = {"baris_awal":len(df),"kolom_awal":len(df.columns),
              "langkah":[],"kolom_diubah":{},"outlier_flags":{}}
    log = report["langkah"].append
    df = df.copy()

    # 1. Bersihkan nama kolom
    old = list(df.columns)
    new = []
    seen = {}
    for n in [_clean_col_name(c) for c in old]:
        seen[n] = seen.get(n,0)
        new.append(f"{n}_{seen[n]}" if seen[n] else n)
        seen[n] += 1
    renames = {o:n for o,n in zip(old,new) if o!=n}
    df.columns = new
    log(f"Nama kolom diperbaiki: {renames}" if renames else "Nama kolom: tidak ada perubahan.")

    # 2. Hapus kolom/baris semua-null
    before_c = len(df.columns); df.dropna(axis=1,how='all',inplace=True)
    if (d:=before_c-len(df.columns)): log(f"Dihapus {d} kolom semua-null.")
    before_r = len(df); df.dropna(axis=0,how='all',inplace=True)
    if (d:=before_r-len(df)): log(f"Dihapus {d} baris semua-null.")

    # 3. Ganti null-synonym → NaN
    total_null = 0
    for col in df.select_dtypes(include='object').columns:
        mask = df[col].apply(_is_null_syn); c=mask.sum()
        if c: df.loc[mask,col]=np.nan; total_null+=c
    if total_null: log(f"Diganti {total_null:,} nilai null-synonym (na/null/-/kosong/dsb).")

    # 4. Strip whitespace & karakter invisible
    ws=0
    for col in df.select_dtypes(include='object').columns:
        before = df[col].copy()
        df[col] = (df[col].astype(str)
                   .str.replace('\u00a0',' ',regex=False)
                   .str.replace('\u200b','',regex=False)
                   .str.replace('\ufeff','',regex=False)
                   .str.replace('\u200c','',regex=False)
                   .str.strip()
                   .replace('nan',np.nan))
        ws += (before.fillna('')!=df[col].fillna('')).sum()
    if ws: log(f"Diperbaiki {ws:,} sel: whitespace & karakter invisible dihapus.")

    # 5. Deteksi tipe & konversi + flag outlier
    for col in list(df.columns):
        ct = _col_type(df[col])
        orig = str(df[col].dtype)
        if ct == 'numeric':
            conv = df[col].apply(lambda v: _parse_num(v) if not pd.isnull(v) else np.nan)
            ok = conv.notna().sum(); tot = df[col].notna().sum()
            df[col] = conv
            report["kolom_diubah"][col] = {"dari":orig,"ke":"float64","ok":int(ok),"tot":int(tot)}
            log(f"Kolom '{col}': {orig} → numeric ({ok}/{tot} berhasil — backtick/koma/titik/simbol dibersihkan).")
            valid = df[col].dropna()
            if len(valid)>=10:
                om = _flag_iqr(valid); n_out=om.sum()
                if n_out:
                    report["outlier_flags"][col]={"jumlah":int(n_out),"pct":f"{n_out/len(valid)*100:.1f}%"}
                    log(f"  ⚠ '{col}': {n_out} outlier (IQR×3) — data TIDAK dihapus, ditandai di kolom flag.")
                    df[f"{col}_outlier_flag"]=False
                    df.loc[valid[om].index,f"{col}_outlier_flag"]=True
        elif ct == 'date':
            conv = df[col].apply(_try_date)
            ok = conv.notna().sum()
            df[col] = pd.to_datetime(conv, errors='coerce')
            report["kolom_diubah"][col] = {"dari":orig,"ke":"datetime64","ok":int(ok)}
            log(f"Kolom '{col}': {orig} → datetime ({ok} berhasil).")
        elif ct == 'mixed':
            conv = df[col].apply(lambda v: _parse_num(v) if not pd.isnull(v) else np.nan)
            frac = conv.notna().sum()/max(df[col].notna().sum(),1)
            if frac>=0.6:
                df[col]=conv
                report["kolom_diubah"][col]={"dari":orig,"ke":"float64 (partial)"}
                log(f"Kolom '{col}' (mixed): {frac*100:.0f}% dikonversi ke numeric.")
            else:
                log(f"Kolom '{col}' (mixed): tetap string ({frac*100:.0f}% numerik — tidak cukup dominan).")
        elif ct == 'categorical':
            if df[col].nunique()<=50:
                df[col]=df[col].str.strip().str.title()
                log(f"Kolom '{col}' (kat.): casing distandarisasi.")

    # 6. Hapus duplikat exact
    before=len(df); df.drop_duplicates(inplace=True)
    d=before-len(df)
    log(f"Dihapus {d:,} baris duplikat exact." if d else "Tidak ada duplikat exact.")

    # 7. Fill NaN numerik dengan MEDIAN (lebih robust dari 0)
    for col in df.select_dtypes(include=[np.number]).columns:
        n=df[col].isna().sum()
        if n:
            med=df[col].median()
            df[col].fillna(med,inplace=True)
            log(f"Kolom '{col}': {n} NaN → diisi median ({med:.4f}).")

    # 8. Fill NaN kategorik
    cat_fill=0
    for col in df.select_dtypes(include='object').columns:
        n=df[col].isna().sum()
        if n: df[col].fillna('Unknown',inplace=True); cat_fill+=n
    if cat_fill: log(f"Diisi {cat_fill:,} NaN teks dengan 'Unknown'.")

    df.reset_index(drop=True,inplace=True)
    report["baris_akhir"]=len(df); report["kolom_akhir"]=len(df.columns)
    return df, report

def render_cleaning_report(report: dict) -> str:
    lines=[
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  LAPORAN SMART DATA CLEANING",
        f"  {datetime.now().strftime('%d %B %Y, %H:%M:%S')}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  Data Awal  : {report['baris_awal']:,} baris × {report['kolom_awal']} kolom",
        f"  Data Bersih: {report['baris_akhir']:,} baris × {report['kolom_akhir']} kolom",
        "","  LANGKAH PEMBERSIHAN:",
    ]
    for i,s in enumerate(report["langkah"],1): lines.append(f"  {i:2d}. {s}")
    if report.get("kolom_diubah"):
        lines+=["","  KONVERSI TIPE KOLOM:"]
        for col,info in report["kolom_diubah"].items():
            lines.append(f"     • {col}: {info['dari']} → {info['ke']}")
    if report.get("outlier_flags"):
        lines+=["","  ⚠ OUTLIER TERDETEKSI (diflag, tidak dihapus):"]
        for col,info in report["outlier_flags"].items():
            lines.append(f"     • {col}: {info['jumlah']} outlier ({info['pct']}) → lihat kolom *_outlier_flag")
    lines+=["","━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════
# ── KONFIGURASI HALAMAN (harus sebelum st.* lainnya) ──
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Data Pilot Pro",
    layout="wide",
    page_icon="🛸",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# ── SISTEM LISENSI — GATE UTAMA ──
# ═══════════════════════════════════════════════════════════════
#
# Urutan pencarian lisensi:
#   1. st.session_state['_lic_key']  (sudah diinput di session ini)
#   2. File .datapilot_license       (auto-login permanen)
#   3. Env var DATAPILOT_KEY
#   4. Tidak ada → tampilkan form aktivasi
#
def _logout_license():
    """Hapus lisensi dari session state dan file — kembali ke halaman aktivasi."""
    st.session_state.pop("lic_info", None)
    st.session_state.pop("_lic_key", None)
    # Hapus file lisensi jika ada
    try:
        if _LICENSE_FILE.exists():
            _LICENSE_FILE.unlink()
    except Exception:
        pass
    # Bersihkan semua session state terkait analisis
    for k in list(st.session_state.keys()):
        st.session_state.pop(k, None)
    st.rerun()

def _get_session_lic() -> dict:
    """Ambil status lisensi dari session state (persisten per-session)."""
    if "lic_info" in st.session_state and st.session_state["lic_info"].get("valid"):
        return st.session_state["lic_info"]
    # Coba auto-load dari file / env
    info = validate_license()
    if info["valid"]:
        st.session_state["lic_info"] = info
    return info

lic_info = _get_session_lic()

# ── Jika BELUM valid → tampilkan halaman aktivasi ──────────────
if not lic_info.get("valid"):
    st.markdown("""
    <style>
    .lic-wrap {max-width:520px; margin:80px auto 0 auto; text-align:center}
    .lic-title {font-size:2.2em; font-weight:800; color:#ff4b4b; margin-bottom:6px}
    .lic-sub   {color:#78909c; font-size:1.05em; margin-bottom:32px}
    .lic-hint  {font-size:0.82em; color:#546e7a; margin-top:10px; line-height:1.6}
    </style>
    <div class='lic-wrap'>
      <div class='lic-title'>🔒 Data Pilot Pro</div>
      <div class='lic-sub'>Masukkan license key untuk melanjutkan.</div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        raw_input = st.text_input(
            "License Key:",
            type="password",
            placeholder="DPP-XXXX...",
            help="Key diawali DPP-  •  Dapatkan key dari pemilik platform",
        )

        if st.button("🔑 Aktivasi Lisensi", use_container_width=True):
            if not raw_input.strip():
                st.warning("Masukkan license key terlebih dahulu.")
            elif not raw_input.strip().upper().startswith("DPP-"):
                st.error("❌ Format salah — key harus diawali dengan **DPP-**")
            else:
                test = validate_license(raw_input.strip())
                if test["valid"]:
                    # Simpan ke session state (langsung aktif tanpa reload)
                    st.session_state["lic_info"] = test
                    # Simpan ke file agar auto-login di session berikutnya
                    try:
                        _LICENSE_FILE.write_text(raw_input.strip(), encoding="utf-8")
                        st.success(f"✅ {test['message']}  —  Menyimpan lisensi…")
                    except Exception:
                        st.success(f"✅ {test['message']}  (lisensi aktif sesi ini)")
                    st.rerun()
                else:
                    st.error(f"❌ {test['message']}")

        st.markdown("""
        <div class='lic-hint'>
        💡 <b>Cara mendapatkan key:</b><br>
        Jalankan perintah berikut di terminal proyek Anda:<br>
        <code>python -c "from app import generate_license_key; print(generate_license_key('Nama Anda', 365, 'PRO'))"</code><br><br>
        Atau gunakan <code>license_engine.py</code> jika tersedia:<br>
        <code>python license_engine.py generate --owner "Nama Anda" --days 365 --tier PRO</code>
        </div>
        """, unsafe_allow_html=True)

    st.stop()   # Hentikan eksekusi — tidak ada yang bisa diakses tanpa lisensi

# ── Lisensi valid — lanjut ke app ──────────────────────────────
# Refresh lic_info dari session state (pastikan paling baru)
lic_info = st.session_state.get("lic_info", lic_info)

# ─────────────────────────────────────────────
# CSS CUSTOM
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main { background: #0a0e1a; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0d1520 100%);
}

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; font-weight: 700; }

.metric-card {
    background: linear-gradient(135deg, #1a2035 0%, #1e2d45 100%);
    border: 1px solid #2a3f5f;
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
    box-shadow: 0 4px 20px rgba(0,100,255,0.08);
}

.formula-badge {
    background: #0d2137;
    border: 1px solid #1e4060;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82em;
    color: #64b5f6;
    margin: 8px 0;
}

.insight-box {
    background: linear-gradient(135deg, #0d2137, #0a1929);
    border-left: 4px solid #00e5ff;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 12px 0;
}

.status-excellent { color: #00e676; font-weight: 700; }
.status-good { color: #ffeb3b; font-weight: 700; }
.status-warning { color: #ff9800; font-weight: 700; }
.status-critical { color: #f44336; font-weight: 700; }

.section-header {
    background: linear-gradient(90deg, #1a3a5c, transparent);
    border-left: 4px solid #00b4d8;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 20px 0 12px 0;
    font-size: 1.1em;
    font-weight: 600;
    color: #e0f7fa;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1520 0%, #111827 100%);
    border-right: 1px solid #1e3a5f;
}

.stButton > button {
    background: linear-gradient(135deg, #0066cc, #0099ff);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 20px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0077ee, #00aaff);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0,150,255,0.4);
}

.export-tag {
    display: inline-block;
    background: #1a3a5c;
    border: 1px solid #2a5f8c;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78em;
    color: #90caf9;
    margin: 2px;
}

/* NLQ — tombol contoh pertanyaan */
div[data-testid="stButton"] button[kind="secondary"]:has(+ *) {
    font-size: 0.82em !important;
    text-align: left !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 42px;
    line-height: 1.35;
    padding: 6px 10px !important;
    background: linear-gradient(135deg, #0d2137, #0a1929) !important;
    border: 1px solid #1e4060 !important;
    color: #90caf9 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    border-color: #00b4d8 !important;
    color: #00e5ff !important;
    background: linear-gradient(135deg, #0d2d45, #0a2035) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 180, 216, 0.2) !important;
}

/* War Room debate bubble */
.debate-turn-log   { border-left: 4px solid #00e5b4; background:#0a1f1a; }
.debate-turn-fin   { border-left: 4px solid #ffd54f; background:#1a1500; }
.debate-turn-cmd   { border-left: 4px solid #ff4500; background:#1a0d07; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FUNGSI UTILITAS
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    lf = pl.from_pandas(df).lazy()
    lf = lf.unique().fill_null(0)
    return lf.collect().to_pandas()

def force_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r'[^\d.\-]', '', regex=True)
    s = s.replace('', '0')
    return pd.to_numeric(s, errors='coerce').fillna(0)

def status_label(value, thresholds, labels):
    """thresholds: [crit, warn, good] ascending; labels: [critical, warning, good, excellent]"""
    if value <= thresholds[0]: return labels[0], "status-critical"
    elif value <= thresholds[1]: return labels[1], "status-warning"
    elif value <= thresholds[2]: return labels[2], "status-good"
    else: return labels[3], "status-excellent"

def pct(v): return f"{v:.2f}%"
def curr(v): return f"{v:,.2f}"
def ratio(v): return f"{v:.4f}"


# ═══════════════════════════════════════════════════════════════
# FORMULA ENGINE — SEMUA KATEGORI
# ═══════════════════════════════════════════════════════════════

class FinancialFormulas:
    @staticmethod
    def roi(revenue, cost):
        cost_safe = np.where(cost == 0, 1, cost)
        return ((revenue - cost) / cost_safe) * 100

    @staticmethod
    def gross_profit_margin(revenue, cogs):
        rev_safe = np.where(revenue == 0, 1, revenue)
        return ((revenue - cogs) / rev_safe) * 100

    @staticmethod
    def net_profit_margin(net_profit, revenue):
        rev_safe = np.where(revenue == 0, 1, revenue)
        return (net_profit / rev_safe) * 100

    @staticmethod
    def ebitda_margin(ebitda, revenue):
        rev_safe = np.where(revenue == 0, 1, revenue)
        return (ebitda / rev_safe) * 100

    @staticmethod
    def current_ratio(current_assets, current_liabilities):
        cl_safe = np.where(current_liabilities == 0, 1, current_liabilities)
        return current_assets / cl_safe

    @staticmethod
    def debt_to_equity(total_debt, total_equity):
        eq_safe = np.where(total_equity == 0, 1, total_equity)
        return total_debt / eq_safe

    @staticmethod
    def asset_turnover(revenue, total_assets):
        ta_safe = np.where(total_assets == 0, 1, total_assets)
        return revenue / ta_safe

    @staticmethod
    def roe(net_income, shareholders_equity):
        eq_safe = np.where(shareholders_equity == 0, 1, shareholders_equity)
        return (net_income / eq_safe) * 100

    @staticmethod
    def roa(net_income, total_assets):
        ta_safe = np.where(total_assets == 0, 1, total_assets)
        return (net_income / ta_safe) * 100

    @staticmethod
    def working_capital(current_assets, current_liabilities):
        return current_assets - current_liabilities

    @staticmethod
    def breakeven(fixed_cost, price_per_unit, variable_cost_per_unit):
        margin = price_per_unit - variable_cost_per_unit
        margin_safe = np.where(margin == 0, 0.001, margin)
        return fixed_cost / margin_safe

    @staticmethod
    def clv(avg_purchase_value, purchase_frequency, customer_lifespan):
        return avg_purchase_value * purchase_frequency * customer_lifespan

    @staticmethod
    def cac(total_marketing_cost, new_customers):
        nc_safe = np.where(new_customers == 0, 1, new_customers)
        return total_marketing_cost / nc_safe


class SalesMarketingFormulas:
    @staticmethod
    def conversion_rate(conversions, total_visitors):
        tv_safe = np.where(total_visitors == 0, 1, total_visitors)
        return (conversions / tv_safe) * 100

    @staticmethod
    def average_order_value(total_revenue, num_orders):
        no_safe = np.where(num_orders == 0, 1, num_orders)
        return total_revenue / no_safe

    @staticmethod
    def churn_rate(lost_customers, total_customers_start):
        tc_safe = np.where(total_customers_start == 0, 1, total_customers_start)
        return (lost_customers / tc_safe) * 100

    @staticmethod
    def retention_rate(customers_end, customers_start, new_customers):
        cs_safe = np.where(customers_start == 0, 1, customers_start)
        return ((customers_end - new_customers) / cs_safe) * 100

    @staticmethod
    def nps_score(promoters, detractors, total_respondents):
        tr_safe = np.where(total_respondents == 0, 1, total_respondents)
        return ((promoters - detractors) / tr_safe) * 100

    @staticmethod
    def roas(revenue, ad_spend):
        as_safe = np.where(ad_spend == 0, 1, ad_spend)
        return revenue / as_safe

    @staticmethod
    def ctr(clicks, impressions):
        imp_safe = np.where(impressions == 0, 1, impressions)
        return (clicks / imp_safe) * 100

    @staticmethod
    def market_share(company_sales, total_market_sales):
        tm_safe = np.where(total_market_sales == 0, 1, total_market_sales)
        return (company_sales / tm_safe) * 100


class SupplyChainFormulas:
    @staticmethod
    def inventory_turnover(cogs, avg_inventory):
        inv_safe = np.where(avg_inventory == 0, 1, avg_inventory)
        return cogs / inv_safe

    @staticmethod
    def days_inventory_outstanding(avg_inventory, cogs):
        cogs_safe = np.where(cogs == 0, 1, cogs)
        return (avg_inventory / cogs_safe) * 365

    @staticmethod
    def days_sales_outstanding(accounts_receivable, annual_revenue):
        rev_safe = np.where(annual_revenue == 0, 1, annual_revenue)
        return (accounts_receivable / rev_safe) * 365

    @staticmethod
    def days_payable_outstanding(accounts_payable, cogs):
        cogs_safe = np.where(cogs == 0, 1, cogs)
        return (accounts_payable / cogs_safe) * 365

    @staticmethod
    def cash_conversion_cycle(dio, dso, dpo):
        return dio + dso - dpo

    @staticmethod
    def fill_rate(orders_shipped_complete, total_orders):
        to_safe = np.where(total_orders == 0, 1, total_orders)
        return (orders_shipped_complete / to_safe) * 100

    @staticmethod
    def otif(on_time_in_full, total_orders):
        to_safe = np.where(total_orders == 0, 1, total_orders)
        return (on_time_in_full / to_safe) * 100

    @staticmethod
    def perfect_order_rate(perfect_orders, total_orders):
        to_safe = np.where(total_orders == 0, 1, total_orders)
        return (perfect_orders / to_safe) * 100

    @staticmethod
    def reorder_point(avg_daily_demand, lead_time_days, safety_stock):
        return (avg_daily_demand * lead_time_days) + safety_stock

    @staticmethod
    def safety_stock(z_score, std_demand, lead_time):
        return z_score * std_demand * np.sqrt(lead_time)

    @staticmethod
    def eoq(annual_demand, ordering_cost, holding_cost_per_unit):
        hc_safe = np.where(holding_cost_per_unit == 0, 0.001, holding_cost_per_unit)
        return np.sqrt((2 * annual_demand * ordering_cost) / hc_safe)

    @staticmethod
    def supply_chain_cost_ratio(total_sc_cost, total_revenue):
        rev_safe = np.where(total_revenue == 0, 1, total_revenue)
        return (total_sc_cost / rev_safe) * 100

    @staticmethod
    def supplier_defect_rate(defective_units, total_units_received):
        tu_safe = np.where(total_units_received == 0, 1, total_units_received)
        return (defective_units / tu_safe) * 100


class OperationalFormulas:
    @staticmethod
    def oee(availability, performance, quality):
        return (availability / 100) * (performance / 100) * (quality / 100) * 100

    @staticmethod
    def utilization_rate(actual_hours, available_hours):
        ah_safe = np.where(available_hours == 0, 1, available_hours)
        return (actual_hours / ah_safe) * 100

    @staticmethod
    def defect_rate(defective_units, total_units):
        tu_safe = np.where(total_units == 0, 1, total_units)
        return (defective_units / tu_safe) * 100

    @staticmethod
    def first_pass_yield(good_units, total_units):
        tu_safe = np.where(total_units == 0, 1, total_units)
        return (good_units / tu_safe) * 100

    @staticmethod
    def cycle_time(total_time, units_produced):
        up_safe = np.where(units_produced == 0, 1, units_produced)
        return total_time / up_safe

    @staticmethod
    def throughput(units_produced, time_period):
        tp_safe = np.where(time_period == 0, 1, time_period)
        return units_produced / tp_safe

    @staticmethod
    def labor_productivity(output_value, labor_hours):
        lh_safe = np.where(labor_hours == 0, 1, labor_hours)
        return output_value / lh_safe

    @staticmethod
    def capacity_utilization(actual_output, max_possible_output):
        mp_safe = np.where(max_possible_output == 0, 1, max_possible_output)
        return (actual_output / mp_safe) * 100

    @staticmethod
    def mttr(total_downtime_hours, num_failures):
        nf_safe = np.where(num_failures == 0, 1, num_failures)
        return total_downtime_hours / nf_safe

    @staticmethod
    def mtbf(total_operating_time, num_failures):
        nf_safe = np.where(num_failures == 0, 1, num_failures)
        return total_operating_time / nf_safe

    @staticmethod
    def cost_per_unit(total_cost, units_produced):
        up_safe = np.where(units_produced == 0, 1, units_produced)
        return total_cost / up_safe

    @staticmethod
    def employee_turnover(employees_left, avg_employees):
        ae_safe = np.where(avg_employees == 0, 1, avg_employees)
        return (employees_left / ae_safe) * 100


class HRAnalyticsFormulas:
    @staticmethod
    def revenue_per_employee(total_revenue, num_employees):
        ne_safe = np.where(num_employees == 0, 1, num_employees)
        return total_revenue / ne_safe

    @staticmethod
    def absenteeism_rate(days_absent, total_workdays):
        tw_safe = np.where(total_workdays == 0, 1, total_workdays)
        return (days_absent / tw_safe) * 100

    @staticmethod
    def training_roi(benefits_from_training, training_cost):
        tc_safe = np.where(training_cost == 0, 1, training_cost)
        return ((benefits_from_training - training_cost) / tc_safe) * 100

    @staticmethod
    def offer_acceptance_rate(accepted_offers, total_offers):
        to_safe = np.where(total_offers == 0, 1, total_offers)
        return (accepted_offers / to_safe) * 100


# ─────────────────────────────────────────────
# DEFINISI SEMUA FORMULA UNTUK UI
# ─────────────────────────────────────────────
FORMULA_CATALOG = {
    "💰 Financial Analysis": {
        "ROI (Return on Investment)": {
            "cols": ["Revenue", "Cost"],
            "formula_str": "ROI = ((Revenue - Cost) / Cost) × 100",
            "unit": "%",
            "thresholds": [0, 10, 20],
            "labels": ["Rugi / Negatif", "Rendah", "Stabil", "Sangat Baik"],
            "fn": lambda d, c: FinancialFormulas.roi(d[c[0]], d[c[1]])
        },
        "Gross Profit Margin": {
            "cols": ["Revenue", "COGS"],
            "formula_str": "GPM = ((Revenue - COGS) / Revenue) × 100",
            "unit": "%",
            "thresholds": [20, 40, 60],
            "labels": ["Kritis", "Rendah", "Normal", "Sangat Baik"],
            "fn": lambda d, c: FinancialFormulas.gross_profit_margin(d[c[0]], d[c[1]])
        },
        "Net Profit Margin": {
            "cols": ["Net_Profit", "Revenue"],
            "formula_str": "NPM = (Net Profit / Revenue) × 100",
            "unit": "%",
            "thresholds": [0, 5, 15],
            "labels": ["Rugi", "Tipis", "Normal", "Excellent"],
            "fn": lambda d, c: FinancialFormulas.net_profit_margin(d[c[0]], d[c[1]])
        },
        "EBITDA Margin": {
            "cols": ["EBITDA", "Revenue"],
            "formula_str": "EBITDA Margin = (EBITDA / Revenue) × 100",
            "unit": "%",
            "thresholds": [10, 20, 30],
            "labels": ["Kritis", "Rendah", "Sehat", "Excellent"],
            "fn": lambda d, c: FinancialFormulas.ebitda_margin(d[c[0]], d[c[1]])
        },
        "Current Ratio (Likuiditas)": {
            "cols": ["Current_Assets", "Current_Liabilities"],
            "formula_str": "Current Ratio = Current Assets / Current Liabilities",
            "unit": "x",
            "thresholds": [1, 1.5, 2],
            "labels": ["Berbahaya", "Perlu Perhatian", "Cukup", "Sangat Likuid"],
            "fn": lambda d, c: FinancialFormulas.current_ratio(d[c[0]], d[c[1]])
        },
        "Debt-to-Equity Ratio": {
            "cols": ["Total_Debt", "Total_Equity"],
            "formula_str": "D/E = Total Debt / Total Equity",
            "unit": "x",
            "thresholds": [3, 2, 1],  # Lower is better, reversed
            "labels": ["Sangat Aman", "Baik", "Perhatian", "Berbahaya"],
            "fn": lambda d, c: FinancialFormulas.debt_to_equity(d[c[0]], d[c[1]])
        },
        "ROE (Return on Equity)": {
            "cols": ["Net_Income", "Shareholders_Equity"],
            "formula_str": "ROE = (Net Income / Equity) × 100",
            "unit": "%",
            "thresholds": [0, 10, 20],
            "labels": ["Negatif", "Rendah", "Normal", "Excellent"],
            "fn": lambda d, c: FinancialFormulas.roe(d[c[0]], d[c[1]])
        },
        "ROA (Return on Assets)": {
            "cols": ["Net_Income", "Total_Assets"],
            "formula_str": "ROA = (Net Income / Total Assets) × 100",
            "unit": "%",
            "thresholds": [0, 5, 10],
            "labels": ["Negatif", "Rendah", "Normal", "Excellent"],
            "fn": lambda d, c: FinancialFormulas.roa(d[c[0]], d[c[1]])
        },
        "Customer Lifetime Value (CLV)": {
            "cols": ["Avg_Purchase_Value", "Purchase_Frequency", "Customer_Lifespan"],
            "formula_str": "CLV = Avg Purchase Value × Frequency × Lifespan",
            "unit": "",
            "thresholds": [100, 500, 1000],
            "labels": ["Sangat Rendah", "Rendah", "Normal", "Premium"],
            "fn": lambda d, c: FinancialFormulas.clv(d[c[0]], d[c[1]], d[c[2]])
        },
        "CAC (Customer Acquisition Cost)": {
            "cols": ["Marketing_Cost", "New_Customers"],
            "formula_str": "CAC = Total Marketing Cost / New Customers",
            "unit": "",
            "thresholds": [500, 200, 100],
            "labels": ["Sangat Efisien", "Efisien", "Normal", "Mahal"],
            "fn": lambda d, c: FinancialFormulas.cac(d[c[0]], d[c[1]])
        },
    },
    "📈 Sales & Marketing": {
        "Conversion Rate": {
            "cols": ["Conversions", "Total_Visitors"],
            "formula_str": "CVR = (Conversions / Visitors) × 100",
            "unit": "%",
            "thresholds": [1, 3, 5],
            "labels": ["Sangat Rendah", "Rendah", "Normal", "Excellent"],
            "fn": lambda d, c: SalesMarketingFormulas.conversion_rate(d[c[0]], d[c[1]])
        },
        "Average Order Value (AOV)": {
            "cols": ["Total_Revenue", "Num_Orders"],
            "formula_str": "AOV = Total Revenue / Number of Orders",
            "unit": "",
            "thresholds": [50, 100, 300],
            "labels": ["Sangat Rendah", "Rendah", "Normal", "Premium"],
            "fn": lambda d, c: SalesMarketingFormulas.average_order_value(d[c[0]], d[c[1]])
        },
        "Churn Rate": {
            "cols": ["Lost_Customers", "Total_Customers_Start"],
            "formula_str": "Churn = (Lost Customers / Total Start) × 100",
            "unit": "%",
            "thresholds": [15, 10, 5],
            "labels": ["Excellent", "Baik", "Perlu Perhatian", "Berbahaya"],
            "fn": lambda d, c: SalesMarketingFormulas.churn_rate(d[c[0]], d[c[1]])
        },
        "ROAS (Return on Ad Spend)": {
            "cols": ["Revenue", "Ad_Spend"],
            "formula_str": "ROAS = Revenue / Ad Spend",
            "unit": "x",
            "thresholds": [1, 2, 4],
            "labels": ["Rugi", "Break-even", "Profitabel", "Excellent"],
            "fn": lambda d, c: SalesMarketingFormulas.roas(d[c[0]], d[c[1]])
        },
        "Click-Through Rate (CTR)": {
            "cols": ["Clicks", "Impressions"],
            "formula_str": "CTR = (Clicks / Impressions) × 100",
            "unit": "%",
            "thresholds": [0.5, 2, 5],
            "labels": ["Sangat Rendah", "Rendah", "Normal", "Excellent"],
            "fn": lambda d, c: SalesMarketingFormulas.ctr(d[c[0]], d[c[1]])
        },
        "Market Share": {
            "cols": ["Company_Sales", "Total_Market_Sales"],
            "formula_str": "Market Share = (Company Sales / Total Market) × 100",
            "unit": "%",
            "thresholds": [5, 15, 30],
            "labels": ["Kecil", "Sedang", "Besar", "Dominan"],
            "fn": lambda d, c: SalesMarketingFormulas.market_share(d[c[0]], d[c[1]])
        },
        "NPS (Net Promoter Score)": {
            "cols": ["Promoters", "Detractors", "Total_Respondents"],
            "formula_str": "NPS = ((Promoters - Detractors) / Total) × 100",
            "unit": "pts",
            "thresholds": [0, 30, 70],
            "labels": ["Berbahaya", "Perlu Perbaikan", "Baik", "World Class"],
            "fn": lambda d, c: SalesMarketingFormulas.nps_score(d[c[0]], d[c[1]], d[c[2]])
        },
    },
    "🔗 Supply Chain": {
        "Inventory Turnover": {
            "cols": ["COGS", "Avg_Inventory"],
            "formula_str": "ITO = COGS / Average Inventory",
            "unit": "x/year",
            "thresholds": [2, 5, 10],
            "labels": ["Lambat / Mati", "Rendah", "Normal", "Agresif"],
            "fn": lambda d, c: SupplyChainFormulas.inventory_turnover(d[c[0]], d[c[1]])
        },
        "Days Inventory Outstanding (DIO)": {
            "cols": ["Avg_Inventory", "COGS"],
            "formula_str": "DIO = (Avg Inventory / COGS) × 365",
            "unit": "hari",
            "thresholds": [60, 45, 30],
            "labels": ["Excellent", "Baik", "Normal", "Lambat"],
            "fn": lambda d, c: SupplyChainFormulas.days_inventory_outstanding(d[c[0]], d[c[1]])
        },
        "Days Sales Outstanding (DSO)": {
            "cols": ["Accounts_Receivable", "Annual_Revenue"],
            "formula_str": "DSO = (AR / Annual Revenue) × 365",
            "unit": "hari",
            "thresholds": [30, 45, 60],
            "labels": ["Excellent", "Baik", "Normal", "Lambat"],
            "fn": lambda d, c: SupplyChainFormulas.days_sales_outstanding(d[c[0]], d[c[1]])
        },
        "Cash Conversion Cycle (CCC)": {
            "cols": ["DIO", "DSO", "DPO"],
            "formula_str": "CCC = DIO + DSO - DPO",
            "unit": "hari",
            "thresholds": [0, 30, 60],
            "labels": ["Negatif (Excellent)", "Excellent", "Normal", "Lambat"],
            "fn": lambda d, c: SupplyChainFormulas.cash_conversion_cycle(d[c[0]], d[c[1]], d[c[2]])
        },
        "Fill Rate": {
            "cols": ["Orders_Shipped_Complete", "Total_Orders"],
            "formula_str": "Fill Rate = (Complete Orders / Total Orders) × 100",
            "unit": "%",
            "thresholds": [85, 90, 95],
            "labels": ["Kritis", "Rendah", "Normal", "World Class"],
            "fn": lambda d, c: SupplyChainFormulas.fill_rate(d[c[0]], d[c[1]])
        },
        "OTIF (On-Time In-Full)": {
            "cols": ["On_Time_In_Full", "Total_Orders"],
            "formula_str": "OTIF = (OTIF Deliveries / Total Orders) × 100",
            "unit": "%",
            "thresholds": [80, 90, 95],
            "labels": ["Kritis", "Rendah", "Baik", "Excellent"],
            "fn": lambda d, c: SupplyChainFormulas.otif(d[c[0]], d[c[1]])
        },
        "Perfect Order Rate": {
            "cols": ["Perfect_Orders", "Total_Orders"],
            "formula_str": "POR = (Perfect Orders / Total Orders) × 100",
            "unit": "%",
            "thresholds": [80, 90, 95],
            "labels": ["Kritis", "Perlu Perbaikan", "Baik", "World Class"],
            "fn": lambda d, c: SupplyChainFormulas.perfect_order_rate(d[c[0]], d[c[1]])
        },
        "Supply Chain Cost Ratio": {
            "cols": ["Total_SC_Cost", "Total_Revenue"],
            "formula_str": "SC Cost % = (SC Cost / Revenue) × 100",
            "unit": "%",
            "thresholds": [20, 15, 10],
            "labels": ["Sangat Efisien", "Efisien", "Normal", "Mahal"],
            "fn": lambda d, c: SupplyChainFormulas.supply_chain_cost_ratio(d[c[0]], d[c[1]])
        },
        "Supplier Defect Rate": {
            "cols": ["Defective_Units", "Total_Units_Received"],
            "formula_str": "Defect Rate = (Defective / Total Received) × 100",
            "unit": "%",
            "thresholds": [5, 3, 1],
            "labels": ["Excellent", "Baik", "Perhatian", "Kritis"],
            "fn": lambda d, c: SupplyChainFormulas.supplier_defect_rate(d[c[0]], d[c[1]])
        },
        "EOQ (Economic Order Quantity)": {
            "cols": ["Annual_Demand", "Ordering_Cost", "Holding_Cost_Per_Unit"],
            "formula_str": "EOQ = √(2 × D × S / H)",
            "unit": "unit",
            "thresholds": [0, 100, 500],
            "labels": ["N/A", "Kecil", "Sedang", "Besar"],
            "fn": lambda d, c: SupplyChainFormulas.eoq(d[c[0]], d[c[1]], d[c[2]])
        },
    },
    "⚙️ Operational Excellence": {
        "OEE (Overall Equipment Effectiveness)": {
            "cols": ["Availability_%", "Performance_%", "Quality_%"],
            "formula_str": "OEE = Availability × Performance × Quality / 10000",
            "unit": "%",
            "thresholds": [50, 65, 85],
            "labels": ["Kritis", "Perlu Perbaikan", "Normal", "World Class"],
            "fn": lambda d, c: OperationalFormulas.oee(d[c[0]], d[c[1]], d[c[2]])
        },
        "Utilization Rate": {
            "cols": ["Actual_Hours", "Available_Hours"],
            "formula_str": "Utilization = (Actual Hours / Available Hours) × 100",
            "unit": "%",
            "thresholds": [50, 70, 85],
            "labels": ["Sangat Rendah", "Rendah", "Normal", "Optimal"],
            "fn": lambda d, c: OperationalFormulas.utilization_rate(d[c[0]], d[c[1]])
        },
        "Defect Rate": {
            "cols": ["Defective_Units", "Total_Units"],
            "formula_str": "Defect Rate = (Defects / Total) × 100",
            "unit": "%",
            "thresholds": [5, 3, 1],
            "labels": ["Excellent", "Baik", "Perhatian", "Kritis"],
            "fn": lambda d, c: OperationalFormulas.defect_rate(d[c[0]], d[c[1]])
        },
        "First Pass Yield (FPY)": {
            "cols": ["Good_Units", "Total_Units"],
            "formula_str": "FPY = (Good Units / Total Units) × 100",
            "unit": "%",
            "thresholds": [80, 90, 95],
            "labels": ["Kritis", "Rendah", "Normal", "Excellent"],
            "fn": lambda d, c: OperationalFormulas.first_pass_yield(d[c[0]], d[c[1]])
        },
        "Labor Productivity": {
            "cols": ["Output_Value", "Labor_Hours"],
            "formula_str": "Productivity = Output Value / Labor Hours",
            "unit": "",
            "thresholds": [10, 50, 100],
            "labels": ["Sangat Rendah", "Rendah", "Normal", "Tinggi"],
            "fn": lambda d, c: OperationalFormulas.labor_productivity(d[c[0]], d[c[1]])
        },
        "Capacity Utilization": {
            "cols": ["Actual_Output", "Max_Possible_Output"],
            "formula_str": "Capacity Util = (Actual / Max) × 100",
            "unit": "%",
            "thresholds": [50, 70, 85],
            "labels": ["Under-utilized", "Rendah", "Normal", "Optimal"],
            "fn": lambda d, c: OperationalFormulas.capacity_utilization(d[c[0]], d[c[1]])
        },
        "MTTR (Mean Time to Repair)": {
            "cols": ["Total_Downtime_Hours", "Num_Failures"],
            "formula_str": "MTTR = Total Downtime / Number of Failures",
            "unit": "jam",
            "thresholds": [8, 4, 2],
            "labels": ["Excellent", "Baik", "Perhatian", "Kritis"],
            "fn": lambda d, c: OperationalFormulas.mttr(d[c[0]], d[c[1]])
        },
        "MTBF (Mean Time Between Failures)": {
            "cols": ["Total_Operating_Time", "Num_Failures"],
            "formula_str": "MTBF = Total Operating Time / Failures",
            "unit": "jam",
            "thresholds": [100, 500, 1000],
            "labels": ["Kritis", "Rendah", "Normal", "Excellent"],
            "fn": lambda d, c: OperationalFormulas.mtbf(d[c[0]], d[c[1]])
        },
        "Cost per Unit": {
            "cols": ["Total_Cost", "Units_Produced"],
            "formula_str": "CPU = Total Cost / Units Produced",
            "unit": "",
            "thresholds": [500, 200, 100],
            "labels": ["Sangat Efisien", "Efisien", "Normal", "Mahal"],
            "fn": lambda d, c: OperationalFormulas.cost_per_unit(d[c[0]], d[c[1]])
        },
    },
    "👥 HR Analytics": {
        "Employee Turnover Rate": {
            "cols": ["Employees_Left", "Avg_Employees"],
            "formula_str": "Turnover = (Employees Left / Avg Employees) × 100",
            "unit": "%",
            "thresholds": [20, 15, 10],
            "labels": ["Excellent", "Baik", "Normal", "Tinggi / Masalah"],
            "fn": lambda d, c: OperationalFormulas.employee_turnover(d[c[0]], d[c[1]])
        },
        "Revenue per Employee": {
            "cols": ["Total_Revenue", "Num_Employees"],
            "formula_str": "Rev/Employee = Total Revenue / Employees",
            "unit": "",
            "thresholds": [50000, 100000, 200000],
            "labels": ["Sangat Rendah", "Rendah", "Normal", "Produktif"],
            "fn": lambda d, c: HRAnalyticsFormulas.revenue_per_employee(d[c[0]], d[c[1]])
        },
        "Absenteeism Rate": {
            "cols": ["Days_Absent", "Total_Workdays"],
            "formula_str": "Absenteeism = (Days Absent / Total Workdays) × 100",
            "unit": "%",
            "thresholds": [10, 5, 2],
            "labels": ["Excellent", "Baik", "Normal", "Tinggi"],
            "fn": lambda d, c: HRAnalyticsFormulas.absenteeism_rate(d[c[0]], d[c[1]])
        },
        "Training ROI": {
            "cols": ["Training_Benefits", "Training_Cost"],
            "formula_str": "Training ROI = ((Benefits - Cost) / Cost) × 100",
            "unit": "%",
            "thresholds": [0, 50, 100],
            "labels": ["Rugi", "Break-even", "Baik", "Excellent"],
            "fn": lambda d, c: HRAnalyticsFormulas.training_roi(d[c[0]], d[c[1]])
        },
        "Offer Acceptance Rate": {
            "cols": ["Accepted_Offers", "Total_Offers"],
            "formula_str": "OAR = (Accepted / Total Offers) × 100",
            "unit": "%",
            "thresholds": [60, 75, 90],
            "labels": ["Kritis", "Rendah", "Normal", "Excellent"],
            "fn": lambda d, c: HRAnalyticsFormulas.offer_acceptance_rate(d[c[0]], d[c[1]])
        },
    },
}


# ─────────────────────────────────────────────
# PDF GENERATOR
# ─────────────────────────────────────────────
def generate_professional_pdf(report_data: dict, filename: str):
    doc = SimpleDocTemplate(filename, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                  fontSize=22, textColor=colors.HexColor('#0d47a1'),
                                  spaceAfter=6)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
                               fontSize=14, textColor=colors.HexColor('#1565c0'),
                               spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=10, leading=16)
    formula_style = ParagraphStyle('Formula', parent=styles['Normal'],
                                    fontSize=9, textColor=colors.HexColor('#1a237e'),
                                    backColor=colors.HexColor('#e8eaf6'),
                                    borderPadding=6, leading=14)
    meta_style = ParagraphStyle('Meta', parent=styles['Normal'],
                                 fontSize=8, textColor=colors.gray)

    # Header
    story.append(Paragraph("📊 LAPORAN ANALITIK EKSEKUTIF", title_style))
    story.append(Paragraph(f"Data Pilot Pro | Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", meta_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1565c0')))
    story.append(Spacer(1, 12))

    # Formula Info
    story.append(Paragraph(f"Kategori: {report_data['category']}", h2_style))
    story.append(Paragraph(f"Formula: {report_data['formula_name']}", body_style))
    story.append(Paragraph(f"Rumus: {report_data['formula_str']}", formula_style))
    story.append(Spacer(1, 10))

    # Summary Statistics Table
    story.append(Paragraph("Ringkasan Statistik", h2_style))
    table_data = [["Metrik", "Nilai"]]
    for k, v in report_data['summary'].items():
        table_data.append([k, str(v)])

    tbl = Table(table_data, colWidths=[10*cm, 7*cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565c0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdbdbd')),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 14))

    # Insight
    story.append(Paragraph("💡 Insight & Rekomendasi Strategis", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#90caf9')))
    story.append(Spacer(1, 6))
    for para in report_data['insight'].split('\n\n'):
        story.append(Paragraph(para.strip(), body_style))
        story.append(Spacer(1, 6))

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.gray))
    story.append(Paragraph("Laporan ini dihasilkan secara otomatis oleh Data Pilot Pro. Untuk keputusan kritis, silakan konsultasikan dengan tim analis.", meta_style))

    doc.build(story)


# ─────────────────────────────────────────────
# POWER BI / TABLEAU EXPORT
# ─────────────────────────────────────────────
def prepare_bi_export(df: pd.DataFrame, formula_name: str, result_col: str) -> dict:
    """Siapkan 3 format output untuk BI tools"""
    # CSV utama
    csv_main = df.to_csv(index=False)

    # Summary per kolom numerik
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary_rows = []
    for col in numeric_cols:
        summary_rows.append({
            "Column": col,
            "Mean": round(df[col].mean(), 4),
            "Median": round(df[col].median(), 4),
            "Std": round(df[col].std(), 4),
            "Min": round(df[col].min(), 4),
            "Max": round(df[col].max(), 4),
            "Sum": round(df[col].sum(), 4),
            "Count": len(df[col])
        })
    summary_df = pd.DataFrame(summary_rows)
    csv_summary = summary_df.to_csv(index=False)

    # Metadata JSON untuk Tableau
    meta = {
        "generated_at": datetime.now().isoformat(),
        "formula": formula_name,
        "result_column": result_col,
        "total_rows": len(df),
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

    return {
        "main_csv": csv_main,
        "summary_csv": csv_summary,
        "meta_json": json.dumps(meta, indent=2)
    }


# ─────────────────────────────────────────────
# VISUALISASI OTOMATIS
# ─────────────────────────────────────────────
def auto_visualize(df: pd.DataFrame, result_col: str, formula_name: str):
    """Auto-generate visualisasi relevan"""
    figs = []

    # 1. Distribution histogram
    fig1 = px.histogram(df, x=result_col, nbins=30,
                        title=f"Distribusi {formula_name}",
                        color_discrete_sequence=['#00b4d8'],
                        template='plotly_dark')
    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,30,50,0.5)')
    figs.append(("📊 Distribusi", fig1))

    # 2. Box plot
    fig2 = px.box(df, y=result_col,
                  title=f"Box Plot: {formula_name}",
                  color_discrete_sequence=['#ff6b35'],
                  template='plotly_dark')
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,30,50,0.5)')
    figs.append(("📦 Box Plot", fig2))

    # 3. Trend jika ada kolom tanggal atau urutan
    date_cols = [c for c in df.columns if any(x in c.lower() for x in ['date', 'tanggal', 'bulan', 'month', 'year', 'tahun', 'period'])]
    if date_cols:
        try:
            df_sorted = df.sort_values(date_cols[0])
            fig3 = px.line(df_sorted, x=date_cols[0], y=result_col,
                           title=f"Tren {formula_name} vs Waktu",
                           color_discrete_sequence=['#00e5ff'],
                           template='plotly_dark')
            fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,30,50,0.5)')
            figs.append(("📈 Tren Waktu", fig3))
        except:
            pass

    # 4. Scatter dengan trendline manual (tanpa statsmodels)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != result_col]
    if numeric_cols:
        x_data = df[numeric_cols[0]].values.astype(float)
        y_data = df[result_col].values.astype(float)
        # Manual linear regression pakai numpy
        mask = np.isfinite(x_data) & np.isfinite(y_data)
        if mask.sum() > 2:
            coeffs = np.polyfit(x_data[mask], y_data[mask], 1)
            x_sorted = np.linspace(x_data[mask].min(), x_data[mask].max(), 100)
            y_trend = np.polyval(coeffs, x_sorted)
            corr = np.corrcoef(x_data[mask], y_data[mask])[0, 1]

            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=x_data, y=y_data, mode='markers',
                marker=dict(color='#69d2e7', size=6, opacity=0.7),
                name='Data'
            ))
            fig4.add_trace(go.Scatter(
                x=x_sorted, y=y_trend, mode='lines',
                line=dict(color='#ff6b35', width=2, dash='dash'),
                name=f'Tren Linear (r={corr:.2f})'
            ))
            fig4.update_layout(
                title=f"Korelasi: {numeric_cols[0]} vs {result_col} | r = {corr:.3f}",
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,30,50,0.5)',
                xaxis_title=numeric_cols[0],
                yaxis_title=result_col
            )
            figs.append(("🔗 Korelasi", fig4))

    return figs


# ═══════════════════════════════════════════════════════════════
# ── THE COMMANDER'S CHAT — AI ENGINE (GEMINI) ──
# ═══════════════════════════════════════════════════════════════

def _get_gemini_models(api_key: str) -> dict:
    """
    Auto-detect model Gemini yang tersedia untuk API key ini.
    Return dict: {'text': 'nama-model', 'vision': 'nama-model'}
    Hasil di-cache di session_state agar tidak query berulang.
    """
    cache_key = f"_gemini_models_{api_key[:8]}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    genai.configure(api_key=api_key.strip())

    # Prioritas model text (dari paling diinginkan ke fallback)
    TEXT_PREFS  = [
        "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro",
        "gemini-1.0-pro", "gemini-pro",
    ]
    # Prioritas model vision/multimodal
    VISION_PREFS = [
        "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro",
        "gemini-pro-vision", "gemini-1.0-pro-vision",
    ]

    try:
        # List semua model yang support generateContent
        all_models = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in (m.supported_generation_methods or [])
        ]
    except Exception:
        all_models = []

    def _pick(prefs, candidates):
        # Pilih model pertama dari preferensi yang ada di list tersedia
        for pref in prefs:
            for c in candidates:
                if pref in c:
                    return c
        # Jika list_models gagal / kosong, fallback ke hardcode
        return prefs[0]

    result = {
        "text":   _pick(TEXT_PREFS,   all_models) if all_models else "gemini-1.5-flash",
        "vision": _pick(VISION_PREFS, all_models) if all_models else "gemini-1.5-flash",
        "all":    all_models,
    }
    st.session_state[cache_key] = result
    return result


def _call_gemini(prompt: str, system: str = "", api_key: str = "") -> str:
    """Panggil Gemini text — pakai model dari sidebar, langsung lapor error tanpa freeze."""
    if not _GEMINI_OK:
        return "❌ `google-generativeai` belum terinstall. Jalankan: pip install google-generativeai"
    if not api_key or not api_key.strip():
        return "❌ Gemini API Key belum diset di sidebar."
    try:
        genai.configure(api_key=api_key.strip())
        # Pakai model yang dipilih user di sidebar — tidak auto-detect
        model_name = st.session_state.get('gemini_model', 'gemini-2.0-flash')
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system if system else None,
        )
        return model.generate_content(prompt).text
    except Exception as e:
        err = str(e)
        if "API_KEY_INVALID" in err or "API key not valid" in err:
            return "❌ API Key tidak valid. Cek kembali di Google AI Studio."
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            return (
                "⚠️ **API Limit tercapai.**\n\n"
                "Kemungkinan penyebab:\n"
                "- Rate limit per menit (tunggu ~1 menit)\n"
                "- Kuota harian habis (reset jam 07:00 WIB)\n\n"
                "Coba lagi sebentar, atau ganti API Key di sidebar."
            )
        return f"❌ Error Gemini ({model_name}): {err[:400]}"


def _call_gemini_chat(history: list, new_message: str, system: str = "", api_key: str = "") -> str:
    """Panggil Gemini multi-turn chat — pakai model dari sidebar."""
    if not _GEMINI_OK:
        return "❌ `google-generativeai` belum terinstall."
    if not api_key or not api_key.strip():
        return "❌ Gemini API Key belum diset di sidebar."
    try:
        genai.configure(api_key=api_key.strip())
        model_name = st.session_state.get('gemini_model', 'gemini-2.0-flash')
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system if system else None,
        )
        gemini_history = []
        for msg in history[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})
        chat = model.start_chat(history=gemini_history)
        return chat.send_message(new_message).text
    except Exception as e:
        err = str(e)
        if "API_KEY_INVALID" in err or "API key not valid" in err:
            return "❌ API Key tidak valid."
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            return "⚠️ API Limit tercapai. Tunggu ~1 menit lalu coba lagi, atau ganti API Key di sidebar."
        return f"❌ Error Gemini: {err[:400]}"


def _build_data_context(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Buat ringkasan data untuk dikirim ke AI sebagai konteks."""
    if df is None or len(df) == 0:
        return "Tidak ada data yang tersedia."
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    lines = [
        f"Dataset: {len(df):,} baris x {len(df.columns)} kolom",
        f"Kolom numerik: {', '.join(numeric_cols[:10])}",
        f"Kolom kategorikal: {', '.join(cat_cols[:5])}",
        "",
        "Statistik ringkasan:",
    ]
    for col in numeric_cols[:8]:
        s = df[col]
        lines.append(f"  {col}: mean={s.mean():.2f}, min={s.min():.2f}, max={s.max():.2f}, std={s.std():.2f}")
    if '__Result__' in df.columns:
        r = df['__Result__']
        lines.append(f"\nHasil Formula: mean={r.mean():.4f}, min={r.min():.4f}, max={r.max():.4f}")
    lines.append(f"\nContoh data (5 baris pertama):\n{df.head(5).to_string()}")
    return "\n".join(lines)


_COMMANDER_SYSTEM = """Kamu adalah DATA PILOT COMMANDER — analis data senior elit yang berbicara dengan otoritas, presisi, dan kejelasan komando layaknya konsultan McKinsey bertemu perwira staf ahli militer.

GAYA KOMUNIKASI:
- Gunakan bahasa profesional, tegas, dan impresif seperti laporan intelijen atau briefing staf ahli
- Struktur respons selalu: [ASESMEN SITUASI] -> [TEMUAN KUNCI] -> [REKOMENDASI TAKTIS]
- Padat, tajam, tidak ada basa-basi — langsung ke inti permasalahan
- Sebutkan angka spesifik dari data jika relevan
- Gunakan terminologi bisnis & strategis yang kuat (leverage, bottleneck, pivot, critical path, dll)
- Jika ditanya tentang grafik/visualisasi, rekomendasikan jenis chart yang paling tepat beserta alasannya
- Tandai temuan kritis dengan KRITIS dan peluang emas dengan PELUANG

KONTEKS: Kamu menganalisis data bisnis/operasional untuk membantu eksekutif membuat keputusan strategis berbasis data.
Jawab dalam Bahasa Indonesia kecuali diminta lain."""


def render_commander_chat(df: pd.DataFrame):
    """Render tab Commander's Chat dengan Gemini AI."""
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0a1929,#0d2137); border:1px solid #00e5ff44;
                border-radius:12px; padding:16px 20px; margin-bottom:16px'>
    <h3 style='color:#00e5ff; margin:0; font-size:1.3em'>&#9876;&#65039; THE COMMANDER'S CHAT</h3>
    <p style='color:#90caf9; margin:6px 0 0 0; font-size:0.9em'>
    Tanya langsung pada data Anda. <b>Gemini AI</b> menganalisis kolom, statistik, dan tren secara real-time.<br>
    <span style='color:#4fc3f7'>Powered by Google Gemini 1.5 Flash &mdash; Gratis &middot; Cepat &middot; Akurat</span>
    </p>
    </div>
    """, unsafe_allow_html=True)

    api_key = st.session_state.get('gemini_api_key', '')
    if not api_key:
        st.warning("Masukkan **Gemini API Key** di sidebar terlebih dahulu untuk mengaktifkan Commander's Chat.")
        st.markdown("""
        <div style='background:#0d1f30; border:1px solid #1e3a5f; border-radius:8px; padding:12px 16px; font-size:0.88em; color:#90caf9'>
        <b>Cara mendapatkan API Key GRATIS:</b><br>
        1. Buka <a href='https://aistudio.google.com/app/apikey' target='_blank' style='color:#00e5ff'>aistudio.google.com/app/apikey</a><br>
        2. Login dengan akun Google<br>
        3. Klik "Create API Key" &rarr; Salin key<br>
        4. Paste di kolom <b>Gemini API Key</b> di sidebar kiri
        </div>
        """, unsafe_allow_html=True)
        return

    if "commander_history" not in st.session_state:
        st.session_state.commander_history = []

    for msg in st.session_state.commander_history:
        role_icon = "Anda" if msg["role"] == "user" else "Commander AI (Gemini)"
        bg = "#1a2035" if msg["role"] == "user" else "#071520"
        border = "#2a3f5f" if msg["role"] == "user" else "#00e5ff55"
        st.markdown(f"""
        <div style='background:{bg}; border:1px solid {border}; border-radius:12px;
                    padding:12px 16px; margin:8px 0; font-size:0.92em'>
        <b style='color:#90caf9'>{role_icon}</b><br>
        <span style='color:#e0e0e0; white-space:pre-wrap; display:block'>{msg["content"]}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**Pertanyaan cepat — klik untuk tanya:**")
    quick_questions = [
        "Apa penyebab utama performa rendah berdasarkan data ini?",
        "Kolom mana yang paling berpengaruh terhadap hasil formula?",
        "Grafik apa yang paling tepat untuk presentasi ke CEO?",
        "Berikan ringkasan eksekutif 3 kalimat dari data ini",
        "Apa risiko terbesar yang terlihat dari analisis ini?",
        "Rekomendasi tindakan prioritas dalam 30 hari ke depan?",
    ]
    qc1, qc2, qc3 = st.columns(3)
    for i, (col, q) in enumerate(zip([qc1,qc2,qc3,qc1,qc2,qc3], quick_questions)):
        with col:
            label = f"{q[:38]}..." if len(q) > 38 else q
            if st.button(label, key=f"qq_{i}", use_container_width=True):
                st.session_state.commander_pending = q

    user_q = st.chat_input("Tanyakan sesuatu tentang data Anda... (contoh: Kenapa ROI turun bulan ini?)")
    if "commander_pending" in st.session_state:
        user_q = st.session_state.pop("commander_pending")

    if user_q:
        st.session_state.commander_history.append({"role": "user", "content": user_q})
        data_ctx = _build_data_context(df)
        formula_ctx = ""
        if 'formula_name' in st.session_state:
            formula_ctx = (
                f"\nFormula aktif: {st.session_state['formula_name']}"
                f"\nNilai rata-rata: {st.session_state.get('avg_val', 'N/A')} "
                f"{st.session_state.get('unit', '')}"
            )
        full_prompt = f"""KONTEKS DATA YANG SEDANG DIANALISIS:
{data_ctx}
{formula_ctx}

PERTANYAAN OPERATOR: {user_q}

Berikan analisis tajam, terstruktur, dan actionable. Tandai temuan kritis dengan [KRITIS] dan peluang besar dengan [PELUANG]."""

        with st.spinner("Commander Gemini menganalisis data..."):
            ai_resp = _call_gemini_chat(
                st.session_state.commander_history, full_prompt, _COMMANDER_SYSTEM, api_key
            )
        st.session_state.commander_history.append({"role": "assistant", "content": ai_resp})
        st.rerun()

    if st.session_state.get("commander_history"):
        if st.button("Bersihkan Riwayat Chat"):
            st.session_state.commander_history = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# AI SMART NARRATIVE — EXECUTIVE SUMMARY OTOMATIS
# ═══════════════════════════════════════════════════════════════

def render_ai_narrative(df: pd.DataFrame, formula_name: str, avg_val: float, unit: str, status: str):
    """Render tab Smart Narrative dengan Gemini AI."""
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0f1f10,#0d2515); border:1px solid #00e67644;
                border-radius:12px; padding:16px 20px; margin-bottom:16px'>
    <h3 style='color:#00e676; margin:0; font-size:1.3em'>&#128220; SMART NARRATIVE &amp; EXECUTIVE SUMMARY</h3>
    <p style='color:#a5d6a7; margin:6px 0 0 0; font-size:0.9em'>
    Gemini AI menulis laporan eksekutif profesional secara otomatis seperti konsultan senior McKinsey.<br>
    Siap untuk Board Meeting, CEO Briefing, atau laporan stakeholder.
    </p>
    </div>
    """, unsafe_allow_html=True)

    api_key = st.session_state.get('gemini_api_key', '')
    if not api_key:
        st.warning("Masukkan **Gemini API Key** di sidebar untuk mengaktifkan fitur ini.")
        return

    col_tone, col_len, col_lang = st.columns(3)
    with col_tone:
        tone = st.selectbox("Nada Laporan", [
            "Formal Militer (Perwira Staf)",
            "Eksekutif Bisnis (C-Level)",
            "Teknis Detail (Analis Senior)",
            "Board Level (Direksi & Komisaris)"
        ])
    with col_len:
        length = st.selectbox("Panjang", [
            "Ringkas - 1 halaman (300 kata)",
            "Standar - 2 halaman (600 kata)",
            "Komprehensif - 3+ halaman (1000+ kata)"
        ])
    with col_lang:
        lang = st.selectbox("Bahasa Output", ["Indonesia", "English", "Bilingual (ID + EN)"])

    include_sections = st.multiselect(
        "Seksi yang dimasukkan:",
        ["Asesmen Situasi", "Temuan Kunci & Angka", "Benchmark Industri",
         "Risiko & Peluang", "Rencana Aksi Prioritas", "Pesan untuk Manajemen"],
        default=["Asesmen Situasi", "Temuan Kunci & Angka", "Risiko & Peluang", "Rencana Aksi Prioritas"]
    )

    if st.button("Generate AI Narrative Sekarang", use_container_width=True, type="primary"):
        data_ctx = _build_data_context(df)
        brief_ctx = st.session_state.get('auto_insight', '')[:1500]
        sections_str = "\n".join([f"- {s}" for s in include_sections])
        narrative_system = (
            f"Kamu adalah konsultan strategis senior kelas dunia dengan keahlian data analytics. "
            f"Tulis dengan nada: {tone}. Panjang: {length}. Bahasa: {lang}. "
            f"Gunakan terminologi profesional yang kuat. Struktur laporan harus impresif dan siap cetak."
        )
        narrative_prompt = f"""Tulis Executive Summary Report berdasarkan data analitik berikut:

FORMULA: {formula_name}
STATUS: {status}
NILAI RATA-RATA: {avg_val:.4f} {unit}

DATA KONTEKS:
{data_ctx}

ANALYTICAL BRIEF (referensi):
{brief_ctx}

SEKSI YANG HARUS ADA:
{sections_str}

INSTRUKSI: Buka dengan asesmen powerful, gunakan angka spesifik, identifikasi risiko dan peluang nyata,
rencana aksi konkret dengan PIC dan deadline, tutup dengan pesan kunci yang mendesak tindakan.
Tulis seolah laporan ini akan dibacakan di hadapan Direktur Utama dalam rapat strategis besok pagi."""

        with st.spinner("Gemini sedang menulis laporan eksekutif..."):
            ai_narrative = _call_gemini(narrative_prompt, narrative_system, api_key)
        st.session_state['ai_narrative'] = ai_narrative
        st.success("Laporan AI berhasil dibuat!")

    if 'ai_narrative' in st.session_state:
        st.markdown("<div class='section-header'>Hasil AI Narrative — Siap Digunakan</div>", unsafe_allow_html=True)
        edited = st.text_area(
            "Edit sebelum digunakan:",
            value=st.session_state['ai_narrative'],
            height=520,
            label_visibility="collapsed"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "Download (.txt)", edited.encode('utf-8'),
                f"AI_Report_{formula_name.replace(' ','_')}.txt",
                "text/plain", use_container_width=True
            )
        with c2:
            if st.button("Regenerate", use_container_width=True):
                del st.session_state['ai_narrative']
                st.rerun()
        with c3:
            st.metric("Jumlah Kata", f"{len(edited.split()):,}")


# ═══════════════════════════════════════════════════════════════
# PREDICTIVE WAR-GAMING — SIMULASI & FORECASTING
# ═══════════════════════════════════════════════════════════════

def render_wargaming(df: pd.DataFrame, formula_name: str, formula_def: dict):
    """Render tab Predictive War-Gaming dengan Gemini AI."""
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a0a1f,#200d35); border:1px solid #ce93d844;
                border-radius:12px; padding:16px 20px; margin-bottom:16px'>
    <h3 style='color:#ce93d8; margin:0; font-size:1.3em'>&#127919; PREDICTIVE WAR-GAMING</h3>
    <p style='color:#e1bee7; margin:6px 0 0 0; font-size:0.9em'>
    Simulasikan dampak perubahan variabel terhadap KPI Anda secara real-time.<br>
    <b>Gemini AI</b> menganalisis skenario dan memberikan rekomendasi strategis.
    </p>
    </div>
    """, unsafe_allow_html=True)

    api_key = st.session_state.get('gemini_api_key', '')

    if 'calc_df' not in st.session_state or '__Result__' not in st.session_state.get('calc_df', pd.DataFrame()).columns:
        st.info("Jalankan **kalkulasi formula** terlebih dahulu di tab Analisis & Kalkulasi.")
        return

    avg_val = st.session_state.get('avg_val', 0)
    unit = st.session_state.get('unit', '')

    st.markdown("### Panel Kontrol Simulasi")
    sim_cols = st.columns(2)
    with sim_cols[0]:
        cost_change = st.slider("Perubahan Biaya / Cost (%)", -50, 100, 0, 5,
                                help="Negatif = efisiensi biaya. Positif = biaya naik.")
        revenue_change = st.slider("Perubahan Revenue / Output (%)", -50, 100, 0, 5,
                                   help="Proyeksi perubahan pendapatan atau output.")
    with sim_cols[1]:
        months_ahead = st.slider("Proyeksi ke depan (bulan)", 1, 12, 6)
        volatility = st.slider("Volatilitas Pasar (%)", 0, 30, 5,
                               help="Tingkat ketidakpastian proyeksi.")

    np.random.seed(42)
    cost_mult = 1 + cost_change / 100
    rev_mult  = 1 + revenue_change / 100
    sim_results = []
    for month in range(1, months_ahead + 1):
        trend  = 1 + (rev_mult - 1) * month / months_ahead
        cost_f = 1 + (cost_mult - 1) * month / months_ahead
        noise  = np.random.normal(0, volatility / 100 * abs(avg_val) * 0.1)
        if any(k in formula_name for k in ['ROI', 'Margin', 'Return', 'Profit']):
            val = avg_val * trend / max(cost_f, 0.01) + noise
        else:
            val = avg_val * trend + noise
        sim_results.append({
            'Periode': f"Bln +{month}",
            'Proyeksi': round(val, 4),
            'Batas Atas': round(val * (1 + volatility / 100), 4),
            'Batas Bawah': round(val * (1 - volatility / 100), 4),
        })

    df_sim = pd.DataFrame(sim_results)

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(
        x=df_sim['Periode'], y=df_sim['Proyeksi'],
        mode='lines+markers', name='Proyeksi',
        line=dict(color='#ce93d8', width=3), marker=dict(size=8)
    ))
    fig_sim.add_trace(go.Scatter(
        x=pd.concat([df_sim['Periode'], df_sim['Periode'][::-1]]),
        y=pd.concat([df_sim['Batas Atas'], df_sim['Batas Bawah'][::-1]]),
        fill='toself', fillcolor='rgba(206,147,216,0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name=f'Confidence Band +/-{volatility}%', hoverinfo='skip'
    ))
    fig_sim.add_hline(
        y=avg_val, line_dash="dash", line_color="#90caf9",
        annotation_text=f"Baseline: {avg_val:.4f} {unit}",
        annotation_position="bottom right"
    )
    fig_sim.update_layout(
        title=f"War-Game Projection — {formula_name}",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,10,25,0.5)',
        xaxis_title="Periode",
        yaxis_title=f"Nilai ({unit})" if unit else "Nilai",
        hovermode='x unified'
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    end_val   = df_sim['Proyeksi'].iloc[-1]
    delta_pct = ((end_val - avg_val) / abs(avg_val) * 100) if avg_val != 0 else 0
    scenario  = "Optimistis" if delta_pct > 5 else ("Pesimistis" if delta_pct < -5 else "Stabil")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Baseline", f"{avg_val:.4f} {unit}")
    with m2: st.metric(f"Proyeksi Bln +{months_ahead}", f"{end_val:.4f} {unit}", delta=f"{delta_pct:+.1f}%")
    with m3: st.metric("Skenario", scenario)
    with m4: st.metric("Volatilitas", f"+/-{volatility}%")

    with st.expander("Tabel Proyeksi Lengkap"):
        st.dataframe(df_sim, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### AI Strategic War-Gaming Analysis (Gemini)")
    if not api_key:
        st.warning("Masukkan Gemini API Key di sidebar untuk mengaktifkan analisis AI.")
        return

    if st.button("Minta Gemini Analisis Skenario Ini", use_container_width=True, type="primary"):
        wargame_prompt = f"""Saya menjalankan simulasi war-gaming untuk KPI: {formula_name}

PARAMETER SIMULASI:
- Baseline saat ini: {avg_val:.4f} {unit}
- Perubahan biaya: {cost_change:+}%
- Perubahan revenue: {revenue_change:+}%
- Proyeksi {months_ahead} bulan ke depan: {end_val:.4f} {unit}
- Delta vs baseline: {delta_pct:+.1f}%
- Volatilitas pasar: +/-{volatility}%
- Skenario overall: {scenario}

PROYEKSI BULANAN:
{df_sim.to_string(index=False)}

Berikan analisis war-gaming strategis:
1. ASESMEN SKENARIO: Apakah asumsi ini realistis?
2. RISIKO UTAMA: 3 risiko terbesar yang harus dimitigasi
3. TRIGGER POINT: Di level berapa manajemen harus eskalasi respons?
4. RENCANA TAKTIS: Tindakan konkret untuk 3 bulan pertama
5. CONTINGENCY PLAN: Jika proyeksi meleset lebih dari {volatility*2}%

Format: briefing eksekutif — tajam, terstruktur, actionable."""

        with st.spinner("Gemini menjalankan war-game analysis..."):
            wargame_resp = _call_gemini(wargame_prompt, _COMMANDER_SYSTEM, api_key)
        st.session_state['wargame_result'] = wargame_resp

    if 'wargame_result' in st.session_state:
        st.markdown(f"""
        <div style='background:#120820; border:1px solid #ce93d8; border-radius:12px;
                    padding:18px 20px; margin-top:12px'>
        <b style='color:#ce93d8; font-size:1.05em'>War-Game Intelligence Report — Gemini AI</b><br><br>
        <span style='color:#e0e0e0; white-space:pre-wrap; line-height:1.7'>{st.session_state['wargame_result']}</span>
        </div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download War-Game Report",
                st.session_state['wargame_result'].encode('utf-8'),
                f"WarGame_{formula_name.replace(' ','_')}.txt",
                "text/plain", use_container_width=True
            )
        with c2:
            if st.button("Reset Hasil", use_container_width=True):
                del st.session_state['wargame_result']
                st.rerun()



# ═══════════════════════════════════════════════════════════════
# ── AI VISION OCR — "MEMBACA" GAMBAR/SCREENSHOT ──
# ═══════════════════════════════════════════════════════════════

def _img_hash(img_bytes: bytes) -> str:
    """MD5 hash dari bytes gambar — untuk cache hasil OCR."""
    import hashlib
    return hashlib.md5(img_bytes).hexdigest()


def _compress_image_for_gemini(img: "Image.Image", quality: int = 60, max_px: int = 768) -> "Image.Image":
    """
    Kompres gambar sebelum dikirim ke Gemini.
    - Resize ke max_px di sisi terpanjang
    - Konversi ke RGB (hapus transparansi)
    Makin kecil gambar = makin sedikit token = hemat kuota.
    """
    # Fix mode (RGBA/P → RGB)
    if img.mode in ('RGBA', 'LA', 'P'):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        bg.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        img = bg
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize
    w, h = img.size
    if max(w, h) > max_px:
        ratio = max_px / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    return img


def _estimate_tokens(img: "Image.Image") -> int:
    """Estimasi kasar jumlah token gambar (Gemini ~258 token per 512×512 tile)."""
    w, h = img.size
    tiles = max(1, (w // 512) + 1) * max(1, (h // 512) + 1)
    return tiles * 258


def render_vision_ocr():
    """Render tab AI Vision OCR — hemat kuota dengan cache & kompresi."""
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0a1f1a,#0d2d25); border:1px solid #00bfa544;
                border-radius:12px; padding:16px 20px; margin-bottom:16px'>
    <h3 style='color:#00e5b4; margin:0; font-size:1.3em'>&#128065;&#65039; AI VISION OCR — Baca Gambar Jadi Data</h3>
    <p style='color:#80cbc4; margin:6px 0 0 0; font-size:0.9em'>
    Upload screenshot laporan, foto tabel, atau gambar grafik.<br>
    <b>Gemini Vision</b> mengekstrak data → DataFrame CSV instan.<br>
    <span style='color:#4db6ac'>&#9889; Hemat kuota: kompresi otomatis + cache hasil.</span>
    </p>
    </div>
    """, unsafe_allow_html=True)

    api_key = st.session_state.get('gemini_api_key', '')
    if not api_key:
        st.warning("Masukkan **Gemini API Key** di sidebar untuk mengaktifkan AI Vision OCR.")
        return
    if not _GEMINI_OK:
        st.error("Install library: `pip install google-generativeai`")
        return
    if not _PIL_OK:
        st.error("Install Pillow: `pip install Pillow`")
        return

    # ── Layout upload + info ──
    col_upload, col_info = st.columns([2, 1])
    with col_upload:
        uploaded_img = st.file_uploader(
            "Upload gambar (screenshot, foto tabel, grafik):",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key="vision_upload"
        )
    with col_info:
        st.markdown("""
        <div style='background:#0a1f1a; border:1px solid #00bfa533; border-radius:8px;
                    padding:12px; font-size:0.83em; color:#80cbc4'>
        <b>Contoh yang bisa dibaca:</b><br>
        ✅ Screenshot tabel Excel / Power BI<br>
        ✅ Foto laporan keuangan cetak<br>
        ✅ Tabel dari PDF scan<br>
        ✅ Grafik dengan angka tertera<br><br>
        <b style='color:#4db6ac'>💡 Tips hemat kuota:</b><br>
        • Crop gambar — buang area kosong<br>
        • Gunakan mode "Tabel → CSV" (paling efisien)<br>
        • Hasil di-cache: gambar sama = gratis
        </div>
        """, unsafe_allow_html=True)

    extract_mode = st.radio(
        "Mode Ekstraksi:",
        ["Tabel/Data Numerik → CSV", "Grafik → Deskripsi + Angka", "Laporan Teks → Ringkasan Data"],
        horizontal=True
    )

    # ── Pengaturan kompresi ──
    with st.expander("⚙️ Pengaturan Kompresi (hemat kuota)", expanded=False):
        cq1, cq2 = st.columns(2)
        with cq1:
            max_px = st.select_slider(
                "Resolusi Maksimum:",
                options=[384, 512, 640, 768, 1024],
                value=512,
                help="Makin kecil = makin hemat kuota. 512px cukup untuk tabel terbaca."
            )
        with cq2:
            jpeg_q = st.slider(
                "Kualitas JPEG:", 40, 95, 60,
                help="Makin rendah = ukuran file lebih kecil = lebih hemat kuota."
            )

    custom_instruction = st.text_input(
        "Instruksi tambahan (opsional):",
        placeholder="Contoh: fokus pada kolom revenue dan profit saja"
    )

    # ── Preview & estimasi sebelum kirim ──
    if uploaded_img:
        raw_bytes = uploaded_img.getvalue()
        img_orig = Image.open(_io.BytesIO(raw_bytes))
        img_comp = _compress_image_for_gemini(img_orig, jpeg_q, max_px)

        # Hitung ukuran setelah kompresi
        buf_comp = _io.BytesIO()
        img_comp.save(buf_comp, format='JPEG', quality=jpeg_q)
        comp_bytes = buf_comp.getvalue()
        orig_kb = len(raw_bytes) / 1024
        comp_kb = len(comp_bytes) / 1024
        est_tokens = _estimate_tokens(img_comp)
        saved_pct = (1 - comp_kb / orig_kb) * 100 if orig_kb > 0 else 0

        # Tampilkan metrik
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Ukuran Asli",    f"{orig_kb:.0f} KB")
        with m2: st.metric("Setelah Kompres", f"{comp_kb:.0f} KB", delta=f"-{saved_pct:.0f}%")
        with m3: st.metric("Resolusi Kirim",  f"{img_comp.width}×{img_comp.height}px")
        with m4: st.metric("Est. Token Gambar", f"~{est_tokens:,}")

        # Preview gambar yang akan dikirim
        col_prev1, col_prev2 = st.columns(2)
        with col_prev1:
            st.image(img_orig, caption=f"Asli ({img_orig.width}×{img_orig.height}px)", use_container_width=True)
        with col_prev2:
            st.image(img_comp, caption=f"Akan dikirim ({img_comp.width}×{img_comp.height}px)", use_container_width=True)

        # Cek cache — hash berdasarkan gambar TERKOMPRESI + mode
        cache_key = f"vision_cache_{_img_hash(comp_bytes)}_{extract_mode[:6]}"
        is_cached = cache_key in st.session_state

        if is_cached:
            st.success("✅ **Hasil tersedia dari cache** — tidak perlu panggil API lagi! (0 token terpakai)")
            if st.button("Gunakan Hasil Cache", use_container_width=True, type="primary"):
                st.session_state['vision_raw']  = st.session_state[cache_key]['raw']
                st.session_state['vision_mode'] = st.session_state[cache_key]['mode']
                st.rerun()
        else:
            btn_label = f"🔍 Ekstrak Data ({est_tokens:,} token est.)"
            if st.button(btn_label, use_container_width=True, type="primary"):

                # ── Prompt ringkas (hemat input token) ──
                if extract_mode == "Tabel/Data Numerik → CSV":
                    prompt = (
                        "Ekstrak semua data tabel dari gambar ini. "
                        "Output HANYA format CSV mentah (comma separator, header baris pertama). "
                        "Tidak ada penjelasan. Angka ribuan pakai titik → hapus titiknya."
                    )
                elif extract_mode == "Grafik → Deskripsi + Angka":
                    prompt = (
                        "Analisis grafik ini. Output: "
                        "1) Jenis & deskripsi singkat, "
                        "2) Data CSV (label,nilai), "
                        "3) 3 insight utama."
                    )
                else:
                    prompt = (
                        "Baca laporan ini. Output CSV: Metrik,Nilai,Satuan. "
                        "Tambah ringkasan 2 kalimat di akhir."
                    )

                if custom_instruction:
                    prompt += f" FOKUS: {custom_instruction}"

                with st.spinner(f"Gemini Vision memproses ({est_tokens:,} token est.)..."):
                    try:
                        # Pakai model yang dipilih user di sidebar — langsung, tanpa auto-detect
                        model_nm = st.session_state.get('gemini_model', 'gemini-2.0-flash')
                        genai.configure(api_key=api_key.strip())
                        model    = genai.GenerativeModel(model_nm)
                        response = model.generate_content([prompt, img_comp])
                        raw_text = response.text

                        st.session_state[cache_key]     = {'raw': raw_text, 'mode': extract_mode}
                        st.session_state['vision_raw']  = raw_text
                        st.session_state['vision_mode'] = extract_mode
                        st.success(f"✅ Berhasil! Model: `{model_nm}` | {img_comp.width}×{img_comp.height}px")
                        st.rerun()

                    except Exception as e:
                        err_str = str(e)
                        is_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()
                        if is_limit:
                            st.error(
                                "⚠️ **API Limit tercapai.**\n\n"
                                "**Kemungkinan penyebab:**\n"
                                "- Rate limit per menit (15 RPM) → tunggu ~1 menit lalu klik lagi\n"
                                "- Kuota harian habis → tunggu reset jam **07:00 WIB**\n\n"
                                "**Tips:** Gambar yang sama tidak akan re-call API (sudah di-cache).\n"
                                "Jika masih gagal, buat API Key baru di [aistudio.google.com](https://aistudio.google.com/app/apikey)"
                            )
                        elif "API_KEY_INVALID" in err_str or "API key not valid" in err_str:
                            st.error("❌ API Key tidak valid. Cek kembali di sidebar.")
                        elif "not found" in err_str.lower() or "404" in err_str:
                            st.error(
                                f"❌ Model `{model_nm}` tidak ditemukan di akun Anda.\n\n"
                                f"Coba ganti model di sidebar ke pilihan lain."
                            )
                        else:
                            st.error(f"❌ Error: {err_str[:400]}")
                        return

    # ── Tampilkan hasil ──
    if 'vision_raw' in st.session_state:
        raw_text = st.session_state['vision_raw']
        mode     = st.session_state.get('vision_mode', '')

        st.markdown("---")
        st.markdown("### Hasil Ekstraksi AI Vision")

        csv_match   = re.search(r'```(?:csv)?\s*([\s\S]+?)```', raw_text)
        csv_content = csv_match.group(1).strip() if csv_match else raw_text.strip()

        with st.expander("Raw Output dari Gemini Vision"):
            st.text(raw_text)

        # Coba parse ke DataFrame
        if "Tabel" in mode or "Grafik" in mode:
            try:
                import io
                df_vision = pd.read_csv(io.StringIO(csv_content))
                st.success(f"DataFrame berhasil dibuat: {df_vision.shape[0]} baris x {df_vision.shape[1]} kolom")
                st.dataframe(df_vision, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    csv_dl = df_vision.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV", csv_dl, "vision_extracted.csv", "text/csv", use_container_width=True)
                with c2:
                    if st.button("Gunakan sebagai Data Analisis", use_container_width=True, type="primary"):
                        st.session_state['df_clean'] = df_vision
                        st.session_state['vision_to_analysis'] = True
                        st.success("Data siap! Pergi ke tab Analisis & Kalkulasi.")
                with c3:
                    st.metric("Kolom Terdeteksi", len(df_vision.columns))

                # Auto visualize hasil ekstraksi
                numeric_cols = df_vision.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 1 and len(df_vision) > 1:
                    st.markdown("**Preview Chart dari Data Ekstraksi:**")
                    fig_v = px.bar(
                        df_vision, y=numeric_cols[0],
                        title=f"Data dari Gambar — {numeric_cols[0]}",
                        template='plotly_dark', color_discrete_sequence=['#00e5b4']
                    )
                    fig_v.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,31,26,0.5)')
                    st.plotly_chart(fig_v, use_container_width=True)

            except Exception:
                st.info("Format bukan CSV standar — menampilkan sebagai teks terstruktur:")
                st.text_area("Hasil Ekstraksi:", value=csv_content, height=300)
                st.download_button("Download Teks", csv_content.encode('utf-8'),
                                   "vision_extracted.txt", "text/plain", use_container_width=True)
        else:
            st.text_area("Ringkasan Data dari Laporan:", value=raw_text, height=350)

        if st.button("Reset / Upload Gambar Baru"):
            for k in ['vision_raw', 'vision_mode']:
                st.session_state.pop(k, None)
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# ── ANOMALY DETECTION — SISTEM DETEKSI ANOMALI OTOMATIS ──
# ═══════════════════════════════════════════════════════════════

def render_anomaly_detection(df: pd.DataFrame):
    """Render tab Anomaly Detection dengan Isolation Forest & LOF."""
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1f0a0a,#2d1010); border:1px solid #ef535044;
                border-radius:12px; padding:16px 20px; margin-bottom:16px'>
    <h3 style='color:#ff6b6b; margin:0; font-size:1.3em'>&#128680; AUTOMATED ANOMALY DETECTION</h3>
    <p style='color:#ffcdd2; margin:6px 0 0 0; font-size:0.9em'>
    Machine Learning mendeteksi data <b>aneh/outlier</b> secara otomatis.<br>
    Gunakan <b>Isolation Forest</b> atau <b>Local Outlier Factor</b> untuk mengidentifikasi transaksi mencurigakan,
    lonjakan tidak wajar, atau data yang perlu investigasi.
    </p>
    </div>
    """, unsafe_allow_html=True)

    if not _SKLEARN_OK:
        st.error("Install scikit-learn: `pip install scikit-learn`")
        return

    if df is None or len(df) == 0:
        st.info("Upload dan bersihkan data di tab **Analisis & Kalkulasi** terlebih dahulu.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        st.warning("Tidak ada kolom numerik untuk analisis anomali.")
        return

    # Konfigurasi
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        method = st.selectbox(
            "Algoritma Deteksi:",
            ["Isolation Forest (Direkomendasikan)", "Local Outlier Factor", "Z-Score (Statistikal)"],
            help="Isolation Forest: terbaik untuk data multidimensi\nLOF: baik untuk deteksi kluster lokal"
        )
    with col_cfg2:
        contamination = st.slider(
            "Estimasi % Anomali:", 1, 20, 5,
            help="Perkiraan persentase data yang dianggap anomali. Default 5%."
        ) / 100
    with col_cfg3:
        selected_cols = st.multiselect(
            "Kolom yang Dianalisis:",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))],
            help="Pilih kolom yang relevan untuk deteksi anomali"
        )

    if not selected_cols:
        st.warning("Pilih minimal 1 kolom untuk dianalisis.")
        return

    if st.button("Jalankan Deteksi Anomali", use_container_width=True, type="primary"):
        with st.spinner("Machine Learning sedang menganalisis pola data..."):
            try:
                df_work = df[selected_cols].copy().fillna(df[selected_cols].median())
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_work)

                if "Isolation Forest" in method:
                    detector = IsolationForest(
                        contamination=contamination, random_state=42, n_estimators=200
                    )
                    preds = detector.fit_predict(X_scaled)
                    scores = detector.score_samples(X_scaled)
                    label_name = "Isolation Forest"

                elif "Local Outlier Factor" in method:
                    detector = LocalOutlierFactor(
                        n_neighbors=min(20, len(df_work)-1),
                        contamination=contamination
                    )
                    preds = detector.fit_predict(X_scaled)
                    scores = detector.negative_outlier_factor_
                    label_name = "Local Outlier Factor"

                else:  # Z-Score
                    z_scores = np.abs((X_scaled))
                    max_z = z_scores.max(axis=1)
                    threshold = 2.5
                    preds = np.where(max_z > threshold, -1, 1)
                    scores = -max_z
                    label_name = "Z-Score"

                df_result = df.copy()
                df_result['__Anomaly__'] = np.where(preds == -1, 'ANOMALI', 'Normal')
                df_result['__AnomalyScore__'] = np.round(-scores, 4)

                n_anomali = (preds == -1).sum()
                n_normal = (preds == 1).sum()
                anomaly_pct = n_anomali / len(df) * 100

                st.session_state['anomaly_df'] = df_result
                st.session_state['anomaly_method'] = label_name
                st.session_state['n_anomali'] = n_anomali

            except Exception as e:
                st.error(f"Error deteksi anomali: {str(e)}")
                return

    if 'anomaly_df' in st.session_state:
        df_result = st.session_state['anomaly_df']
        n_anomali = st.session_state.get('n_anomali', 0)
        label_name = st.session_state.get('anomaly_method', '')
        n_normal = len(df_result) - n_anomali
        anomaly_pct = n_anomali / len(df_result) * 100

        # Metrik ringkasan
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Total Data", f"{len(df_result):,}")
        with m2: st.metric("Normal", f"{n_normal:,}", delta=f"{100-anomaly_pct:.1f}%")
        with m3: st.metric("Anomali Terdeteksi", f"{n_anomali:,}",
                           delta=f"{anomaly_pct:.1f}%", delta_color="inverse")
        with m4: st.metric("Metode", label_name.split()[0])

        # Status keseluruhan
        if anomaly_pct > 10:
            health_color, health_icon, health_text = "#ef5350", "🔴", "KRITIS — Anomali tinggi, investigasi segera!"
        elif anomaly_pct > 5:
            health_color, health_icon, health_text = "#ff9800", "🟡", "WASPADA — Ada anomali signifikan"
        else:
            health_color, health_icon, health_text = "#4caf50", "🟢", "SEHAT — Anomali dalam batas wajar"

        st.markdown(f"""
        <div style='background:#1a0d0d; border:2px solid {health_color}; border-radius:10px;
                    padding:12px 16px; margin:12px 0; text-align:center'>
        <span style='font-size:1.4em'>{health_icon}</span>
        <b style='color:{health_color}; font-size:1.1em; margin-left:8px'>STATUS KESEHATAN DATA: {health_text}</b>
        </div>
        """, unsafe_allow_html=True)

        # Visualisasi 1: Scatter plot anomali (2 kolom pertama)
        if len(selected_cols) >= 2:
            fig_scatter = px.scatter(
                df_result, x=selected_cols[0], y=selected_cols[1],
                color='__Anomaly__',
                color_discrete_map={'ANOMALI': '#ef5350', 'Normal': '#4fc3f7'},
                size='__AnomalyScore__',
                hover_data=selected_cols[:4],
                title=f"Peta Anomali — {selected_cols[0]} vs {selected_cols[1]}",
                template='plotly_dark'
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(25,5,5,0.5)'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Visualisasi 2: Anomaly score distribution
        fig_dist = go.Figure()
        normal_scores = df_result[df_result['__Anomaly__'] == 'Normal']['__AnomalyScore__']
        anomaly_scores = df_result[df_result['__Anomaly__'] == 'ANOMALI']['__AnomalyScore__']
        fig_dist.add_trace(go.Histogram(x=normal_scores, name='Normal',
                                        marker_color='#4fc3f7', opacity=0.7, nbinsx=30))
        fig_dist.add_trace(go.Histogram(x=anomaly_scores, name='Anomali',
                                        marker_color='#ef5350', opacity=0.8, nbinsx=20))
        fig_dist.update_layout(
            barmode='overlay', title='Distribusi Anomaly Score',
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(25,5,5,0.5)',
            xaxis_title='Anomaly Score', yaxis_title='Jumlah Data'
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Visualisasi 3: Top anomali per kolom
        st.markdown("### 🚨 Top 10 Baris Paling Anomali")
        df_anomali = df_result[df_result['__Anomaly__'] == 'ANOMALI'].sort_values(
            '__AnomalyScore__', ascending=False
        ).head(10)
        if len(df_anomali) > 0:
            # Bangun Plotly table — tidak butuh jinja2, lebih kaya visual
            display_cols = ['__AnomalyScore__'] + [
                c for c in selected_cols if c in df_anomali.columns
            ][:6]
            tbl_df = df_anomali[display_cols].reset_index(drop=True)

            # Warna per baris berdasarkan AnomalyScore (merah makin gelap = makin anomali)
            score_vals = tbl_df['__AnomalyScore__'].values
            s_min, s_max = score_vals.min(), score_vals.max()
            def _score_color(v):
                if s_max == s_min:
                    ratio = 1.0
                else:
                    ratio = (v - s_min) / (s_max - s_min)
                r = int(180 + ratio * 75)          # 180 → 255
                g = int(30  - ratio * 10)          # 30  → 20
                b = int(30  - ratio * 10)          # 30  → 20
                return f"rgb({r},{g},{b})"

            row_colors = []
            for col_name in tbl_df.columns:
                if col_name == '__AnomalyScore__':
                    row_colors.append([_score_color(v) for v in score_vals])
                else:
                    row_colors.append(['rgba(15,30,50,0.85)'] * len(tbl_df))

            header_vals = [c.replace('__AnomalyScore__', '🔴 Anomaly Score')
                           for c in tbl_df.columns]
            cell_vals   = [tbl_df[c].round(4).tolist()
                           if pd.api.types.is_float_dtype(tbl_df[c]) else tbl_df[c].tolist()
                           for c in tbl_df.columns]

            fig_tbl = go.Figure(data=[go.Table(
                columnwidth=[120] + [100] * (len(display_cols) - 1),
                header=dict(
                    values=header_vals,
                    fill_color='#1a3a5c',
                    font=dict(color='#90caf9', size=12, family='Space Grotesk'),
                    align='center',
                    height=36,
                    line_color='#2a5f8c',
                ),
                cells=dict(
                    values=cell_vals,
                    fill_color=row_colors,
                    font=dict(color='white', size=11, family='Space Grotesk'),
                    align='center',
                    height=30,
                    line_color='rgba(42,95,140,0.4)',
                ),
            )])
            fig_tbl.update_layout(
                margin=dict(l=0, r=0, t=8, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                height=max(120, 36 + 30 * len(tbl_df) + 20),
            )
            st.plotly_chart(fig_tbl, use_container_width=True)

            # Bar chart horizontal anomaly score
            fig_bar_anom = go.Figure(go.Bar(
                x=score_vals[::-1],
                y=[f"Baris #{i+1}" for i in range(len(score_vals))][::-1],
                orientation='h',
                marker=dict(
                    color=score_vals[::-1],
                    colorscale=[[0,'#ff8a65'],[0.5,'#ef5350'],[1,'#b71c1c']],
                    showscale=True,
                    colorbar=dict(title='Score', thickness=12),
                ),
                text=[f"{v:.4f}" for v in score_vals[::-1]],
                textposition='outside',
                textfont=dict(color='#ffccbc', size=10),
            ))
            fig_bar_anom.update_layout(
                title="Anomaly Score — Top 10 Tertinggi",
                xaxis_title="Anomaly Score",
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,5,5,0.5)',
                margin=dict(l=80, r=60, t=40, b=30),
                height=320,
            )
            st.plotly_chart(fig_bar_anom, use_container_width=True)
        else:
            st.info("Tidak ada anomali terdeteksi dengan parameter saat ini.")

        # Box plot per kolom untuk lihat distribusi
        st.markdown("### Distribusi Nilai per Kolom (Normal vs Anomali)")
        for col in selected_cols[:4]:
            fig_box = px.box(
                df_result, x='__Anomaly__', y=col,
                color='__Anomaly__',
                color_discrete_map={'ANOMALI': '#ef5350', 'Normal': '#4fc3f7'},
                title=f"Distribusi {col}",
                template='plotly_dark'
            )
            fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(25,5,5,0.4)', showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        # Download hasil
        c1, c2 = st.columns(2)
        with c1:
            csv_anomaly = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("Download Hasil Lengkap (CSV)", csv_anomaly,
                               "anomaly_detection_result.csv", "text/csv", use_container_width=True)
        with c2:
            only_anomali = df_result[df_result['__Anomaly__'] == 'ANOMALI'].to_csv(index=False).encode('utf-8')
            st.download_button("Download Hanya Anomali (CSV)", only_anomali,
                               "anomali_only.csv", "text/csv", use_container_width=True)

        # AI Analysis anomali
        api_key = st.session_state.get('gemini_api_key', '')
        if api_key and n_anomali > 0:
            st.markdown("---")
            st.markdown("### Gemini AI — Interpretasi Anomali")
            if st.button("Minta Gemini Analisis Pola Anomali", use_container_width=True):
                top_anomali_str = df_anomali[selected_cols[:5]].to_string() if len(df_anomali) > 0 else "Tidak ada"
                stats_str = df_result[selected_cols].describe().to_string()
                anomaly_prompt = f"""Saya menjalankan {label_name} pada dataset dan menemukan anomali.

HASIL DETEKSI:
- Total data: {len(df_result):,} baris
- Anomali terdeteksi: {n_anomali} ({anomaly_pct:.1f}%)
- Kolom yang dianalisis: {', '.join(selected_cols)}

STATISTIK DATA:
{stats_str}

TOP 10 ANOMALI (nilai tertinggi anomaly score):
{top_anomali_str}

Berikan analisis:
1. INTERPRETASI: Apa yang menyebabkan anomali ini? (berdasarkan nilai yang terdeteksi)
2. RISIKO: Apa risiko bisnis dari anomali ini?
3. INVESTIGASI: Kolom mana yang paling perlu diinvestigasi? Kenapa?
4. REKOMENDASI: Langkah preventif yang harus diambil manajemen
5. PRIORITAS: Apakah ini perlu eskalasi ke level direktur? Kenapa?

Format: briefing eksekutif, gunakan angka spesifik, maksimal 500 kata."""

                with st.spinner("Gemini menganalisis pola anomali..."):
                    ai_resp = _call_gemini(anomaly_prompt, _COMMANDER_SYSTEM, api_key)
                    st.session_state['anomaly_ai'] = ai_resp

        if 'anomaly_ai' in st.session_state:
            st.markdown(f"""
            <div style='background:#1f0505; border:1px solid #ef5350; border-radius:12px;
                        padding:18px; margin-top:12px'>
            <b style='color:#ef9a9a'>Anomaly Intelligence Report — Gemini AI</b><br><br>
            <span style='color:#e0e0e0; white-space:pre-wrap; line-height:1.7'>{st.session_state['anomaly_ai']}</span>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ── ADVANCED FORECASTING — PROPHET / ETS + CONFIDENCE INTERVAL ──
# ═══════════════════════════════════════════════════════════════

def render_advanced_forecast(df: pd.DataFrame):
    """Render tab Advanced Forecasting dengan Prophet/ETS dan Confidence Interval."""
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0f0f1f,#151530); border:1px solid #7986cb44;
                border-radius:12px; padding:16px 20px; margin-bottom:16px'>
    <h3 style='color:#9fa8da; margin:0; font-size:1.3em'>&#128202; ADVANCED FORECASTING</h3>
    <p style='color:#c5cae9; margin:6px 0 0 0; font-size:0.9em'>
    Prediksi masa depan menggunakan <b>Exponential Smoothing (ETS)</b> atau <b>Prophet (Meta)</b>
    dengan Confidence Interval &mdash; area bayangan yang menunjukkan tingkat ketidakpastian.<br>
    <span style='color:#7986cb'>Dari alat pelaporan menjadi alat perencanaan strategis.</span>
    </p>
    </div>
    """, unsafe_allow_html=True)

    if df is None or len(df) == 0:
        st.info("Upload dan bersihkan data terlebih dahulu.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("Tidak ada kolom numerik untuk forecasting.")
        return

    # Konfigurasi
    col1, col2, col3 = st.columns(3)
    with col1:
        target_col = st.selectbox("Kolom Target (yang diprediksi):", numeric_cols)
    with col2:
        # Cek apakah ada kolom tanggal
        date_cols = [c for c in df.columns if any(k in c.lower() for k in
                     ['date', 'tanggal', 'tgl', 'bulan', 'month', 'waktu', 'time', 'periode'])]
        date_col_options = ["Gunakan Index (urutan baris)"] + date_cols
        date_col = st.selectbox("Kolom Tanggal/Periode (opsional):", date_col_options)
    with col3:
        forecast_periods = st.slider("Periode Prediksi ke Depan:", 3, 24, 6,
                                     help="Jumlah periode (bulan/baris) yang akan diprediksi")

    col4, col5 = st.columns(2)
    with col4:
        method_options = ["Exponential Smoothing (ETS)"]
        if _PROPHET_OK:
            method_options.append("Prophet (Meta) — Lebih Canggih")
        method_options.append("Holt Linear (Trend)")
        forecast_method = st.selectbox("Metode Forecasting:", method_options)
    with col5:
        confidence = st.select_slider(
            "Confidence Interval:", options=[80, 85, 90, 95, 99], value=95,
            help="95% artinya nilai aktual 95% kemungkinan masuk dalam area bayangan"
        )

    if st.button("Jalankan Advanced Forecasting", use_container_width=True, type="primary"):
        if not _STATSMODELS_OK and "ETS" in forecast_method:
            st.error("Install statsmodels: `pip install statsmodels`")
            return
        if _PROPHET_OK is False and "Prophet" in forecast_method:
            st.error("Install Prophet: `pip install prophet`")
            return

        with st.spinner("Model forecasting sedang dilatih..."):
            try:
                # Siapkan data time series
                series = df[target_col].dropna().values
                n = len(series)

                if n < 6:
                    st.error("Data minimal 6 titik untuk forecasting yang bermakna.")
                    return

                alpha_ci = 1 - confidence / 100

                if "Prophet" in forecast_method and _PROPHET_OK:
                    # Setup Prophet
                    if date_col != "Gunakan Index (urutan baris)" and date_col in df.columns:
                        try:
                            ds_col = pd.to_datetime(df[date_col].dropna())
                        except Exception:
                            ds_col = pd.date_range(start='2023-01-01', periods=n, freq='MS')
                    else:
                        ds_col = pd.date_range(start='2023-01-01', periods=n, freq='MS')

                    prophet_df = pd.DataFrame({'ds': ds_col[:n], 'y': series})
                    m_prophet = Prophet(interval_width=confidence/100, yearly_seasonality='auto',
                                       weekly_seasonality=False, daily_seasonality=False)
                    m_prophet.fit(prophet_df)
                    future = m_prophet.make_future_dataframe(periods=forecast_periods, freq='MS')
                    forecast_df = m_prophet.predict(future)

                    # Pisahkan historis dan prediksi
                    hist_df = forecast_df.iloc[:n]
                    pred_df = forecast_df.iloc[n:]

                    dates_hist = [d.strftime('%b %Y') for d in hist_df['ds']]
                    dates_pred = [d.strftime('%b %Y') for d in pred_df['ds']]
                    y_fitted = hist_df['yhat'].values
                    y_pred = pred_df['yhat'].values
                    y_lower = pred_df['yhat_lower'].values
                    y_upper = pred_df['yhat_upper'].values
                    method_label = "Prophet (Meta)"

                else:
                    # Exponential Smoothing / Holt
                    if "Holt" in forecast_method:
                        model_ets = ExponentialSmoothing(series, trend='add', seasonal=None)
                    else:
                        seasonal_periods = min(12, max(2, n // 2))
                        if n >= seasonal_periods * 2:
                            model_ets = ExponentialSmoothing(
                                series, trend='add', seasonal='add',
                                seasonal_periods=seasonal_periods
                            )
                        else:
                            model_ets = ExponentialSmoothing(series, trend='add')

                    fitted_ets = model_ets.fit(optimized=True)
                    y_fitted = fitted_ets.fittedvalues
                    y_pred = fitted_ets.forecast(forecast_periods)

                    # Confidence interval via residual std
                    residuals = series - y_fitted
                    resid_std = np.std(residuals)
                    z = scipy_stats.norm.ppf(1 - alpha_ci / 2)
                    ci_widths = [z * resid_std * np.sqrt(i + 1) for i in range(forecast_periods)]
                    y_lower = y_pred - ci_widths
                    y_upper = y_pred + ci_widths

                    # Generate date labels
                    dates_hist = [f"P{i+1}" for i in range(n)]
                    dates_pred = [f"P{n+i+1}" for i in range(forecast_periods)]
                    method_label = "Exponential Smoothing (ETS)" if "ETS" in forecast_method else "Holt Linear"

                st.session_state['forecast_result'] = {
                    'dates_hist': dates_hist,
                    'dates_pred': dates_pred,
                    'series': series,
                    'y_fitted': y_fitted,
                    'y_pred': y_pred,
                    'y_lower': y_lower,
                    'y_upper': y_upper,
                    'target_col': target_col,
                    'method_label': method_label,
                    'confidence': confidence,
                    'forecast_periods': forecast_periods,
                }

            except Exception as e:
                st.error(f"Error forecasting: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
                return

    if 'forecast_result' in st.session_state:
        fr = st.session_state['forecast_result']
        dates_hist   = fr['dates_hist']
        dates_pred   = fr['dates_pred']
        series       = np.array(fr['series'], dtype=float)
        y_fitted     = np.array(fr['y_fitted'], dtype=float)
        y_pred       = np.array(fr['y_pred'], dtype=float)
        y_lower      = np.array(fr['y_lower'], dtype=float)
        y_upper      = np.array(fr['y_upper'], dtype=float)
        target_col_r = fr['target_col']
        method_label = fr['method_label']
        confidence_r = fr['confidence']
        forecast_periods_r = fr['forecast_periods']

        n_h = len(dates_hist)
        n_p = len(dates_pred)
        # Indeks numerik — WAJIB agar add_shape & separator berfungsi
        # (add_vline TIDAK bisa terima string categorical x-axis)
        x_h = list(range(n_h))
        x_p = list(range(n_h, n_h + n_p))
        all_labels = list(dates_hist) + list(dates_pred)

        # ── CHART UTAMA ──
        fig_fc = go.Figure()

        # Confidence interval — area bayangan
        fig_fc.add_trace(go.Scatter(
            x=x_p + x_p[::-1],
            y=list(y_upper) + list(y_lower[::-1]),
            fill='toself', fillcolor='rgba(121,134,203,0.18)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'CI {confidence_r}%', hoverinfo='skip'
        ))
        # Data aktual historis
        fig_fc.add_trace(go.Scatter(
            x=x_h, y=series, mode='lines+markers', name='Data Aktual',
            line=dict(color='#4fc3f7', width=2), marker=dict(size=5),
            customdata=dates_hist,
            hovertemplate='<b>%{customdata}</b><br>Aktual: %{y:,.4f}<extra></extra>'
        ))
        # Model fit
        fig_fc.add_trace(go.Scatter(
            x=x_h, y=y_fitted, mode='lines', name='Model Fit',
            line=dict(color='#ffd54f', width=1.5, dash='dot'), opacity=0.8,
            customdata=dates_hist,
            hovertemplate='<b>%{customdata}</b><br>Fitted: %{y:,.4f}<extra></extra>'
        ))
        # Prediksi
        fig_fc.add_trace(go.Scatter(
            x=x_p, y=y_pred, mode='lines+markers',
            name=f'Prediksi ({method_label})',
            line=dict(color='#ce93d8', width=3), marker=dict(size=9, symbol='diamond'),
            customdata=dates_pred,
            hovertemplate='<b>%{customdata}</b><br>Prediksi: %{y:,.4f}<extra></extra>'
        ))
        # CI upper
        fig_fc.add_trace(go.Scatter(
            x=x_p, y=y_upper, mode='lines',
            line=dict(color='#9fa8da', width=1, dash='dash'),
            name='Batas Atas CI', opacity=0.6,
            customdata=dates_pred,
            hovertemplate='<b>%{customdata}</b><br>Atas: %{y:,.4f}<extra></extra>'
        ))
        # CI lower
        fig_fc.add_trace(go.Scatter(
            x=x_p, y=y_lower, mode='lines',
            line=dict(color='#9fa8da', width=1, dash='dash'),
            name='Batas Bawah CI', opacity=0.6,
            customdata=dates_pred,
            hovertemplate='<b>%{customdata}</b><br>Bawah: %{y:,.4f}<extra></extra>'
        ))

        # Garis pemisah — pakai add_shape (aman untuk numeric x-axis)
        sep = n_h - 0.5
        fig_fc.add_shape(type="line", x0=sep, x1=sep, y0=0, y1=1,
                         xref="x", yref="paper",
                         line=dict(color="#78909c", width=2, dash="dash"))
        fig_fc.add_annotation(x=sep + 0.2, y=0.97, xref="x", yref="paper",
                              text="▶ Mulai Prediksi", showarrow=False,
                              font=dict(color="#b0bec5", size=11), xanchor="left")

        # Tick labels: tampilkan nama periode, bukan angka
        total_pts = n_h + n_p
        step = max(1, total_pts // 14)
        tick_vals = list(range(0, total_pts, step))
        tick_text = [all_labels[i] for i in tick_vals if i < len(all_labels)]

        fig_fc.update_layout(
            title=f"Advanced Forecast — {target_col_r} | {method_label} | CI {confidence_r}%",
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(5,5,20,0.6)',
            xaxis=dict(title="Periode", tickmode='array',
                       tickvals=tick_vals, ticktext=tick_text, tickangle=-35),
            yaxis_title=target_col_r,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Metrik proyeksi
        last_actual = series[-1]
        last_pred = y_pred[-1]
        delta_fc = ((last_pred - last_actual) / abs(last_actual) * 100) if last_actual != 0 else 0
        trend_dir = "Naik" if delta_fc > 0 else "Turun"
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Nilai Terakhir Aktual", f"{last_actual:.4f}")
        with m2: st.metric(f"Prediksi +{forecast_periods_r} periode",
                           f"{last_pred:.4f}", delta=f"{delta_fc:+.1f}%")
        with m3: st.metric("Tren", f"{'📈' if delta_fc>0 else '📉'} {trend_dir}")
        with m4:
            range_ci = (y_upper[-1] - y_lower[-1]) / 2
            st.metric(f"Uncertainty ±(CI {confidence_r}%)", f"±{range_ci:.4f}")

        # Tabel prediksi
        df_fc_table = pd.DataFrame({
            'Periode': dates_pred,
            'Prediksi': np.round(y_pred, 4),
            f'Batas Bawah ({confidence_r}%)': np.round(y_lower, 4),
            f'Batas Atas ({confidence_r}%)': np.round(y_upper, 4),
            'Uncertainty (±)': np.round((y_upper - y_lower) / 2, 4)
        })
        with st.expander("Tabel Prediksi Lengkap"):
            st.dataframe(df_fc_table, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Tabel Prediksi (CSV)",
                df_fc_table.to_csv(index=False).encode('utf-8'),
                f"forecast_{target_col_r}.csv", "text/csv", use_container_width=True
            )

        # AI Forecast Interpretation
        api_key = st.session_state.get('gemini_api_key', '')
        if api_key:
            st.markdown("---")
            st.markdown("### Gemini AI — Interpretasi Strategis Prediksi")
            if st.button("Minta Gemini Interpretasi Hasil Forecast", use_container_width=True):
                fc_prompt = f"""Saya baru menjalankan Advanced Forecasting untuk metrik: {target_col_r}

METODE: {method_label}
CONFIDENCE INTERVAL: {confidence_r}%

DATA HISTORIS (ringkasan):
- Jumlah titik data: {len(series)}
- Nilai min: {series.min():.4f}, max: {series.max():.4f}, mean: {series.mean():.4f}
- Nilai terakhir aktual: {last_actual:.4f}

HASIL PREDIKSI ({forecast_periods_r} periode ke depan):
{df_fc_table.to_string(index=False)}

ANALISIS YANG DIBUTUHKAN:
1. INTERPRETASI TREN: Apa yang ditunjukkan tren ini untuk bisnis?
2. SKENARIO TERBAIK & TERBURUK: Berdasarkan confidence interval, apa implikasinya?
3. TRIGGER KEPUTUSAN: Kapan manajemen harus intervensi jika tren ini berlanjut?
4. PELUANG: Apakah ada window peluang yang terlihat dari prediksi ini?
5. REKOMENDASI ALOKASI SUMBER DAYA: Bagaimana menyesuaikan anggaran/sumber daya berdasarkan prediksi ini?

Format: briefing strategis untuk CFO/direktur, konkret dan berbasis angka."""

                with st.spinner("Gemini menginterpretasi hasil forecast..."):
                    fc_text = _call_gemini(fc_prompt, _COMMANDER_SYSTEM, api_key)
                    st.session_state['forecast_ai'] = fc_text

        if 'forecast_ai' in st.session_state:
            st.markdown(f"""
            <div style='background:#080810; border:1px solid #7986cb; border-radius:12px;
                        padding:18px; margin-top:12px'>
            <b style='color:#9fa8da; font-size:1.05em'>Strategic Forecast Intelligence — Gemini AI</b><br><br>
            <span style='color:#e0e0e0; white-space:pre-wrap; line-height:1.7'>{st.session_state['forecast_ai']}</span>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MAIN APP UI
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div style='text-align:center; padding:20px 0 10px 0'>
  <h1 style='font-size:2.4em; font-weight:800; background:linear-gradient(135deg,#00b4d8,#0077b6,#00e5ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent'>
    🛸 DATA PILOT PRO
  </h1>
  <p style='color:#90caf9; font-size:1.05em; margin:0'>Executive Analytics Engine · 40+ KPI · AI Vision OCR · Anomaly Detection · Advanced Forecast · BI-Ready Export</p>
</div>
""", unsafe_allow_html=True)

# ── TAB NAVIGASI UTAMA ──
# ── TAB NAVIGASI UTAMA ──
# ══════════════════════════════════════════════════════════════════════════
#  WAR ROOM — AI MULTI-AGENT COLLABORATIVE TASK FORCE
# ══════════════════════════════════════════════════════════════════════════

# Profil masing-masing agen
_AGENTS = {
    "logistic": {
        "name":  "Staf Ahli Logistik & Supply Chain",
        "emoji": "🚢",
        "color": "#00e5b4",
        "bg":    "#0a1f1a",
        "border":"#00bfa5",
        "role":  (
            "Kamu adalah Staf Ahli Logistik & Supply Chain kelas dunia dengan 20 tahun pengalaman "
            "di bidang operasi logistik multinasional, manajemen inventori, dan optimasi rantai pasok. "
            "Fokusmu HANYA pada: lead time, fill rate, OTIF, inventory turnover, safety stock, "
            "supplier performance, distribusi, dan efisiensi pengiriman. "
            "Gaya bicara: tajam, berbasis data, menggunakan terminologi supply chain profesional. "
            "Selalu awali responsmu dengan '🚢 [LOGISTIC AGENT]:' "
            "Berikan analisis 3–5 poin terstruktur. Tutup dengan 1 temuan kritis paling mendesak."
        ),
    },
    "financial": {
        "name":  "Direktur Finansial & Risiko",
        "emoji": "💰",
        "color": "#ffd54f",
        "bg":    "#1a1500",
        "border":"#f9a825",
        "role":  (
            "Kamu adalah Direktur Finansial & Manajemen Risiko berpengalaman 25 tahun di perusahaan "
            "Fortune 500. Fokusmu HANYA pada: ROI, profit margin, cashflow, EBITDA, CAC, CLV, "
            "cost efficiency, financial risk, dan alokasi modal. "
            "Gaya bicara: presisi angka, konservatif, berbasis rasio keuangan internasional. "
            "Selalu awali responsmu dengan '💰 [FINANCIAL AGENT]:' "
            "Berikan analisis dari perspektif keuangan dengan angka spesifik dari data. "
            "Identifikasi minimal 1 risiko finansial tersembunyi dan 1 peluang penghematan."
        ),
    },
    "commander": {
        "name":  "Panglima Operasional — Chief Commander",
        "emoji": "⭐",
        "color": "#ff6b35",
        "bg":    "#1a0d07",
        "border":"#ff4500",
        "role":  (
            "Kamu adalah Chief Commander — pemimpin operasional tertinggi yang mengintegrasikan "
            "seluruh perspektif strategis menjadi INSTRUKSI TAKTIS yang bisa langsung dieksekusi. "
            "Kamu menerima laporan dari Logistic Agent dan Financial Agent, lalu menyintesisnya. "
            "Gaya bicara: commanding, decisive, seperti briefing militer tingkat tinggi. "
            "Selalu awali responsmu dengan '⭐ [CHIEF COMMANDER]:' "
            "Format output WAJIB:\n"
            "▌ ASESMEN SITUASI (2 kalimat ringkasan)\n"
            "▌ PERINTAH EKSEKUSI (3 poin aksi konkret dengan PIC dan deadline)\n"
            "▌ STATUS OPERASI: [KRITIS / WASPADA / AMAN / EKSPANSI]\n"
            "Gunakan bahasa Indonesia yang tegas, profesional, dan impresif."
        ),
    },
}

# Token estimator sederhana (bisa diganti tiktoken jika tersedia)
def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def _war_room_call(agent_key: str, user_query: str,
                   data_context: str, prev_reports: str,
                   api_key: str) -> str:
    """
    Panggil satu agen War Room.
    Menggunakan arsitektur _call_gemini() yang sudah ada di app ini
    — tidak bergantung pada _GEMINI_TEXT_MODELS atau _gemini_try_models.
    """
    if not _GEMINI_OK:
        return "❌ `google-generativeai` belum terinstall. Jalankan: pip install google-generativeai"
    if not api_key or not api_key.strip():
        return "❌ Gemini API Key belum diset di sidebar."

    agent = _AGENTS[agent_key]

    # Bangun prompt sesuai peran agen
    if agent_key == "commander":
        content = (
            f"LAPORAN MASUK DARI TIM AHLI:\n{prev_reports}\n\n"
            f"KONTEKS DATA:\n{data_context}\n\n"
            f"MISI DARI OPERATOR:\n{user_query}\n\n"
            "Sintesakan laporan di atas dan keluarkan COMMAND ORDER yang jelas dan actionable."
        )
    else:
        content = (
            f"KONTEKS DATA YANG TERSEDIA:\n{data_context}\n\n"
            f"MISI DARI OPERATOR:\n{user_query}\n\n"
            "Analisis dari sudut pandang keahlianmu dan berikan laporan terstruktur."
        )

    try:
        genai.configure(api_key=api_key.strip())
        model_name = st.session_state.get("gemini_model", "gemini-2.0-flash")
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=agent["role"],
        )
        return model.generate_content(content).text
    except Exception as e:
        err = str(e)
        if "API_KEY_INVALID" in err or "API key not valid" in err:
            return "❌ API Key tidak valid. Cek kembali di Google AI Studio."
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            return (
                "⚠️ API Limit tercapai.\n"
                "Kemungkinan penyebab:\n"
                "- Rate limit per menit (tunggu ~1 menit)\n"
                "- Kuota harian habis (reset jam 07:00 WIB)\n"
                "Coba lagi sebentar, atau ganti API Key di sidebar."
            )
        return f"❌ Error [{agent['name']}]: {err[:400]}"


def render_war_room(df):
    """Render tab Command Control War Room — AI Multi-Agent Task Force."""

    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d0a1f,#1a0d2e,#0d1520);
                border:1px solid #7c4dff55; border-radius:14px;
                padding:20px 24px; margin-bottom:20px'>
      <div style='font-size:1.6em; font-weight:800; color:#e040fb; letter-spacing:1px'>
        ⚔️ COMMAND CONTROL WAR ROOM
      </div>
      <div style='color:#ce93d8; font-size:0.9em; margin-top:6px; line-height:1.6'>
        <b>AI Multi-Agent Collaborative Task Force</b> — Tiga agen AI spesialis bekerja
        secara sekuensial dan menghasilkan <b>Command Order</b> siap eksekusi.<br>
        Klik Misi Cepat → langsung diproses otomatis. Upload berkas untuk konteks lebih kaya.
      </div>
      <div style='margin-top:14px; display:flex; gap:10px; flex-wrap:wrap'>
        <span style='background:#0a1f1a;border:1px solid #00bfa5;border-radius:20px;
                     padding:4px 14px;font-size:0.82em;color:#00e5b4'>🚢 Logistic Agent</span>
        <span style='background:#1a1500;border:1px solid #f9a825;border-radius:20px;
                     padding:4px 14px;font-size:0.82em;color:#ffd54f'>💰 Financial Agent</span>
        <span style='background:#1a0d07;border:1px solid #ff4500;border-radius:20px;
                     padding:4px 14px;font-size:0.82em;color:#ff6b35'>⭐ Chief Commander</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── API Key check ──────────────────────────────────────────────────
    api_key = st.session_state.get("gemini_api_key", "")
    if not api_key:
        st.warning("⚠️ Masukkan **Gemini API Key** di sidebar untuk mengaktifkan War Room.")
        st.markdown("""<div style='background:#0d1f30;border:1px solid #1e3a5f;border-radius:8px;
                    padding:12px 16px;font-size:0.88em;color:#90caf9'>
        Dapatkan API Key GRATIS di
        <a href='https://aistudio.google.com/app/apikey' target='_blank'
           style='color:#00e5ff'>aistudio.google.com/app/apikey</a></div>
        """, unsafe_allow_html=True)
        return

    # ── Session state init ─────────────────────────────────────────────
    if "war_room_sessions"    not in st.session_state:
        st.session_state.war_room_sessions    = []
    if "war_room_auto_deploy" not in st.session_state:
        st.session_state.war_room_auto_deploy = False
    if "war_room_extra_ctx"   not in st.session_state:
        st.session_state.war_room_extra_ctx   = ""
    if "war_room_query"       not in st.session_state:
        st.session_state.war_room_query       = ""

    # ══════════════════════════════════════════════════════════════════
    # BAGIAN A — UPLOAD BERKAS TAMBAHAN
    # ══════════════════════════════════════════════════════════════════
    with st.expander("📎 Upload Berkas Tambahan — Perkaya Konteks AI (CSV · Excel · TXT · PDF)",
                     expanded=False):
        st.markdown("""
        <div style='background:#0d1a2e;border-left:3px solid #7c4dff;
                    padding:10px 14px;border-radius:0 8px 8px 0;
                    font-size:0.87em;color:#ce93d8;margin-bottom:12px'>
        Upload laporan, data tambahan, atau catatan strategis yang ingin dianalisis bersama
        data utama. Semua isi berkas akan dibaca oleh ketiga agen sebagai konteks tambahan.
        </div>
        """, unsafe_allow_html=True)

        wr_files = st.file_uploader(
            "Upload berkas (bisa lebih dari satu):",
            type=["csv", "xlsx", "xls", "txt", "pdf"],
            accept_multiple_files=True,
            key="wr_file_uploader",
        )

        if wr_files:
            extra_parts = []
            for uf in wr_files:
                try:
                    ext = uf.name.rsplit(".", 1)[-1].lower()
                    if ext == "csv":
                        dfe = pd.read_csv(uf)
                        extra_parts.append(
                            f"\n[BERKAS CSV: {uf.name}]\n"
                            f"Dimensi: {len(dfe):,} baris x {len(dfe.columns)} kolom\n"
                            f"Kolom: {', '.join(dfe.columns.tolist())}\n"
                            f"Statistik:\n{dfe.describe().to_string()}\n"
                            f"5 baris pertama:\n{dfe.head(5).to_string()}"
                        )
                        st.success(f"✅ {uf.name} — {len(dfe):,} baris · {len(dfe.columns)} kolom")
                    elif ext in ("xlsx", "xls"):
                        dfe = pd.read_excel(uf)
                        extra_parts.append(
                            f"\n[BERKAS EXCEL: {uf.name}]\n"
                            f"Dimensi: {len(dfe):,} baris x {len(dfe.columns)} kolom\n"
                            f"Kolom: {', '.join(dfe.columns.tolist())}\n"
                            f"Statistik:\n{dfe.describe().to_string()}\n"
                            f"5 baris pertama:\n{dfe.head(5).to_string()}"
                        )
                        st.success(f"✅ {uf.name} — {len(dfe):,} baris · {len(dfe.columns)} kolom")
                    elif ext == "txt":
                        raw = uf.read().decode("utf-8", errors="ignore")
                        extra_parts.append(
                            f"\n[BERKAS TXT: {uf.name}]\n"
                            + raw[:3000]
                            + ("…(terpotong)" if len(raw) > 3000 else "")
                        )
                        st.success(f"✅ {uf.name} — {len(raw):,} karakter")
                    elif ext == "pdf":
                        pdf_text = ""
                        try:
                            import pdfplumber
                            with pdfplumber.open(uf) as pdf_obj:
                                pdf_text = "\n".join(
                                    p.extract_text() or "" for p in pdf_obj.pages[:10])
                        except ImportError:
                            try:
                                import PyPDF2
                                reader = PyPDF2.PdfReader(uf)
                                pdf_text = "\n".join(
                                    reader.pages[i].extract_text() or ""
                                    for i in range(min(10, len(reader.pages))))
                            except ImportError:
                                pdf_text = "[PDF: install pdfplumber → pip install pdfplumber]"
                        extra_parts.append(
                            f"\n[BERKAS PDF: {uf.name}]\n"
                            + pdf_text[:3000]
                            + ("…(terpotong)" if len(pdf_text) > 3000 else "")
                        )
                        st.success(f"✅ {uf.name} — {len(pdf_text):,} karakter diekstrak")
                except Exception as ef:
                    st.error(f"❌ Gagal membaca {uf.name}: {ef}")

            if extra_parts:
                st.session_state.war_room_extra_ctx = "\n".join(extra_parts)

        if st.session_state.war_room_extra_ctx:
            n_berkas = st.session_state.war_room_extra_ctx.count("[BERKAS")
            st.markdown(
                f"<div style='font-size:0.84em;color:#7c4dff;margin-top:6px'>"
                f"📎 {n_berkas} berkas siap dikirim ke AI</div>",
                unsafe_allow_html=True
            )
            col_prev, col_clr = st.columns([3, 1])
            with col_prev:
                with st.expander("👁️ Preview konteks berkas", expanded=False):
                    st.text(st.session_state.war_room_extra_ctx[:800]
                            + ("…" if len(st.session_state.war_room_extra_ctx) > 800 else ""))
            with col_clr:
                if st.button("🗑️ Hapus Berkas", key="btn_clr_wr_files", use_container_width=True):
                    st.session_state.war_room_extra_ctx = ""
                    st.rerun()

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # BAGIAN B — BANGUN DATA CONTEXT
    # ══════════════════════════════════════════════════════════════════
    main_ctx  = _build_data_context(df) if df is not None and len(df) > 0 else ""
    extra_ctx = st.session_state.war_room_extra_ctx

    if not main_ctx and not extra_ctx:
        data_ctx = "Belum ada data diupload. Agen akan memberikan analisis umum berdasarkan misi."
    else:
        parts_ctx = []
        if main_ctx:
            parts_ctx.append("=== DATA UTAMA (dari tab Analisis) ===\n" + main_ctx)
        if extra_ctx:
            parts_ctx.append("=== DATA TAMBAHAN (berkas upload War Room) ===" + extra_ctx)
        data_ctx = "\n\n".join(parts_ctx)

    # Status data singkat
    badges = []
    if main_ctx:
        badges.append("✅ Data utama")
    else:
        badges.append("⚠️ Belum ada data utama")
    if extra_ctx:
        badges.append(f"✅ {extra_ctx.count('[BERKAS')} berkas tambahan")
    st.markdown(
        "<div style='font-size:0.83em;color:#78909c;margin-bottom:14px'>"
        + " &nbsp;·&nbsp; ".join(badges) + "</div>",
        unsafe_allow_html=True
    )

    # ══════════════════════════════════════════════════════════════════
    # BAGIAN C — MISI CEPAT (klik = langsung auto-deploy, tanpa ketik lagi)
    # ══════════════════════════════════════════════════════════════════
    PRESETS = [
        ("📊", "Analisis komprehensif data ini untuk briefing direksi besok"),
        ("🔍", "Identifikasi bottleneck paling kritis dan buat rencana perbaikan 30 hari"),
        ("💡", "Evaluasi efisiensi operasional dan berikan rekomendasi penghematan biaya"),
        ("⚠️", "Analisis risiko utama dan strategi mitigasi yang harus segera dieksekusi"),
        ("📈", "Bandingkan performa aktual vs benchmark industri dan tentukan prioritas"),
        ("🔥", "Buat war plan untuk turnaround bisnis dalam kondisi performa saat ini"),
    ]

    st.markdown(
        "<div style='font-weight:700;color:#ce93d8;font-size:0.97em;margin-bottom:10px'>"
        "🎯 Misi Cepat — Klik langsung proses ke Task Force:</div>",
        unsafe_allow_html=True
    )
    pc = st.columns(3)
    for i, (icon, txt) in enumerate(PRESETS):
        with pc[i % 3]:
            label = f"{icon} {txt[:38]}…" if len(txt) > 38 else f"{icon} {txt}"
            if st.button(label, key=f"wr_preset_{i}", use_container_width=True):
                st.session_state["war_room_query"]       = txt
                st.session_state["war_room_auto_deploy"] = True
                st.session_state.pop("war_room_input", None)
                st.rerun()

    st.markdown("---")

    # ── Mode selector debate / sekuensial ──
    mc1, mc2 = st.columns([2, 1])
    with mc1:
        debate_mode = st.toggle(
            "🔁 **Mode Debate — Logistic ↔ Financial Timbal Balik**",
            value=False,
            help="OFF = Sekuensial Cepat (3 panggilan API, hemat kuota)\n"
                 "ON  = Logistic & Financial berdebat bolak-balik, Commander moderasi & sintesis.\n"
                 "Debate jauh lebih kaya perspektif, tapi lebih banyak token."
        )
    with mc2:
        debate_rounds = 1
        if debate_mode:
            debate_rounds = st.slider("Jumlah Putaran Debate:", 1, 3, 2,
                                      help="1 putaran = Logistic bicara → Financial merespons → Commander.\n"
                                           "Makin banyak putaran = analisis makin dalam.")
            est_calls = debate_rounds * 2 + 1
            st.caption(f"Est. panggilan API: **{est_calls}×** | ~{est_calls * 800:,} token")
        else:
            st.caption("Est. panggilan API: **3×** | ~2,400 token")

    # ══════════════════════════════════════════════════════════════════
    # BAGIAN D — INPUT MANUAL + TOMBOL DEPLOY (fallback)
    # ══════════════════════════════════════════════════════════════════
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        mission_query = st.text_area(
            "📋 Atau ketik misi sendiri, lalu klik Deploy:",
            value=st.session_state.get("war_room_query", ""),
            height=80,
            placeholder="Contoh: Apa penyebab ROI turun dan langkah perbaikannya?",
            key="war_room_input",
        )
    with col_btn:
        st.write(""); st.write("")
        manual_deploy = st.button(
            "🚀 Deploy", use_container_width=True,
            type="primary", key="btn_deploy_war"
        )
        if st.button("🗑️ Reset", use_container_width=True, key="btn_reset_war"):
            for _k in ["war_room_sessions", "war_room_query", "war_room_auto_deploy",
                       "war_room_extra_ctx", "war_room_input"]:
                st.session_state.pop(_k, None)
            st.session_state.war_room_sessions    = []
            st.session_state.war_room_extra_ctx   = ""
            st.session_state.war_room_auto_deploy = False
            st.rerun()

    # Tentukan apakah perlu deploy dan query apa
    _auto = st.session_state.get("war_room_auto_deploy", False)
    _q    = (st.session_state.get("war_room_query","").strip() if _auto
             else mission_query.strip())
    should_deploy = (manual_deploy or _auto) and bool(_q)
    if _auto:
        st.session_state["war_room_auto_deploy"] = False   # reset flag setelah dikonsumsi

    # Pastikan debate_mode & rounds punya nilai default (saat auto-deploy dari preset)
    if 'debate_mode' not in dir():
        debate_mode = False
    if 'debate_rounds' not in dir():
        debate_rounds = 2

    # ══════════════════════════════════════════════════════════════════
    # BAGIAN E — DEPLOY SEQUENCE (pilih mode: Sekuensial atau Debate)
    # ══════════════════════════════════════════════════════════════════
    if should_deploy:
        st.session_state["war_room_query"] = _q
        query = _q

        rec = {
            "query": query, "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
            "logistic": "", "financial": "", "commander": "", "debate": [],
            "tokens_est": 0, "has_extra": bool(extra_ctx),
        }

        ph_hdr = st.empty()
        ph_log = st.empty()
        ph_fin = st.empty()
        ph_cmd = st.empty()

        ph_hdr.markdown(f"""
        <div style='background:linear-gradient(135deg,#0d0a1f,#1a0d2e);border:1px solid #7c4dff;
                    border-radius:10px;padding:12px 18px;margin-bottom:8px;
                    font-size:0.9em;color:#ce93d8'>
          <b>⚔️ MISI AKTIF:</b> {query}<br>
          <span style='font-size:0.83em;color:#7c4dff'>
            {"📎 " + str(extra_ctx.count("[BERKAS")) + " berkas tambahan dimuat" if extra_ctx else "📂 Hanya data utama"}
            &nbsp;·&nbsp; Mode: {'🔁 Debate (Timbal-Balik)' if debate_mode else '⚡ Sekuensial (Cepat)'}
          </span>
        </div>
        """, unsafe_allow_html=True)

        tok_total = 0

        if debate_mode:
            # ── MODE DEBATE: Logistic ↔ Financial saling balas, Commander moderasi ──
            debate_log: list[dict] = []
            ph_debate  = st.empty()

            def _render_debate(items: list[dict]) -> str:
                html = ""
                for turn in items:
                    role = turn["role"]
                    if role == "logistic":
                        html += (
                            f"<div style='background:#0a1f1a;border-left:4px solid #00e5b4;"
                            f"border-radius:0 10px 10px 0;padding:12px 16px;margin:8px 0'>"
                            f"<div style='color:#00e5b4;font-weight:700;font-size:0.88em'>"
                            f"🚢 LOGISTIC — Putaran {turn['round']}</div>"
                            f"<div style='color:#b2dfdb;font-size:0.87em;white-space:pre-wrap;"
                            f"margin-top:8px;line-height:1.7'>{turn['text']}</div></div>"
                        )
                    elif role == "financial":
                        html += (
                            f"<div style='background:#1a1500;border-left:4px solid #ffd54f;"
                            f"border-radius:0 10px 10px 0;padding:12px 16px;margin:8px 0'>"
                            f"<div style='color:#ffd54f;font-weight:700;font-size:0.88em'>"
                            f"💰 FINANCIAL — Putaran {turn['round']}</div>"
                            f"<div style='color:#fff9c4;font-size:0.87em;white-space:pre-wrap;"
                            f"margin-top:8px;line-height:1.7'>{turn['text']}</div></div>"
                        )
                    elif role == "commander":
                        html += (
                            f"<div style='background:linear-gradient(135deg,#1a0d07,#2d1200);"
                            f"border:2px solid #ff4500;border-radius:10px;"
                            f"padding:14px 18px;margin:10px 0;"
                            f"box-shadow:0 0 16px rgba(255,69,0,0.15)'>"
                            f"<div style='color:#ff6b35;font-weight:800;font-size:0.95em'>"
                            f"⭐ CHIEF COMMANDER — Sintesis Putaran {turn['round']}</div>"
                            f"<div style='color:#ffccbc;font-size:0.87em;white-space:pre-wrap;"
                            f"margin-top:8px;line-height:1.8'>{turn['text']}</div></div>"
                        )
                return f"<div>{html}</div>"

            # Bangun prompt context dengan memori diskusi sebelumnya
            def _build_debate_ctx(debate_items: list[dict]) -> str:
                if not debate_items:
                    return ""
                lines = ["\n\n=== RIWAYAT DISKUSI SEBELUMNYA ==="]
                for t in debate_items:
                    tag = {"logistic": "🚢 Logistic", "financial": "💰 Financial",
                           "commander": "⭐ Commander"}[t["role"]]
                    lines.append(f"\n[Putaran {t['round']} — {tag}]\n{t['text']}")
                return "\n".join(lines)

            # Putaran debat
            for rnd in range(1, debate_rounds + 1):
                prev_ctx = _build_debate_ctx(debate_log)

                # Logistic berbicara
                ph_hdr.markdown(
                    f"<div style='background:#0a1f1a;border:1px solid #00bfa5;border-radius:10px;"
                    f"padding:10px 16px;color:#00e5b4'>🚢 <b>LOGISTIC</b> merespons — Putaran {rnd}/{debate_rounds}…</div>",
                    unsafe_allow_html=True
                )
                log_prompt_extra = (
                    f"{prev_ctx}\n\nBerikan perspektif lanjutan Anda untuk Putaran {rnd}. "
                    "Tanggapi poin finansial sebelumnya jika ada. Tetap fokus pada supply chain & logistik."
                    if rnd > 1 else ""
                )
                log_r = _war_room_call(
                    "logistic", query + log_prompt_extra, data_ctx, prev_ctx, api_key
                )
                debate_log.append({"role": "logistic", "round": rnd, "text": log_r})
                tok_total += _est_tokens(log_r)
                ph_debate.markdown(_render_debate(debate_log), unsafe_allow_html=True)

                # Financial merespons
                ph_hdr.markdown(
                    f"<div style='background:#1a1500;border:1px solid #f9a825;border-radius:10px;"
                    f"padding:10px 16px;color:#ffd54f'>💰 <b>FINANCIAL</b> merespons — Putaran {rnd}/{debate_rounds}…</div>",
                    unsafe_allow_html=True
                )
                prev_ctx2 = _build_debate_ctx(debate_log)
                fin_prompt_extra = (
                    f"\n\nBerikan perspektif finansial Putaran {rnd}. "
                    "Tanggapi temuan logistik di atas dari sudut pandang risiko keuangan dan efisiensi modal."
                )
                fin_r = _war_room_call(
                    "financial", query + fin_prompt_extra, data_ctx, prev_ctx2, api_key
                )
                debate_log.append({"role": "financial", "round": rnd, "text": fin_r})
                tok_total += _est_tokens(fin_r)
                ph_debate.markdown(_render_debate(debate_log), unsafe_allow_html=True)

            # Commander memberi sintesis final setelah semua putaran
            ph_hdr.markdown(
                "<div style='background:#1a0d07;border:1px solid #ff4500;border-radius:10px;"
                "padding:10px 16px;color:#ff6b35'>⭐ <b>CHIEF COMMANDER</b> menerbitkan PERINTAH FINAL…</div>",
                unsafe_allow_html=True
            )
            full_debate_ctx = _build_debate_ctx(debate_log)
            cmd_r = _war_room_call("commander", query, data_ctx, full_debate_ctx, api_key)
            debate_log.append({"role": "commander", "round": debate_rounds, "text": cmd_r})
            tok_total += _est_tokens(cmd_r)
            ph_hdr.empty()
            ph_debate.markdown(_render_debate(debate_log), unsafe_allow_html=True)

            # Simpan ke record
            rec["debate"]    = debate_log
            rec["logistic"]  = "\n\n".join(t["text"] for t in debate_log if t["role"] == "logistic")
            rec["financial"] = "\n\n".join(t["text"] for t in debate_log if t["role"] == "financial")
            rec["commander"] = cmd_r
            rec["tokens_est"] = tok_total

            # Buat teks download debate
            debate_txt_parts = ["╔══════════════════════════════════════════╗",
                                 "║  WAR ROOM — DEBATE MODE TRANSCRIPT      ║",
                                 "╚══════════════════════════════════════════╝",
                                 f"Misi: {query}", f"Waktu: {rec['timestamp']}",
                                 f"Total Putaran: {debate_rounds}", ""]
            for t in debate_log:
                label = {"logistic": "🚢 LOGISTIC", "financial": "💰 FINANCIAL",
                         "commander": "⭐ CHIEF COMMANDER"}[t["role"]]
                debate_txt_parts.append(f"{'═'*48}")
                debate_txt_parts.append(f"{label} — Putaran {t['round']}")
                debate_txt_parts.append("─" * 48)
                debate_txt_parts.append(t["text"])
                debate_txt_parts.append("")
            full_txt = "\n".join(debate_txt_parts)

        else:
            # ── MODE SEKUENSIAL CEPAT (original) ──
            ph_log.markdown("""<div style='background:#0a1f1a;border:1px solid #00bfa5;border-radius:10px;
                        padding:12px 18px;font-size:0.9em;color:#00e5b4'>
              🚢 <b>LOGISTIC AGENT</b> sedang menganalisis… <span style='color:#546e7a'>(mohon tunggu)</span>
            </div>""", unsafe_allow_html=True)

            log_r = _war_room_call("logistic", query, data_ctx, "", api_key)
            rec["logistic"] = log_r
            tok1 = _est_tokens(log_r)
            tok_total += tok1

            ph_log.markdown(f"""
            <div style='background:#0a1f1a;border:1px solid #00bfa5;border-radius:10px;padding:14px 18px;margin:6px 0'>
              <div style='color:#00e5b4;font-weight:700;font-size:0.95em;margin-bottom:8px'>
                🚢 STAF AHLI LOGISTIK & SUPPLY CHAIN
                <span style='color:#37474f;font-weight:400;font-size:0.83em'> — ~{tok1} tokens</span>
              </div>
              <div style='color:#b2dfdb;font-size:0.88em;white-space:pre-wrap;line-height:1.75'>
{log_r}
              </div>
            </div>""", unsafe_allow_html=True)

            ph_fin.markdown("""<div style='background:#1a1500;border:1px solid #f9a825;border-radius:10px;
                        padding:12px 18px;font-size:0.9em;color:#ffd54f'>
              💰 <b>FINANCIAL AGENT</b> sedang menganalisis… <span style='color:#546e7a'>(mohon tunggu)</span>
            </div>""", unsafe_allow_html=True)

            fin_r = _war_room_call("financial", query, data_ctx, log_r, api_key)
            rec["financial"] = fin_r
            tok2 = _est_tokens(fin_r)
            tok_total += tok2

            ph_fin.markdown(f"""
            <div style='background:#1a1500;border:1px solid #f9a825;border-radius:10px;padding:14px 18px;margin:6px 0'>
              <div style='color:#ffd54f;font-weight:700;font-size:0.95em;margin-bottom:8px'>
                💰 DIREKTUR FINANSIAL & RISIKO
                <span style='color:#37474f;font-weight:400;font-size:0.83em'> — ~{tok2} tokens</span>
              </div>
              <div style='color:#fff9c4;font-size:0.88em;white-space:pre-wrap;line-height:1.75'>
{fin_r}
              </div>
            </div>""", unsafe_allow_html=True)

            ph_cmd.markdown("""<div style='background:#1a0d07;border:1px solid #ff4500;border-radius:10px;
                        padding:12px 18px;font-size:0.9em;color:#ff6b35'>
              ⭐ <b>CHIEF COMMANDER</b> menyintesis semua laporan… <span style='color:#546e7a'>(mohon tunggu)</span>
            </div>""", unsafe_allow_html=True)

            combined = (f"=== LAPORAN LOGISTIC AGENT ===\n{log_r}\n\n"
                        f"=== LAPORAN FINANCIAL AGENT ===\n{fin_r}")
            cmd_r = _war_room_call("commander", query, data_ctx, combined, api_key)
            rec["commander"] = cmd_r
            tok3 = _est_tokens(cmd_r)
            tok_total += tok3
            rec["tokens_est"] = tok_total

            ph_hdr.empty()

            ph_cmd.markdown(f"""
            <div style='background:linear-gradient(135deg,#1a0d07,#2d1200);border:2px solid #ff4500;
                        border-radius:12px;padding:16px 20px;margin:6px 0;
                        box-shadow:0 0 24px rgba(255,69,0,0.18)'>
              <div style='color:#ff6b35;font-weight:800;font-size:1.05em;margin-bottom:10px;letter-spacing:0.5px'>
                ⭐ PANGLIMA OPERASIONAL — CHIEF COMMANDER
                <span style='color:#37474f;font-weight:400;font-size:0.8em'>
                  — ~{tok3} tokens &nbsp;·&nbsp; Total: ~{rec['tokens_est']} tokens
                </span>
              </div>
              <div style='color:#ffccbc;font-size:0.9em;white-space:pre-wrap;line-height:1.8'>
{cmd_r}
              </div>
            </div>""", unsafe_allow_html=True)

            full_txt = (
                "╔══════════════════════════════════════════╗\n"
                "║  COMMAND CONTROL WAR ROOM — REPORT      ║\n"
                "╚══════════════════════════════════════════╝\n"
                f"Misi      : {query}\n"
                f"Timestamp : {rec['timestamp']}\n"
                f"Est Token : ~{rec['tokens_est']}\n"
                f"Data Extra: {'Ya' if rec['has_extra'] else 'Tidak'}\n\n"
                f"{'═'*48}\n🚢 LOGISTIC AGENT\n{'─'*48}\n{log_r}\n\n"
                f"{'═'*48}\n💰 FINANCIAL AGENT\n{'─'*48}\n{fin_r}\n\n"
                f"{'═'*48}\n⭐ CHIEF COMMANDER — COMMAND ORDER\n{'─'*48}\n{cmd_r}\n"
            )

        # ── Download & simpan ──
        rec["tokens_est"] = tok_total
        st.session_state.war_room_sessions.insert(0, rec)
        st.download_button(
            "📥 Download Command Order (.txt)",
            data=full_txt.encode("utf-8"),
            file_name=f"WarRoom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_cmd_{datetime.now().strftime('%H%M%S')}",
        )

    # ══════════════════════════════════════════════════════════════════
    # BAGIAN F — RIWAYAT MISI
    # ══════════════════════════════════════════════════════════════════
    sessions = st.session_state.war_room_sessions
    if sessions:
        st.markdown("---")
        st.markdown(
            f"<div style='font-weight:700;color:#ce93d8;margin-bottom:8px'>"
            f"📂 Riwayat Misi Sesi Ini ({len(sessions)} misi tersimpan)</div>",
            unsafe_allow_html=True
        )
        start_from = 1 if should_deploy else 0
        for idx, sess in enumerate(sessions[start_from:], start=1):
            tag = " 📎" if sess.get("has_extra") else ""
            with st.expander(
                f"[{sess['timestamp']}]{tag}  {sess['query'][:62]}…  |  ~{sess['tokens_est']} tokens",
                expanded=False
            ):
                th = st.tabs(["🚢 Logistic", "💰 Financial", "⭐ Commander"])
                with th[0]:
                    st.markdown(f"<div style='white-space:pre-wrap;color:#b2dfdb;font-size:0.88em;"
                                f"line-height:1.7'>{sess['logistic']}</div>", unsafe_allow_html=True)
                with th[1]:
                    st.markdown(f"<div style='white-space:pre-wrap;color:#fff9c4;font-size:0.88em;"
                                f"line-height:1.7'>{sess['financial']}</div>", unsafe_allow_html=True)
                with th[2]:
                    st.markdown(f"<div style='white-space:pre-wrap;color:#ffccbc;font-size:0.88em;"
                                f"line-height:1.7'>{sess['commander']}</div>", unsafe_allow_html=True)
                dl = (f"WAR ROOM REPORT\nMisi: {sess['query']}\nTimestamp: {sess['timestamp']}\n\n"
                      f"LOGISTIC:\n{sess['logistic']}\n\nFINANCIAL:\n{sess['financial']}\n\n"
                      f"COMMANDER:\n{sess['commander']}")
                st.download_button(
                    "📥 Download Laporan",
                    data=dl.encode("utf-8"),
                    file_name=f"WarRoom_Sesi_{idx}.txt",
                    mime="text/plain",
                    key=f"dl_hist_{idx}_{sess['timestamp'].replace(' ','').replace(',','')}",
                    use_container_width=True,
                )





# ═══════════════════════════════════════════════════════════════
# ── AUTO-INSIGHT NLQ — NATURAL LANGUAGE QUERY ENGINE ──
# ═══════════════════════════════════════════════════════════════

def _nlq_schema_summary(df: pd.DataFrame, df_hist: pd.DataFrame | None = None) -> str:
    """Buat ringkasan skema lengkap untuk kedua DataFrame (current + historical)."""
    def _df_info(d: pd.DataFrame, label: str) -> str:
        num_cols  = d.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols  = d.select_dtypes(include='object').columns.tolist()
        date_cols = d.select_dtypes(include=['datetime']).columns.tolist()
        lines = [
            f"[{label}]",
            f"  Shape  : {len(d):,} baris × {len(d.columns)} kolom",
            f"  Kolom  : {', '.join(d.columns.tolist()[:20])}",
            f"  Numerik: {', '.join(num_cols[:12])}",
            f"  Kategori:{', '.join(cat_cols[:8])}",
        ]
        if date_cols:
            lines.append(f"  Tanggal: {', '.join(date_cols[:4])}")
        # Nilai unik kolom kategorikal
        for c in cat_cols[:4]:
            uniq = d[c].dropna().unique()[:8].tolist()
            lines.append(f"  {c} (unik): {uniq}")
        # Statistik numerik ringkas
        for c in num_cols[:6]:
            s = d[c].dropna()
            lines.append(f"  {c}: min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}")
        lines.append(f"  Sample (3 baris):\n{d.head(3).to_string()}")
        return "\n".join(lines)

    result = _df_info(df, "DATA AKTIF (df)")
    if df_hist is not None:
        result += "\n\n" + _df_info(df_hist, "DATA HISTORIS (df_hist)")
    return result


def _nlq_clean_code(raw: str) -> str:
    """
    Bersihkan output AI menjadi kode Python murni yang bisa dieksekusi.
    Tangani semua variasi markdown fence yang AI kembalikan.
    """
    if not raw:
        return ""

    # 1. Ambil blok kode di dalam ```...``` (prioritas utama)
    fence_match = re.search(
        r'```(?:python|py|Python)?\s*\n([\s\S]+?)\n\s*```',
        raw, re.IGNORECASE
    )
    if fence_match:
        code = fence_match.group(1)
    else:
        # 2. Hapus fence yang tidak lengkap di awal/akhir
        code = re.sub(r'^```(?:python|py)?\s*', '', raw.strip(), flags=re.IGNORECASE)
        code = re.sub(r'\s*```\s*$', '', code)

    # 3. Hapus baris yang merupakan penjelasan teks (bukan kode Python)
    #    Tandai baris yang dimulai dengan kata benda/teks bukan kode
    cleaned_lines = []
    in_multiline_str = False
    for line in code.splitlines():
        stripped = line.strip()
        # Skip baris kosong tapi pertahankan (penting untuk indentasi)
        if not stripped:
            cleaned_lines.append(line)
            continue
        # Deteksi baris yang jelas bukan Python (penjelasan teks Gemini)
        is_explanation = (
            (stripped.startswith('Berikut') or
             stripped.startswith('Kode') or
             stripped.startswith('Ini adalah') or
             stripped.startswith('Output') and ':' not in stripped and '=' not in stripped or
             stripped.startswith('Catatan:') or
             stripped.startswith('Note:'))
            and not stripped.startswith('#')
            and '=' not in stripped
            and ':' not in stripped
        )
        if is_explanation:
            cleaned_lines.append(f"# {stripped}")  # jadikan komentar, jangan buang
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _nlq_generate_code(question: str, schema: str, api_key: str) -> str:
    """Minta Gemini menulis kode Pandas. Return kode Python bersih."""
    model_name = st.session_state.get('gemini_model', 'gemini-2.0-flash')

    # Prompt yang sangat eksplisit tentang format output
    system = """Kamu adalah Python/Pandas expert. Tulis HANYA kode Python — tidak ada teks lain.

ATURAN OUTPUT (WAJIB DIPATUHI):
1. Kembalikan HANYA kode Python murni — tanpa markdown, tanpa penjelasan, tanpa komentar panjang
2. Baris pertama harus kode Python valid (import, variabel, dll)
3. DataFrame tersedia: `df` (data aktif), `df_hist` (historis, bisa None)
4. Hasil WAJIB disimpan ke variabel `result`
5. Untuk chart: buat `fig` menggunakan plotly (px atau go), dan isi juga `result`
6. Gunakan try/except untuk operasi yang mungkin gagal
7. Jangan import library baru — hanya pd, np, px, go yang tersedia
8. Angka ribuan: tulis 1000000 bukan 1.000.000 atau 1,000,000"""

    prompt = (
        f"SKEMA:\n{schema[:2000]}\n\n"
        f"PERTANYAAN: {question}\n\n"
        "Tulis kode Python. Simpan hasil ke `result`. "
        "Buat `fig` plotly jika visualisasi diperlukan."
    )

    try:
        genai.configure(api_key=api_key.strip())
        model = genai.GenerativeModel(model_name=model_name)
        resp  = model.generate_content(system + "\n\n" + prompt)
        raw   = resp.text.strip() if resp.text else ""
        return _nlq_clean_code(raw)
    except Exception as e:
        return f"result = 'ERROR: {str(e)[:200]}'"


def _nlq_generate_narrative(question: str, result_str: str,
                             code: str, api_key: str) -> str:
    """Minta Gemini membuat narasi bisnis dari hasil kode."""
    model_name = st.session_state.get('gemini_model', 'gemini-2.0-flash')
    prompt = (
        f"Pertanyaan bisnis: {question}\n\n"
        f"Hasil analisis data:\n{result_str[:3000]}\n\n"
        "Tulis insight bisnis profesional dalam Bahasa Indonesia (3–5 paragraf). "
        "Struktur: Temuan Utama → Perbandingan/Tren → Risiko/Peluang → Rekomendasi Aksi. "
        "Gunakan angka spesifik dari hasil. Nada: eksekutif senior, tegas, actionable."
    )
    try:
        genai.configure(api_key=api_key.strip())
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=(
                "Kamu adalah Chief Data Officer yang menulis ringkasan eksekutif "
                "berbasis data untuk C-Level. Selalu gunakan angka spesifik, "
                "hindari generalisasi kosong."
            )
        )
        return model.generate_content(prompt).text
    except Exception as e:
        return f"❌ Error narasi: {e}"


def _nlq_execute(code: str, df: pd.DataFrame,
                 df_hist: pd.DataFrame | None) -> tuple[any, str, str]:
    """
    Eksekusi kode Pandas dengan aman.
    1. Compile dulu (tangkap SyntaxError sebelum exec)
    2. Eksekusi dalam namespace terisolasi
    Return: (result_value, result_str, error_str)
    """
    if not code or not code.strip():
        return None, "", "Kode kosong — Gemini tidak menghasilkan kode."

    if code.strip().startswith("result = 'ERROR:"):
        return None, "", code.strip()[len("result = '"):-1]

    # Namespace aman
    safe_ns: dict = {
        "pd": pd, "np": np,
        "px": px, "go": go,
        "df": df.copy(),
        "df_hist": df_hist.copy() if df_hist is not None else None,
        "datetime": datetime,
        "re": re,
        "result": None,
        "fig": None,
        "__builtins__": {
            # Batasi builtins: hanya yang aman untuk analisis data
            "range": range, "len": len, "list": list, "dict": dict,
            "tuple": tuple, "set": set, "str": str, "int": int,
            "float": float, "bool": bool, "print": print,
            "enumerate": enumerate, "zip": zip, "map": map,
            "filter": filter, "sorted": sorted, "reversed": reversed,
            "min": min, "max": max, "sum": sum, "abs": abs,
            "round": round, "isinstance": isinstance, "type": type,
            "hasattr": hasattr, "getattr": getattr,
            "True": True, "False": False, "None": None,
        }
    }

    try:
        # Step 1: Compile dulu — tangkap SyntaxError sebelum exec
        compiled = compile(code, "<nlq_generated>", "exec")
    except SyntaxError as syn_err:
        err_msg = (
            f"SyntaxError baris {syn_err.lineno}: {syn_err.msg}\n"
            f"Baris bermasalah: {syn_err.text or ''}"
        )
        return None, "", err_msg

    try:
        # Step 2: Eksekusi kode yang sudah tercompile
        exec(compiled, safe_ns)  # noqa: S102
        result_val = safe_ns.get("result")

        # Serialisasi hasil ke string untuk narasi
        if isinstance(result_val, pd.DataFrame):
            result_str = result_val.to_string(max_rows=30, max_cols=12)
        elif isinstance(result_val, pd.Series):
            result_str = result_val.to_string(max_entries=30)
        elif isinstance(result_val, dict):
            result_str = "\n".join(f"  {k}: {v}" for k, v in list(result_val.items())[:20])
        elif result_val is None:
            result_str = "(tidak ada hasil eksplisit)"
        else:
            result_str = str(result_val)[:3000]

        return result_val, result_str, ""

    except Exception as exc:
        return None, "", f"{type(exc).__name__}: {exc}"


def render_auto_insight():
    """Tab Auto-Insight: Natural Language Querying + Perbandingan Historis."""

    st.markdown("""
    <div style='background:linear-gradient(135deg,#0a0f2e,#111840); border:1px solid #3f51b5aa;
                border-radius:14px; padding:18px 22px; margin-bottom:18px'>
      <div style='font-size:1.4em; font-weight:800; color:#7986cb; letter-spacing:0.5px'>
        🔍 AUTO-INSIGHT — Natural Language Query Engine
      </div>
      <div style='color:#9fa8da; font-size:0.9em; margin-top:8px; line-height:1.6'>
        Tanya data Anda dalam <b>Bahasa Indonesia biasa</b> — sistem akan otomatis menulis kode
        Pandas, mengeksekusi, dan menyajikan jawaban + narasi bisnis.<br>
        Unggah <b>data historis</b> untuk perbandingan aktual vs masa lalu secara langsung.
      </div>
    </div>
    """, unsafe_allow_html=True)

    api_key = st.session_state.get('gemini_api_key', '')
    if not api_key:
        st.warning("⚠️ Masukkan **Gemini API Key** di sidebar untuk mengaktifkan Auto-Insight.")
        return

    df_active = st.session_state.get('calc_df', st.session_state.get('df_clean', None))
    if df_active is None:
        st.info("📂 Upload & bersihkan data di tab **Analisis & Kalkulasi** terlebih dahulu.")
        return

    # ── SECTION 1: Data Panel ─────────────────────────────────────────
    st.markdown("### 📂 Panel Data")
    col_cur, col_hist = st.columns(2)

    with col_cur:
        st.markdown(f"""
        <div style='background:#0d1f35;border:1px solid #1e3a5f;border-radius:10px;padding:12px 16px'>
          <div style='color:#4fc3f7;font-weight:700'>📊 Data Aktif (df)</div>
          <div style='color:#90caf9;font-size:0.85em;margin-top:6px'>
            {len(df_active):,} baris × {len(df_active.columns)} kolom<br>
            Kolom: {', '.join(df_active.columns[:8].tolist())}{'…' if len(df_active.columns) > 8 else ''}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_hist:
        hist_upload = st.file_uploader(
            "📁 Upload Data Historis (CSV/Excel) — opsional:",
            type=["csv", "xlsx"], key="nlq_hist_upload"
        )
        df_hist = None
        if hist_upload:
            try:
                df_hist = (pd.read_csv(hist_upload)
                           if hist_upload.name.endswith('.csv')
                           else pd.read_excel(hist_upload))
                st.success(f"✅ Data historis: {len(df_hist):,} baris × {len(df_hist.columns)} kolom")
                st.session_state['nlq_df_hist'] = df_hist
            except Exception as e:
                st.error(f"Gagal baca data historis: {e}")
        elif 'nlq_df_hist' in st.session_state:
            df_hist = st.session_state['nlq_df_hist']
            st.caption(f"📎 Menggunakan data historis dari sesi sebelumnya "
                       f"({len(df_hist):,} baris)")

    st.markdown("---")

    # ── SECTION 2: Query Input ────────────────────────────────────────
    st.markdown("### 💬 Tanya Data Anda")

    # ── Init state NLQ ──────────────────────────────────────────────
    if "nlq_query_text" not in st.session_state:
        st.session_state.nlq_query_text   = ""
    if "nlq_auto_run"   not in st.session_state:
        st.session_state.nlq_auto_run     = False

    # Contoh pertanyaan — klik langsung isi DAN jalankan analisis
    EXAMPLES = [
        "Berapa total penjualan per kategori? Urutkan dari terbesar",
        "Tampilkan tren nilai rata-rata per bulan jika ada kolom tanggal",
        "Kolom mana yang paling berkorelasi dengan hasil formula?",
        "Bandingkan rata-rata nilai antara data aktif dan historis per kategori",
        "Tampilkan 10 baris dengan nilai tertinggi di kolom numerik utama",
        "Buat distribusi (histogram) kolom numerik yang paling variatif",
        "Identifikasi outlier: baris mana yang nilainya > 2 standar deviasi dari mean?",
        "Apa perbedaan performa Q1 vs Q2 jika data memiliki kolom periode?",
    ]

    st.markdown(
        "<div style='font-weight:700;color:#90caf9;margin-bottom:10px'>"
        "💡 Contoh pertanyaan — klik untuk isi & langsung analisis:</div>",
        unsafe_allow_html=True
    )
    ex_cols = st.columns(4)
    for i, ex in enumerate(EXAMPLES):
        with ex_cols[i % 4]:
            label = (ex[:30] + "…") if len(ex) > 30 else ex
            if st.button(label, key=f"nlq_ex_{i}", use_container_width=True, help=ex):
                # Set query text + flag auto-run, hapus widget cache lama, rerun
                st.session_state.nlq_query_text = ex
                st.session_state.nlq_auto_run   = True
                st.session_state.pop("nlq_user_q", None)   # hapus cache widget
                st.rerun()

    # Text area — baca dari state (bukan pop, agar tidak hilang di run yang sama)
    current_q = st.session_state.nlq_query_text
    user_q = st.text_area(
        "Pertanyaan Anda (Bahasa Indonesia / English):",
        value=current_q,
        height=80,
        placeholder="Contoh: Tampilkan tren penjualan per bulan dibandingkan tahun lalu",
        key="nlq_user_q"
    )
    # Sinkronkan jika user mengetik manual
    if user_q != current_q:
        st.session_state.nlq_query_text = user_q
        st.session_state.nlq_auto_run   = False

    show_code = st.checkbox("🔧 Tampilkan kode Pandas yang digenerate (untuk debugging)", value=False)

    col_run, col_clr = st.columns([4, 1])
    with col_run:
        run_btn = st.button(
            "🚀 Analisis Sekarang", type="primary", use_container_width=True,
            disabled=not st.session_state.nlq_query_text.strip()
        )
    with col_clr:
        if st.button("🗑️ Reset", use_container_width=True, key="nlq_reset_btn"):
            st.session_state.nlq_query_text = ""
            st.session_state.nlq_auto_run   = False
            st.session_state.pop("nlq_user_q", None)
            st.rerun()

    # Tentukan apakah perlu run: dari tombol manual atau flag auto-run (klik contoh)
    _nlq_auto = st.session_state.get("nlq_auto_run", False)
    _nlq_q    = st.session_state.nlq_query_text.strip()
    should_run = (run_btn or _nlq_auto) and bool(_nlq_q)

    # Reset flag setelah dikonsumsi (sebelum eksekusi, cegah loop)
    if _nlq_auto:
        st.session_state.nlq_auto_run = False

    # ── SECTION 3: Execute & Display ─────────────────────────────────
    if should_run and _nlq_q:
        q = _nlq_q   # alias agar sisa kode di bawah tetap bisa pakai `q`

        # Init riwayat
        if 'nlq_history' not in st.session_state:
            st.session_state.nlq_history = []

        schema = _nlq_schema_summary(df_active, df_hist)

        # Langkah 1: Generate kode
        with st.spinner("🤖 Gemini menulis kode Pandas…"):
            code = _nlq_generate_code(q, schema, api_key)

        if show_code:
            with st.expander("📋 Kode yang Digenerate Gemini", expanded=True):
                st.code(code, language="python")

        # Langkah 2: Eksekusi kode
        with st.spinner("⚙️ Mengeksekusi analisis…"):
            result_val, result_str, err = _nlq_execute(code, df_active, df_hist)

        # Langkah 3: Tampilkan hasil atau auto-fix
        if err:
            # Tampilkan kode & error untuk transparansi
            with st.expander("⚠️ Detail Error (klik untuk lihat)", expanded=True):
                st.error(f"**Error:** `{err}`")
                if show_code:
                    st.code(code, language="python")

            # Auto-fix: kirim kode ASLI + error spesifik ke Gemini
            st.info("🔄 Auto-fixing kode…")
            fix_system = """Kamu adalah Python debugger. Perbaiki kode yang error.
OUTPUT: HANYA kode Python yang sudah diperbaiki — tanpa penjelasan apapun, tanpa markdown fence.
Baris pertama harus kode Python valid."""

            fix_prompt = (
                f"KODE YANG ERROR:\n{code}\n\n"
                f"PESAN ERROR: {err}\n\n"
                f"SKEMA DATA (referensi):\n{schema[:800]}\n\n"
                "Kembalikan HANYA kode yang sudah diperbaiki."
            )

            with st.spinner("🤖 Gemini memperbaiki kode…"):
                try:
                    genai.configure(api_key=api_key.strip())
                    fix_model = genai.GenerativeModel(
                        st.session_state.get('gemini_model', 'gemini-2.0-flash')
                    )
                    fix_resp = fix_model.generate_content(fix_system + "\n\n" + fix_prompt)
                    fixed_code = _nlq_clean_code(fix_resp.text.strip() if fix_resp.text else "")
                except Exception as fix_e:
                    fixed_code = ""
                    st.error(f"❌ Tidak dapat memanggil Gemini untuk auto-fix: {fix_e}")

            if not fixed_code:
                st.error("❌ Auto-fix tidak menghasilkan kode. Coba ubah pertanyaan Anda.")
                st.stop()

            if show_code:
                with st.expander("📋 Kode Setelah Auto-Fix"):
                    st.code(fixed_code, language="python")

            result_val, result_str, err2 = _nlq_execute(fixed_code, df_active, df_hist)
            if err2:
                st.error(
                    f"❌ Auto-fix masih gagal: `{err2}`\n\n"
                    "**Saran:** Coba pertanyaan yang lebih spesifik, atau pastikan "
                    "nama kolom sesuai dengan data."
                )
                if show_code:
                    st.code(fixed_code, language="python")
                st.stop()
            code = fixed_code

        # Tampilkan result
        st.markdown("### 📊 Hasil Analisis")
        result_displayed = False

        # Ambil fig dari namespace — re-eksekusi via helper yang sama
        _, _, _ = _nlq_execute(code, df_active, df_hist)  # warm run sudah di atas
        # Jalankan ulang khusus untuk ambil fig
        ns_fig: dict = {
            "pd": pd, "np": np, "px": px, "go": go,
            "df": df_active.copy(),
            "df_hist": df_hist.copy() if df_hist is not None else None,
            "datetime": datetime, "re": re,
            "result": None, "fig": None,
            "__builtins__": {"range": range, "len": len, "list": list, "dict": dict,
                             "str": str, "int": int, "float": float, "bool": bool,
                             "print": print, "enumerate": enumerate, "zip": zip,
                             "sorted": sorted, "min": min, "max": max, "sum": sum,
                             "abs": abs, "round": round, "isinstance": isinstance,
                             "True": True, "False": False, "None": None}
        }
        try:
            cmp = compile(code, "<fig_run>", "exec")
            exec(cmp, ns_fig)  # noqa: S102
            fig_obj = ns_fig.get("fig")
        except Exception:
            fig_obj = None

        if fig_obj is not None:
            try:
                fig_obj.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,14,26,0.6)",
                )
                st.plotly_chart(fig_obj, use_container_width=True)
                result_displayed = True
            except Exception:
                pass

        if isinstance(result_val, pd.DataFrame) and not result_val.empty:
            st.dataframe(result_val, use_container_width=True, height=min(400, 28 * len(result_val) + 40))
            dl_csv = result_val.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Hasil (CSV)", dl_csv,
                               f"insight_{datetime.now().strftime('%H%M%S')}.csv",
                               "text/csv", use_container_width=False)
            result_displayed = True
        elif isinstance(result_val, (pd.Series, dict, list)):
            if isinstance(result_val, pd.Series):
                df_show = result_val.reset_index()
                df_show.columns = ['Label', 'Nilai']
                st.dataframe(df_show, use_container_width=True)
            else:
                st.json(result_val if isinstance(result_val, (dict, list)) else {})
            result_displayed = True
        elif result_val is not None and not result_displayed:
            st.markdown(f"""
            <div style='background:#0d1f35;border:1px solid #1e3a5f;border-radius:10px;
                        padding:14px 18px;font-family:monospace;color:#e0f7fa;font-size:0.95em'>
            {str(result_val)}
            </div>
            """, unsafe_allow_html=True)

        # Langkah 4: Narasi bisnis
        if result_str and result_str != "(tidak ada hasil)":
            st.markdown("---")
            st.markdown("### 📝 Insight & Narasi Bisnis (AI)")
            with st.spinner("📝 Gemini menyusun insight eksekutif…"):
                narrative = _nlq_generate_narrative(q, result_str, code, api_key)
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#0a0f2e,#111840);
                        border-left:4px solid #7986cb;border-radius:0 12px 12px 0;
                        padding:18px 22px;line-height:1.85;color:#e8eaf6;font-size:0.93em'>
            {narrative.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            # Simpan ke riwayat
            st.session_state.nlq_history.insert(0, {
                "q": q,
                "ts": datetime.now().strftime("%H:%M:%S"),
                "code": code,
                "result": result_str[:500],
                "narrative": narrative,
                "has_hist": df_hist is not None,
            })

            # Download insight
            insight_txt = (
                f"═══ AUTO-INSIGHT REPORT ═══\n"
                f"Pertanyaan : {q}\n"
                f"Waktu      : {datetime.now().strftime('%d %b %Y %H:%M')}\n"
                f"Data Hist  : {'Ya' if df_hist is not None else 'Tidak'}\n\n"
                f"{'─'*50}\nHASIL DATA:\n{result_str}\n\n"
                f"{'─'*50}\nNARASI BISNIS:\n{narrative}\n\n"
                f"{'─'*50}\nKODE PANDAS:\n{code}\n"
            )
            st.download_button(
                "📥 Download Laporan Insight (.txt)",
                insight_txt.encode('utf-8'),
                f"AutoInsight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain", use_container_width=True,
            )

    # ── SECTION 4: Riwayat Query ─────────────────────────────────────
    history = st.session_state.get('nlq_history', [])
    if history:
        st.markdown("---")
        st.markdown(
            f"<div style='color:#7986cb;font-weight:700;margin-bottom:10px'>"
            f"📂 Riwayat Query Sesi Ini ({len(history)} pertanyaan)</div>",
            unsafe_allow_html=True
        )
        for i, h in enumerate(history[:8]):
            tag = " 📎" if h.get("has_hist") else ""
            with st.expander(
                f"[{h['ts']}]{tag}  {h['q'][:60]}…" if len(h['q']) > 60 else f"[{h['ts']}]{tag}  {h['q']}",
                expanded=False
            ):
                ht = st.tabs(["📊 Hasil", "📝 Narasi", "🔧 Kode"])
                with ht[0]:
                    st.text(h['result'])
                with ht[1]:
                    st.markdown(h['narrative'])
                with ht[2]:
                    st.code(h['code'], language="python")

        if st.button("🗑️ Bersihkan Riwayat", key="nlq_clear_hist"):
            st.session_state.nlq_history = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
main_tab1, main_tab2, main_tab5, main_tab3, main_tab4, main_tab6, main_tab7, main_tab8, main_tab9 = st.tabs([
    "📊 Analisis & Kalkulasi",
    "⚔️ Commander's Chat",
    "👁️ AI Vision OCR",
    "📜 Smart Narrative AI",
    "🎯 Predictive War-Gaming",
    "🚨 Anomaly Detection",
    "📈 Advanced Forecast",
    "🏛️ War Room",
    "🔍 Auto-Insight NLQ",
])

# ─── SIDEBAR ───
with st.sidebar:
    # Status lisensi
    tier_color = {"PRO":"#4ade80","ENTERPRISE":"#00e5ff","BASIC":"#ffeb3b","TRIAL":"#ff9800"}.get(lic_info.get("tier","TRIAL"),"#78909c")
    days_txt = lic_info.get("days_left","?")
    days_txt = f"{days_txt} hari" if isinstance(days_txt, int) else str(days_txt)
    st.markdown(f"""
    <div style='padding:12px 14px; background:#0d1f30; border-radius:10px;
                border:1px solid #1e3a5f; margin-bottom:16px; font-size:0.88em'>
      <div style='color:{tier_color}; font-weight:700; font-size:1em'>🛡️ {lic_info.get("tier","?")} — Aktif</div>
      <div style='color:#90caf9; margin-top:4px'><b>Operator:</b> {lic_info.get("owner","?")}</div>
      <div style='color:#78909c; margin-top:2px'><b>Berlaku:</b> {days_txt}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Konfigurasi")

    category = st.selectbox("🗂 Kategori Analisis", list(FORMULA_CATALOG.keys()))
    formulas_in_cat = FORMULA_CATALOG[category]
    formula_name = st.selectbox("📐 Pilih Formula", list(formulas_in_cat.keys()))

    formula_def = formulas_in_cat[formula_name]
    st.markdown(f"""
    <div class='formula-badge'>
    📌 {formula_def['formula_str']}
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📤 Export Settings")
    export_format = st.multiselect("Format Export",
                                    ["CSV (Data Lengkap)", "CSV (Summary BI)", "PDF Laporan", "JSON Metadata"],
                                    default=["CSV (Data Lengkap)", "PDF Laporan"])
    st.divider()

    # ── GEMINI AI SETTINGS ──
    st.markdown("### 🤖 AI Commander Settings")
    st.markdown("""
    <div style='background:#0a1929; border:1px solid #1e3a5f; border-radius:8px;
                padding:10px 12px; margin-bottom:10px; font-size:0.82em; color:#90caf9'>
    Dapatkan API Key <b>GRATIS</b> di:<br>
    <a href='https://aistudio.google.com/app/apikey' target='_blank' style='color:#00e5ff'>
    aistudio.google.com/app/apikey</a>
    </div>
    """, unsafe_allow_html=True)
    gemini_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        value=st.session_state.get('gemini_api_key', ''),
        help="Dapatkan gratis di: aistudio.google.com/app/apikey"
    )
    if gemini_key_input:
        st.session_state['gemini_api_key'] = gemini_key_input
        st.success("✅ API Key tersimpan di sesi ini")

    gemini_model_info = st.selectbox(
        "Model Gemini",
        [
            "gemini-2.5-flash (Gratis, Terbaru & Cepat)",
            "gemini-2.0-flash (Sangat Stabil)",
        ],
        help="Pilih model. Jika salah satu error 404, coba pilihan lain."
    )
    if "2.5" in gemini_model_info:
        st.session_state['gemini_model'] = "gemini-2.5-flash"
    else:
        st.session_state['gemini_model'] = "gemini-2.0-flash"

    st.caption(f"🤖 Model aktif: `{st.session_state.get('gemini_model', '-')}`")
    st.divider()

    st.caption("Data Pilot Pro v3.0 | AI-Powered Edition")
    st.caption("Plotly · Polars · ReportLab · Streamlit · Gemini AI")
    st.divider()

    # ── TOMBOL KELUAR AKUN ──
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a0a0a,#2d0f0f); border:1px solid #5c1a1a;
                border-radius:8px; padding:10px 12px; margin-bottom:8px; font-size:0.82em; color:#ef9a9a'>
    ⚠️ Keluar akun akan menghapus lisensi tersimpan. Siapkan license key sebelum keluar.
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚪 Keluar Akun / Ganti Lisensi", use_container_width=True, type="secondary"):
        _logout_license()

# ─── UPLOAD ───
with main_tab1:
 st.markdown("<div class='section-header'>① Upload Data</div>", unsafe_allow_html=True)
 uploaded_file = st.file_uploader("Upload CSV atau Excel", type=["csv", "xlsx"])

 if uploaded_file:
    # Load data awal
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    # Enforce row limit per tier
    max_rows = lic_info.get("limits", {}).get("max_rows", 500)
    if len(df_raw) > max_rows:
        st.warning(f"⚠️ Tier **{lic_info['tier']}** dibatasi **{max_rows:,}** baris. "
                   f"Data asli: {len(df_raw):,} baris — dipotong otomatis.")
        df_raw = df_raw.head(max_rows)

    if st.button("🧼 Jalankan Smart Cleaning PRO"):
        with st.spinner("Membersihkan data: null-synonym, typo angka, backtick, duplikat, outlier…"):
            df_clean, report = smart_clean(df_raw)
            st.session_state['df_clean'] = df_clean
            st.session_state['cleaning_report'] = report
        st.success(f"✅ Data bersih: {len(df_clean):,} baris × {len(df_clean.columns)} kolom siap dianalisis!")

    if 'cleaning_report' in st.session_state:
        with st.expander("📋 Laporan Detail Smart Cleaning"):
            st.text(render_cleaning_report(st.session_state['cleaning_report']))

# ─── ANALISIS ───
 if 'df_clean' in st.session_state:
    df = st.session_state['df_clean'].copy()

    st.markdown("<div class='section-header'>② Mapping Kolom & Kalkulasi</div>", unsafe_allow_html=True)

    cols_needed = formula_def['cols']
    col_map = {}
    mapping_cols = st.columns(len(cols_needed))
    for i, needed_col in enumerate(cols_needed):
        with mapping_cols[i]:
            # coba auto-match
            auto = next((c for c in df.columns if needed_col.lower() in c.lower()), df.columns[0])
            col_map[needed_col] = st.selectbox(f"→ {needed_col}", df.columns,
                                                index=list(df.columns).index(auto) if auto in df.columns else 0,
                                                key=f"col_{i}")

    if st.button("🚀 Jalankan Kalkulasi", use_container_width=True):
        try:
            with st.spinner("Menghitung formula..."):
                input_data = {needed: force_numeric(df[mapped]) for needed, mapped in col_map.items()}
                result_series = formula_def['fn'](input_data, list(input_data.keys()))

                df['__Result__'] = result_series
                avg_val = result_series.mean()
                median_val = result_series.median()
                min_val = result_series.min()
                max_val = result_series.max()

                st.session_state['calc_df'] = df
                st.session_state['result_col'] = '__Result__'
                st.session_state['avg_val'] = avg_val
                st.session_state['formula_name'] = formula_name
                st.session_state['formula_str'] = formula_def['formula_str']
                st.session_state['category'] = category
                st.session_state['unit'] = formula_def['unit']
                st.session_state['col_map'] = col_map
                st.session_state['summary_stats'] = {
                    "Formula": formula_name,
                    "Rata-rata": f"{avg_val:.4f} {formula_def['unit']}",
                    "Median": f"{median_val:.4f} {formula_def['unit']}",
                    "Min": f"{min_val:.4f} {formula_def['unit']}",
                    "Max": f"{max_val:.4f} {formula_def['unit']}",
                    "Total Baris": len(df),
                    "Kolom Input": ", ".join(col_map.values())
                }

                # ── PROFESSIONAL EXECUTIVE INSIGHT ENGINE ──
                status, _ = status_label(avg_val, formula_def['thresholds'], formula_def['labels'])

                # Hitung statistik tambahan untuk narasi
                std_val = result_series.std()
                cv_pct = (std_val / abs(avg_val) * 100) if avg_val != 0 else 0
                q25 = result_series.quantile(0.25)
                q75 = result_series.quantile(0.75)
                above_avg = (result_series > avg_val).sum()
                pct_above = above_avg / len(result_series) * 100

                # Benchmark referensi industri per formula
                INDUSTRY_BENCHMARKS = {
                    "ROI (Return on Investment)":       {"rendah": "< 10%", "industri": "15–25%", "top": "> 30%"},
                    "Gross Profit Margin":              {"rendah": "< 20%", "industri": "30–50%", "top": "> 60%"},
                    "Net Profit Margin":                {"rendah": "< 5%",  "industri": "8–15%",  "top": "> 20%"},
                    "EBITDA Margin":                    {"rendah": "< 10%", "industri": "15–25%", "top": "> 30%"},
                    "Current Ratio (Likuiditas)":       {"rendah": "< 1.0x","industri": "1.5–2.5x","top":"> 3.0x"},
                    "Debt-to-Equity Ratio":             {"rendah": "> 3.0x","industri": "1.0–2.0x","top":"< 0.5x"},
                    "ROE (Return on Equity)":           {"rendah": "< 10%", "industri": "15–25%", "top": "> 30%"},
                    "ROA (Return on Assets)":           {"rendah": "< 3%",  "industri": "5–10%",  "top": "> 15%"},
                    "Customer Lifetime Value (CLV)":    {"rendah": "< 100", "industri": "300–1000","top":"> 2000"},
                    "CAC (Customer Acquisition Cost)":  {"rendah": "> 500", "industri": "50–200", "top": "< 30"},
                    "Conversion Rate":                  {"rendah": "< 1%",  "industri": "2–5%",   "top": "> 8%"},
                    "Average Order Value (AOV)":        {"rendah": "< 50",  "industri": "100–300","top": "> 500"},
                    "Churn Rate":                       {"rendah": "> 10%", "industri": "3–7%",   "top": "< 1%"},
                    "ROAS (Return on Ad Spend)":        {"rendah": "< 2x",  "industri": "3–5x",   "top": "> 8x"},
                    "Click-Through Rate (CTR)":         {"rendah": "< 0.5%","industri": "1–3%",   "top": "> 5%"},
                    "Market Share":                     {"rendah": "< 5%",  "industri": "10–25%", "top": "> 40%"},
                    "NPS (Net Promoter Score)":         {"rendah": "< 0",   "industri": "30–60",  "top": "> 70"},
                    "Inventory Turnover":               {"rendah": "< 3x",  "industri": "5–10x",  "top": "> 15x"},
                    "Days Inventory Outstanding (DIO)": {"rendah": "> 60 hari","industri":"20–45 hari","top":"< 15 hari"},
                    "Days Sales Outstanding (DSO)":     {"rendah": "> 60 hari","industri":"30–45 hari","top":"< 20 hari"},
                    "Cash Conversion Cycle (CCC)":      {"rendah": "> 60 hari","industri":"15–40 hari","top":"< 0 hari"},
                    "Fill Rate":                        {"rendah": "< 85%", "industri": "92–97%", "top": "> 99%"},
                    "OTIF (On-Time In-Full)":           {"rendah": "< 80%", "industri": "90–95%", "top": "> 98%"},
                    "Perfect Order Rate":               {"rendah": "< 80%", "industri": "90–96%", "top": "> 99%"},
                    "Supply Chain Cost Ratio":          {"rendah": "> 20%", "industri": "8–15%",  "top": "< 5%"},
                    "Supplier Defect Rate":             {"rendah": "> 5%",  "industri": "1–3%",   "top": "< 0.5%"},
                    "EOQ (Economic Order Quantity)":    {"rendah": "N/A",   "industri": "Tergantung demand","top":"Optimal per SKU"},
                    "OEE (Overall Equipment Effectiveness)":{"rendah":"< 50%","industri":"65–75%","top":"> 85%"},
                    "Utilization Rate":                 {"rendah": "< 50%", "industri": "70–85%", "top": "> 90%"},
                    "Defect Rate":                      {"rendah": "> 5%",  "industri": "1–3%",   "top": "< 0.1%"},
                    "First Pass Yield (FPY)":           {"rendah": "< 80%", "industri": "90–95%", "top": "> 99%"},
                    "Labor Productivity":               {"rendah": "< 50",  "industri": "100–200","top": "> 300"},
                    "Capacity Utilization":             {"rendah": "< 50%", "industri": "70–85%", "top": "> 90%"},
                    "MTTR (Mean Time to Repair)":       {"rendah": "> 8 jam","industri":"2–4 jam","top":"< 1 jam"},
                    "MTBF (Mean Time Between Failures)":{"rendah":"< 100 jam","industri":"500–1000 jam","top":"> 2000 jam"},
                    "Cost per Unit":                    {"rendah": "Tinggi","industri":"Tergantung produk","top":"Minimum feasible"},
                    "Employee Turnover Rate":           {"rendah": "> 20%", "industri": "10–15%", "top": "< 5%"},
                    "Revenue per Employee":             {"rendah": "< 50k", "industri": "100–200k","top":"> 400k"},
                    "Absenteeism Rate":                 {"rendah": "> 8%",  "industri": "2–4%",   "top": "< 1%"},
                    "Training ROI":                     {"rendah": "< 0%",  "industri": "50–100%","top": "> 200%"},
                    "Offer Acceptance Rate":            {"rendah": "< 60%", "industri": "75–90%", "top": "> 95%"},
                }

                bmk = INDUSTRY_BENCHMARKS.get(formula_name, {"rendah": "N/A", "industri": "N/A", "top": "N/A"})

                # Narasi kontekstual mendalam per formula × status
                DEEP_CONTEXT = {
                    "ROI (Return on Investment)": {
                        "Sangat Baik": (
                            "ROI di atas 20% mengindikasikan bahwa setiap rupiah yang diinvestasikan menghasilkan "
                            "imbal hasil yang melampaui ekspektasi pasar. Performa ini berada di kuartil atas industri "
                            "dan mencerminkan efisiensi alokasi modal yang sangat baik."
                        ),
                        "Stabil": (
                            "ROI dalam rentang positif namun belum optimal. Bisnis masih menghasilkan return, namun "
                            "terdapat ruang signifikan untuk meningkatkan margin melalui efisiensi biaya atau "
                            "peningkatan harga jual."
                        ),
                        "Rendah": (
                            f"ROI sebesar {avg_val:.2f}% berada jauh di bawah benchmark industri ({bmk['industri']}). "
                            "Kondisi ini mengindikasikan bahwa biaya operasional terlalu tinggi relatif terhadap "
                            "pendapatan yang dihasilkan. Diperlukan review menyeluruh pada struktur cost."
                        ),
                        "Rugi / Negatif": (
                            f"ROI negatif ({avg_val:.2f}%) adalah sinyal kritis — bisnis sedang mengonsumsi modal "
                            "lebih cepat dari kemampuannya menghasilkan pendapatan. Tanpa intervensi segera, "
                            "likuiditas dan keberlangsungan usaha dapat terancam."
                        ),
                    },
                    "Gross Profit Margin": {
                        "Sangat Baik": "Margin kotor di atas 60% menunjukkan pricing power yang kuat dan struktur COGS yang efisien. Bisnis memiliki buffer yang sangat baik untuk menutup biaya overhead dan operasional.",
                        "Normal": f"Gross Margin {avg_val:.2f}% masih dalam rentang industri normal ({bmk['industri']}), namun perlu diwaspadai tren kenaikan COGS akibat inflasi bahan baku atau gangguan supply chain.",
                        "Rendah": f"Gross Margin {avg_val:.2f}% menunjukkan tekanan pada profitabilitas produk. Kemungkinan penyebab: harga jual terlalu rendah, COGS terlalu tinggi, atau product mix yang kurang menguntungkan.",
                        "Kritis": "Gross Margin di bawah 20% menempatkan bisnis pada risiko operasional tinggi. Setiap kenaikan COGS dapat langsung mengakibatkan kerugian bersih.",
                    },
                    "Net Profit Margin": {
                        "Excellent": f"Net Margin {avg_val:.2f}% merupakan pencapaian luar biasa. Menunjukkan kendali penuh atas seluruh lapisan biaya, dari COGS hingga G&A dan financing cost.",
                        "Normal": f"Net Margin {avg_val:.2f}% berada pada rata-rata industri. Peluang peningkatan terletak pada efisiensi biaya tidak langsung (overhead, admin, dan keuangan).",
                        "Tipis": f"Net Margin {avg_val:.2f}% sangat tipis, rentan terhadap perubahan kondisi pasar. Kenaikan suku bunga atau biaya operasional kecil sekalipun dapat mengubah posisi menjadi rugi.",
                        "Rugi": f"Net Margin negatif ({avg_val:.2f}%) menandakan perusahaan belum mampu menutup total biaya dari pendapatan operasional. Evaluasi menyeluruh atas struktur P&L wajib dilakukan segera.",
                    },
                    "Inventory Turnover": {
                        "Agresif": "Inventory Turnover tinggi mencerminkan manajemen stok yang sangat efisien. Namun perlu diperhatikan risiko stockout yang dapat mengganggu fulfillment dan kepuasan pelanggan.",
                        "Normal": f"ITO {avg_val:.2f}x berada di rentang sehat. Pastikan safety stock memadai untuk mengantisipasi lonjakan demand dan keterlambatan pasokan.",
                        "Rendah": f"ITO {avg_val:.2f}x mengindikasikan inventori yang bergerak lambat. Terdapat potensi dead stock, biaya penyimpanan berlebih, dan risiko keusangan produk.",
                        "Lambat / Mati": "ITO sangat rendah adalah alarm merah. Inventori senilai signifikan tertahan dan tidak produktif, menggerus modal kerja dan profitabilitas.",
                    },
                    "OEE (Overall Equipment Effectiveness)": {
                        "World Class": f"OEE {avg_val:.2f}% mencapai standar World Class Manufacturing (WCM). Fasilitas produksi beroperasi mendekati kapasitas optimal dengan waste minimal.",
                        "Normal": f"OEE {avg_val:.2f}% masih berada di bawah target World Class (85%). Analisis komponen availability, performance, dan quality diperlukan untuk mengidentifikasi bottleneck utama.",
                        "Perlu Perbaikan": f"OEE {avg_val:.2f}% menunjukkan inefisiensi signifikan. Downtime tidak terencana, speed loss, atau defect rate perlu ditelusuri melalui analisis Six Big Losses.",
                        "Kritis": "OEE di bawah 50% mengindikasikan masalah sistemik pada operasi produksi. Perlu audit menyeluruh terhadap mesin, SDM, dan proses.",
                    },
                    "OTIF (On-Time In-Full)": {
                        "Excellent": f"OTIF {avg_val:.2f}% adalah pencapaian kelas dunia. Kepuasan pelanggan terjaga, risiko penalti kontrak minimal, dan reputasi brand supply chain sangat kuat.",
                        "Baik": f"OTIF {avg_val:.2f}% menunjukkan performa baik namun masih ada celah. Identifikasi root cause dari {100-avg_val:.1f}% pengiriman yang tidak perfect untuk perbaikan bertarget.",
                        "Rendah": f"OTIF {avg_val:.2f}% berada di bawah ekspektasi pelanggan. Risiko kehilangan kontrak, penalti finansial, dan erosi kepercayaan pelanggan perlu dimitigasi segera.",
                        "Kritis": "OTIF di bawah 80% mengancam hubungan dengan pelanggan utama. Perlu war room khusus supply chain untuk investigasi dan penanganan darurat.",
                    },
                    "Churn Rate": {
                        "Excellent": f"Churn {avg_val:.2f}% sangat rendah — basis pelanggan terjaga dengan sangat baik. Biaya retensi efisien dan CLV dapat dimaksimalkan.",
                        "Baik": f"Churn {avg_val:.2f}% dalam batas normal. Terus perkuat program loyalitas untuk menekan angka ini lebih jauh.",
                        "Perlu Perhatian": f"Churn {avg_val:.2f}% mulai mengkhawatirkan. Untuk setiap pelanggan yang hilang, biaya akuisisi pengganti rata-rata 5–7x lebih mahal dari biaya retensi.",
                        "Berbahaya": f"Churn {avg_val:.2f}% sangat tinggi — bisnis kehilangan pelanggan lebih cepat dari kemampuannya mendapatkan yang baru. Ini adalah ancaman eksistensial jangka menengah.",
                    },
                    "Employee Turnover Rate": {
                        "Excellent": f"Turnover {avg_val:.2f}% sangat rendah — perusahaan berhasil membangun lingkungan kerja yang suportif dan kompetitif. Knowledge retention dan produktivitas terjaga.",
                        "Baik": f"Turnover {avg_val:.2f}% dalam batas sehat. Perhatikan apakah terdapat pola pada departemen atau level jabatan tertentu.",
                        "Normal": f"Turnover {avg_val:.2f}% di rata-rata industri. Setiap turnover karyawan diestimasi menelan biaya 50–200% dari gaji tahunan posisi tersebut.",
                        "Tinggi / Masalah": f"Turnover {avg_val:.2f}% sangat tinggi dan berdampak langsung pada produktivitas, moral tim, dan biaya rekrutmen. Diperlukan audit budaya organisasi dan kompensasi.",
                    },
                }

                # Ambil konteks mendalam
                deep_ctx = DEEP_CONTEXT.get(formula_name, {})
                context_text = deep_ctx.get(status, f"Performa {formula_name} berada pada level {status} dengan nilai rata-rata {avg_val:.4f} {formula_def['unit']}.")

                # Rekomendasi aksi spesifik (3 poin) per formula × status
                ACTION_PLANS = {
                    "ROI (Return on Investment)": {
                        "Sangat Baik": [
                            "Alokasikan minimum 20–30% dari profit untuk reinvestasi pada lini bisnis atau pasar baru.",
                            "Dokumentasikan driver ROI utama dan jadikan template untuk unit bisnis lain.",
                            "Review portfolio investasi secara kuartalan untuk memastikan capital allocation tetap optimal."
                        ],
                        "Stabil": [
                            "Lakukan cost-benefit analysis per lini produk/layanan untuk mengidentifikasi kontributor ROI tertinggi.",
                            "Evaluasi kemungkinan kenaikan harga jual berbasis value proposition — bukan cost-plus semata.",
                            "Negosiasi ulang kontrak dengan vendor utama untuk menekan COGS 5–10%."
                        ],
                        "Rendah": [
                            "Bentuk task force cross-functional untuk audit biaya operasional dalam 30 hari ke depan.",
                            "Terapkan zero-based budgeting untuk periode berikutnya — setiap pengeluaran harus justify-able.",
                            "Identifikasi dan divestasi aset atau lini bisnis yang tidak memberikan return positif."
                        ],
                        "Rugi / Negatif": [
                            "Aktivasi contingency plan keuangan — prioritaskan cash preservation di atas segalanya.",
                            "Susun rencana restrukturisasi biaya dengan target breakeven dalam 90 hari.",
                            "Konsultasikan dengan financial advisor atau pertimbangkan opsi pendanaan bridge untuk stabilisasi."
                        ],
                    },
                    "Net Profit Margin": {
                        "Excellent": [
                            "Pertahankan disiplin biaya yang telah terbukti efektif dan jadikan SOP tertulis.",
                            "Gunakan margin premium untuk memperkuat R&D atau ekspansi pasar.",
                            "Komunikasikan kinerja ini kepada investor/stakeholder sebagai proof of sustainability."
                        ],
                        "Normal": [
                            "Audit biaya G&A dan overhead — biasanya di sini terdapat efisiensi 10–20% yang belum dimanfaatkan.",
                            "Review struktur debt dan renegosiasikan bunga pinjaman jika memungkinkan.",
                            "Pertimbangkan otomasi proses untuk mereduksi labor cost jangka panjang."
                        ],
                        "Tipis": [
                            "Prioritaskan produk/layanan dengan margin tertinggi dalam sales mix.",
                            "Terapkan program efisiensi energi dan infrastruktur untuk menekan fixed cost.",
                            "Review dan eliminasi program atau inisiatif dengan ROI negatif."
                        ],
                        "Rugi": [
                            "Hentikan sementara semua pengeluaran diskresioner (marketing non-esensial, travel, dll).",
                            "Lakukan analisis break-even per segmen bisnis dan fokus hanya pada yang menguntungkan.",
                            "Siapkan laporan kesehatan keuangan lengkap untuk disajikan kepada board dalam 2 minggu."
                        ],
                    },
                    "Inventory Turnover": {
                        "Agresif": [
                            "Monitor fill rate dan stockout rate secara harian untuk mencegah kehilangan penjualan.",
                            "Implementasikan safety stock dinamis berbasis demand variability tiap SKU.",
                            "Perkuat hubungan dengan supplier untuk memperpendek lead time pengisian stok."
                        ],
                        "Normal": [
                            "Segmentasi inventori menggunakan analisis ABC-XYZ untuk optimasi level stok per kategori.",
                            "Negosiasikan consignment arrangement dengan supplier untuk item slow-moving.",
                            "Review forecast accuracy dan perbarui model demand planning setiap kuartal."
                        ],
                        "Rendah": [
                            "Lakukan inventory cleansing — identifikasi dan likuidasi dead stock melalui promosi atau retur ke supplier.",
                            "Terapkan kebijakan minimum order quantity (MOQ) baru berdasarkan actual demand.",
                            "Evaluasi seluruh daftar SKU — pertimbangkan discontinue item dengan turnover di bawah 1x/tahun."
                        ],
                        "Lambat / Mati": [
                            "Deklarasikan item sebagai dead stock dan lakukan write-down atau disposal segera.",
                            "Bekukan pembelian inventori baru sampai level stok turun ke target.",
                            "Review seluruh proses demand planning dan purchasing — kemungkinan ada kegagalan sistemik."
                        ],
                    },
                    "OTIF (On-Time In-Full)": {
                        "Excellent": [
                            "Jadikan OTIF sebagai competitive advantage dalam negosiasi kontrak dengan pelanggan baru.",
                            "Bagikan best practice proses fulfillment ke mitra logistics dan supplier.",
                            "Pertimbangkan service level agreement (SLA) premium dengan harga yang lebih tinggi."
                        ],
                        "Baik": [
                            "Analisis pola kegagalan OTIF berdasarkan rute, supplier, atau kategori produk.",
                            "Implementasikan sistem alerting real-time untuk potensi keterlambatan 48 jam sebelumnya.",
                            "Review kapasitas warehouse dan distribusi pada peak season."
                        ],
                        "Rendah": [
                            "Bentuk crisis team supply chain dengan daily standup selama 30 hari ke depan.",
                            "Audit seluruh supplier dan 3PL untuk identifikasi weakness di rantai pasok.",
                            "Komunikasikan proaktif kepada pelanggan terdampak dan tawarkan kompensasi sebagai goodwill."
                        ],
                        "Kritis": [
                            "Eskalasikan ke level C-Suite — OTIF kritis dapat mengancam kontrak jangka panjang.",
                            "Aktifkan rencana kontingensi: gunakan supplier/carrier alternatif meski dengan biaya lebih tinggi.",
                            "Lakukan customer-by-customer outreach untuk pelanggan strategis dan susun recovery plan tertulis."
                        ],
                    },
                    "OEE (Overall Equipment Effectiveness)": {
                        "World Class": [
                            "Dokumentasikan seluruh praktik preventive maintenance sebagai standar operasi baru.",
                            "Eksplorasi penerapan predictive maintenance berbasis IoT/sensor untuk mencapai near-zero downtime.",
                            "Jadikan pencapaian ini sebagai baseline KPI dan tingkatkan target secara inkremental."
                        ],
                        "Normal": [
                            "Lakukan Six Big Losses analysis untuk mengidentifikasi kontributor OEE terbesar yang dapat diperbaiki.",
                            "Prioritaskan perbaikan pada komponen OEE dengan gap terbesar (Availability, Performance, atau Quality).",
                            "Implementasikan program TPM (Total Productive Maintenance) secara bertahap."
                        ],
                        "Perlu Perbaikan": [
                            "Jadwalkan sesi kaizen intensif dengan tim produksi dan maintenance dalam 2 minggu.",
                            "Review dan perbarui jadwal preventive maintenance — kemungkinan interval terlalu jarang.",
                            "Evaluasi kemampuan dan training operator untuk memastikan machine handling yang benar."
                        ],
                        "Kritis": [
                            "Hentikan lini produksi bermasalah untuk inspeksi dan perbaikan menyeluruh.",
                            "Panggil vendor mesin/OEM untuk asesmen kondisi aset secara menyeluruh.",
                            "Evaluasi feasibilitas capital expenditure untuk penggantian atau upgrade aset kritis."
                        ],
                    },
                    "Churn Rate": {
                        "Excellent": [
                            "Analisis karakteristik pelanggan loyal dan jadikan sebagai ICP (Ideal Customer Profile) untuk akuisisi baru.",
                            "Kembangkan referral program untuk memanfaatkan basis pelanggan setia sebagai channel akuisisi.",
                            "Pertimbangkan upsell/cross-sell program untuk meningkatkan revenue per pelanggan existing."
                        ],
                        "Baik": [
                            "Implementasikan early warning system berdasarkan behavioral signals untuk prediksi churn 30–60 hari ke depan.",
                            "Perkuat program onboarding untuk pelanggan baru — 60% churn terjadi dalam 90 hari pertama.",
                            "Lakukan NPS survey reguler dan tindak lanjuti semua responden Detractor secara personal."
                        ],
                        "Perlu Perhatian": [
                            "Segmentasikan pelanggan berdasarkan nilai dan risiko churn — alokasikan sumber daya retensi secara proporsional.",
                            "Review pricing dan value proposition — churn tinggi sering merupakan sinyal misalignment ekspektasi.",
                            "Bentuk dedicated customer success team untuk 20% pelanggan dengan nilai tertinggi."
                        ],
                        "Berbahaya": [
                            "Lakukan emergency customer research — wawancara mendalam dengan pelanggan yang baru churn.",
                            "Hentikan akuisisi pelanggan baru sementara dan fokuskan seluruh sumber daya pada stabilisasi base.",
                            "Review fundamental: product-market fit, pricing, dan kualitas layanan dari perspektif pelanggan."
                        ],
                    },
                    "Employee Turnover Rate": {
                        "Excellent": [
                            "Jadikan employer branding sebagai competitive advantage dalam perekrutan talent terbaik.",
                            "Dokumentasikan program engagement yang efektif dan replikasi ke seluruh departemen.",
                            "Pertimbangkan program equity atau long-term incentive untuk mempertahankan high performer."
                        ],
                        "Baik": [
                            "Lakukan pulse survey triwulanan untuk mendeteksi perubahan sentimen karyawan secara dini.",
                            "Review kompensasi dan benefi secara annual untuk memastikan tetap kompetitif di pasar.",
                            "Perkuat program pengembangan karir dengan career path yang jelas dan terstruktur."
                        ],
                        "Normal": [
                            "Lakukan exit interview terstruktur dan analisis pola alasan turnover.",
                            "Identifikasi departemen atau manajer dengan turnover di atas rata-rata sebagai prioritas perbaikan.",
                            "Review beban kerja dan work-life balance — burnout adalah driver turnover yang sering diabaikan."
                        ],
                        "Tinggi / Masalah": [
                            "Eskalasikan ke board level — turnover tinggi berdampak langsung pada produktivitas dan biaya yang material.",
                            "Tunjuk Chief People Officer atau konsultan HR eksternal untuk audit budaya organisasi menyeluruh.",
                            "Luncurkan retention package darurat untuk top talent yang berisiko keluar dalam 6 bulan ke depan."
                        ],
                    },
                }

                # Default action plans jika formula belum terdefinisi
                DEFAULT_ACTIONS = {
                    "Sangat Baik": [
                        "Pertahankan strategi yang telah terbukti efektif dan jadikan sebagai standar operasi.",
                        "Eksplorasi peluang skalabilitas untuk mereplikasi keberhasilan ini ke area lain.",
                        "Bangun sistem monitoring untuk memastikan performa ini konsisten dari waktu ke waktu."
                    ],
                    "World Class": [
                        "Dokumentasikan best practice dan jadikan benchmark internal maupun eksternal.",
                        "Investasikan pada inovasi untuk mempertahankan keunggulan kompetitif.",
                        "Bagikan knowledge ke tim lain untuk membangun kapabilitas organisasi secara menyeluruh."
                    ],
                    "Excellent": [
                        "Pertahankan momentum dan tingkatkan target secara bertahap.",
                        "Jadikan hasil ini sebagai leverage dalam negosiasi dengan mitra bisnis dan investor.",
                        "Review apakah terdapat potensi optimasi lebih lanjut yang belum dimanfaatkan."
                    ],
                    "Baik": [
                        "Identifikasi faktor pendorong utama dan perkuat secara terukur.",
                        "Tetapkan target peningkatan 10–15% untuk periode berikutnya dengan milestone yang jelas.",
                        "Benchmarking dengan pemain terbaik di industri untuk mendapatkan insight praktis."
                    ],
                    "Normal": [
                        "Lakukan analisis mendalam untuk membedakan area yang performanya di atas dan di bawah rata-rata.",
                        "Tentukan quick win yang bisa menggerakkan angka ini dalam 30–60 hari ke depan.",
                        "Presentasikan rencana perbaikan kepada manajemen dengan KPI dan timeline yang spesifik."
                    ],
                    "Stabil": [
                        "Optimalkan efisiensi proses untuk meningkatkan output tanpa menambah input.",
                        "Review apakah terdapat biaya atau aktivitas yang tidak memberikan nilai tambah.",
                        "Konsultasikan dengan peers atau asosiasi industri untuk benchmark yang relevan."
                    ],
                    "Rendah": [
                        "Lakukan root cause analysis secara terstruktur (fishbone diagram atau 5-Why) dalam 2 minggu.",
                        "Prioritaskan 2–3 perbaikan dengan impact tertinggi dan execution tercepat.",
                        "Tetapkan KPI recovery dengan monitoring mingguan dan eskalasi jika target tidak tercapai."
                    ],
                    "Perlu Perbaikan": [
                        "Susun improvement plan 90 hari dengan milestones yang terukur dan dapat dipertanggungjawabkan.",
                        "Libatkan tim lintas fungsi untuk menyelesaikan akar masalah secara komprehensif.",
                        "Laporkan progress ke manajemen setiap 2 minggu untuk menjaga akuntabilitas."
                    ],
                    "Kritis": [
                        "Deklarasikan sebagai isu prioritas dan bentuk crisis response team dalam 48 jam.",
                        "Susun rencana stabilisasi jangka pendek (0–30 hari) dan recovery jangka menengah (30–90 hari).",
                        "Siapkan laporan situasi komprehensif untuk disajikan kepada pimpinan dalam minggu ini."
                    ],
                    "Berbahaya": [
                        "Eskalasikan segera ke C-Suite dan board — situasi ini memerlukan keputusan strategis level tertinggi.",
                        "Aktifkan seluruh protokol kontingensi dan alokasikan sumber daya darurat.",
                        "Engage advisor eksternal (konsultan, auditor, atau legal) untuk asesmen situasi independen."
                    ],
                }

                actions = ACTION_PLANS.get(formula_name, {}).get(status, DEFAULT_ACTIONS.get(status, DEFAULT_ACTIONS["Normal"]))

                # Variability assessment
                if cv_pct < 15:
                    variability_note = f"Tingkat konsistensi data sangat tinggi (CV: {cv_pct:.1f}%) — performa stabil dan dapat diprediksi."
                elif cv_pct < 35:
                    variability_note = f"Variabilitas data moderat (CV: {cv_pct:.1f}%) — perlu segmentasi lebih lanjut untuk mengidentifikasi outlier yang mempengaruhi rata-rata."
                else:
                    variability_note = f"Variabilitas data tinggi (CV: {cv_pct:.1f}%) — rata-rata dapat menyesatkan. Disarankan analisis per segmen atau periode untuk gambaran yang lebih akurat."

                # Susun narasi lengkap
                today_str = datetime.now().strftime("%d %B %Y")
                action_lines = "\n".join([f"  {i+1}. {act}" for i, act in enumerate(actions)])

                full_insight = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXECUTIVE ANALYTICAL BRIEF
{formula_name.upper()}
Dipersiapkan: {today_str}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▌ RINGKASAN EKSEKUTIF

Analisis terhadap {len(result_series):,} data poin menunjukkan bahwa {formula_name} saat ini berada pada level {status.upper()} dengan rata-rata {avg_val:.4f} {formula_def['unit']}.

{context_text}

▌ STATISTIK KINERJA

  • Rata-rata (Mean)    : {avg_val:.4f} {formula_def['unit']}
  • Nilai Tengah (Median): {median_val:.4f} {formula_def['unit']}
  • Rentang             : {min_val:.4f} – {max_val:.4f} {formula_def['unit']}
  • Kuartil 25–75%      : {q25:.4f} – {q75:.4f} {formula_def['unit']}
  • Data di atas rata-rata: {above_avg:,} dari {len(result_series):,} ({pct_above:.1f}%)
  • {variability_note}

▌ BENCHMARK INDUSTRI

  • Performa Rendah     : {bmk['rendah']}
  • Rata-rata Industri  : {bmk['industri']}
  • Performa Top Tier   : {bmk['top']}
  • Posisi Saat Ini     : {status} [{avg_val:.4f} {formula_def['unit']}]

▌ RENCANA AKSI STRATEGIS (Prioritas Tinggi)

{action_lines}

▌ PESAN KUNCI UNTUK MANAJEMEN

Hasil analisis ini hendaknya dijadikan dasar diskusi pada rapat manajemen berikutnya. Tim direkomendasikan untuk mengevaluasi rencana aksi di atas terhadap kapasitas dan prioritas bisnis, serta menetapkan penanggung jawab (PIC) dan tenggat waktu yang jelas untuk setiap inisiatif perbaikan.

Dokumen ini dapat langsung digunakan sebagai lampiran dalam Board Meeting Deck atau Executive Summary Report.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Formula : {formula_def['formula_str']}
Kolom Input: {', '.join([f"{k}→{v}" for k, v in col_map.items()])}
Total Data : {len(result_series):,} baris | Generated by Data Pilot Pro
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

                st.session_state['auto_insight'] = full_insight

        except Exception as e:
            st.error(f"❌ Error kalkulasi: {e}")
            st.exception(e)

    # ─── HASIL KALKULASI ───
    if 'calc_df' in st.session_state:
        df_calc = st.session_state['calc_df']
        avg_val = st.session_state['avg_val']
        unit = st.session_state['unit']
        formula_name_s = st.session_state['formula_name']
        summary = st.session_state['summary_stats']

        status, css_class = status_label(avg_val, formula_def['thresholds'], formula_def['labels'])

        st.markdown("<div class='section-header'>③ Hasil Analisis</div>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Rata-rata", f"{avg_val:.4f} {unit}")
        with m2:
            st.metric("Median", f"{df_calc['__Result__'].median():.4f} {unit}")
        with m3:
            st.metric("Max", f"{df_calc['__Result__'].max():.4f} {unit}")
        with m4:
            st.metric("Min", f"{df_calc['__Result__'].min():.4f} {unit}")

        st.markdown(f"""
        <div class='insight-box'>
        <b>Status Performa:</b> <span class='{css_class}'>{status}</span><br>
        <small style='color:#90caf9'>Formula: {formula_def['formula_str']}</small>
        </div>
        """, unsafe_allow_html=True)

        # ─── VISUALISASI ───
        st.markdown("<div class='section-header'>④ Visualisasi Otomatis</div>", unsafe_allow_html=True)

        figs = auto_visualize(df_calc, '__Result__', formula_name_s)
        if figs:
            tab_names = [name for name, _ in figs]
            tabs = st.tabs(tab_names)
            for tab, (_, fig) in zip(tabs, figs):
                with tab:
                    st.plotly_chart(fig, use_container_width=True)

        # ─── INSIGHT EDITOR ───
        st.markdown("<div class='section-header'>⑤ Executive Analytical Brief — Siap Meeting</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:linear-gradient(135deg,#0d2137,#0a1929); border:1px solid #1e4060; border-radius:10px; padding:14px 18px; margin-bottom:14px'>
        <b style='color:#00e5ff'>📋 Tentang Laporan Ini</b><br>
        <span style='color:#90caf9; font-size:0.9em'>Narasi di bawah dihasilkan secara otomatis berdasarkan data Anda, lengkap dengan konteks permasalahan, posisi vs benchmark industri, dan 3 rencana aksi konkret. Anda bisa langsung edit, copy-paste ke PowerPoint, atau export ke PDF.</span>
        </div>
        """, unsafe_allow_html=True)

        insight_text = st.text_area(
            "Edit sebelum digunakan (siap untuk Board Meeting / Laporan CEO):",
            value=st.session_state.get('auto_insight', ''),
            height=420
        )

        ci1, ci2 = st.columns(2)
        with ci1:
            if st.button("📋 Copy ke Clipboard (Ctrl+A lalu Ctrl+C)", use_container_width=True):
                st.info("Klik di dalam text area → Ctrl+A → Ctrl+C untuk copy semua teks.")
        with ci2:
            st.download_button(
                "📝 Download Insight (.txt)",
                data=insight_text.encode('utf-8'),
                file_name=f"Executive_Brief_{formula_name_s.replace(' ','_')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        # ─── EXPORT ───
        st.markdown("<div class='section-header'>⑥ Export Package (Power BI · Tableau · PDF)</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='margin-bottom:8px'>
        <span class='export-tag'>✅ Power BI Ready</span>
        <span class='export-tag'>✅ Tableau Compatible</span>
        <span class='export-tag'>✅ Google Data Studio</span>
        <span class='export-tag'>✅ Excel / Metabase</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📦 Generate Semua Export", use_container_width=True):
            with st.spinner("Menyiapkan semua file export..."):
                os.makedirs('exports', exist_ok=True)

                bi_data = prepare_bi_export(df_calc, formula_name_s, '__Result__')
                report_data = {
                    "category": category,
                    "formula_name": formula_name_s,
                    "formula_str": formula_def['formula_str'],
                    "summary": summary,
                    "insight": insight_text
                }

                # Tulis semua file
                with open("exports/data_lengkap_bi.csv", "w") as f:
                    f.write(bi_data['main_csv'])
                with open("exports/summary_statistik.csv", "w") as f:
                    f.write(bi_data['summary_csv'])
                with open("exports/metadata_tableau.json", "w") as f:
                    f.write(bi_data['meta_json'])
                generate_professional_pdf(report_data, "exports/laporan_eksekutif.pdf")

                st.success("✅ Semua file siap!")

                dl1, dl2, dl3, dl4 = st.columns(4)
                with dl1:
                    with open("exports/data_lengkap_bi.csv", "rb") as f:
                        st.download_button("📊 CSV Data Lengkap", f, "data_lengkap_bi.csv", "text/csv")
                with dl2:
                    with open("exports/summary_statistik.csv", "rb") as f:
                        st.download_button("📋 CSV Summary BI", f, "summary_statistik.csv", "text/csv")
                with dl3:
                    with open("exports/metadata_tableau.json", "rb") as f:
                        st.download_button("🔷 JSON Metadata", f, "metadata_tableau.json", "application/json")
                with dl4:
                    with open("exports/laporan_eksekutif.pdf", "rb") as f:
                        st.download_button("📄 PDF Laporan", f, "Laporan_Eksekutif.pdf", "application/pdf")

                st.info("""
                **📌 Cara Import ke Power BI:** Buka Power BI Desktop → Get Data → Text/CSV → Pilih `data_lengkap_bi.csv`
                
                **📌 Cara Import ke Tableau:** Buka Tableau → Connect → Text File → Pilih `data_lengkap_bi.csv` | Untuk metadata: pakai JSON connector.
                
                **📌 Cara Import ke Google Looker Studio:** Upload `data_lengkap_bi.csv` sebagai Google Sheets, lalu connect ke Looker Studio.
                """)

        # ─── DATA TABLE ───
        with st.expander("🔍 Lihat Data Hasil Kalkulasi Lengkap"):
            st.dataframe(df_calc, use_container_width=True)

 else:
    st.markdown("""
    <div style='text-align:center; padding:60px 20px; color:#546e7a'>
    <div style='font-size:4em'>📂</div>
    <div style='font-size:1.2em; margin-top:12px'>Upload file CSV atau Excel untuk memulai analisis</div>
    <div style='font-size:0.9em; margin-top:6px; color:#37474f'>40+ formula · Financial · Sales · Supply Chain · Operational · HR</div>
    </div>
    """, unsafe_allow_html=True)

# ─── TAB 2: COMMANDER'S CHAT ───
with main_tab2:
    df_for_ai = st.session_state.get('calc_df', st.session_state.get('df_clean', None))
    if df_for_ai is None:
        st.info("⚠️ Upload dan bersihkan data terlebih dahulu di tab **Analisis & Kalkulasi**, lalu kembali ke sini.")
    else:
        render_commander_chat(df_for_ai)

# ─── TAB 3: SMART NARRATIVE AI ───
with main_tab3:
    df_for_narr = st.session_state.get('calc_df', st.session_state.get('df_clean', None))
    if df_for_narr is None:
        st.info("⚠️ Upload dan jalankan kalkulasi formula terlebih dahulu untuk menghasilkan AI Narrative.")
    else:
        fn = st.session_state.get('formula_name', 'Formula')
        av = st.session_state.get('avg_val', 0)
        un = st.session_state.get('unit', '')
        fd = st.session_state.get('formula_str', '')
        status_now, _ = status_label(av, formula_def['thresholds'], formula_def['labels'])
        render_ai_narrative(df_for_narr, fn, av, un, status_now)

# ─── TAB 4: PREDICTIVE WAR-GAMING ───
with main_tab4:
    render_wargaming(
        st.session_state.get('calc_df', pd.DataFrame()),
        st.session_state.get('formula_name', formula_name),
        formula_def
    )

# ─── TAB 5: AI VISION OCR ───
with main_tab5:
    try:
        render_vision_ocr()
    except Exception as _e5:
        st.error(f"❌ Error AI Vision OCR: {_e5}")
        with st.expander("Detail error (klik untuk lihat)"):
            import traceback as _tb5
            st.code(_tb5.format_exc())

# ─── TAB 6: ANOMALY DETECTION ───
with main_tab6:
    try:
        df_for_anomaly = st.session_state.get('calc_df', st.session_state.get('df_clean', None))
        if df_for_anomaly is None:
            st.info("⚠️ Upload dan bersihkan data terlebih dahulu di tab **Analisis & Kalkulasi**.")
        else:
            render_anomaly_detection(df_for_anomaly)
    except Exception as _e6:
        st.error(f"❌ Error Anomaly Detection: {_e6}")
        with st.expander("Detail error (klik untuk lihat)"):
            import traceback as _tb6
            st.code(_tb6.format_exc())

# ─── TAB 7: ADVANCED FORECAST ───
with main_tab7:
    try:
        df_for_fc = st.session_state.get('calc_df', st.session_state.get('df_clean', None))
        if df_for_fc is None:
            st.info("⚠️ Upload dan bersihkan data terlebih dahulu di tab **Analisis & Kalkulasi**.")
        else:
            render_advanced_forecast(df_for_fc)
    except Exception as _e7:
        st.error(f"❌ Error Advanced Forecast: {_e7}")
        with st.expander("Detail error (klik untuk lihat)"):
            import traceback as _tb7
            st.code(_tb7.format_exc())

# ─── TAB 8: COMMAND CONTROL WAR ROOM ───
with main_tab8:
    try:
        df_for_wr = st.session_state.get('calc_df', st.session_state.get('df_clean', None))
        render_war_room(df_for_wr)
    except Exception as _e8:
        st.error(f"❌ Error War Room: {_e8}")
        with st.expander("Detail error (klik untuk lihat)"):
            import traceback as _tb8
            st.code(_tb8.format_exc())

# ─── TAB 9: AUTO-INSIGHT NLQ ───
with main_tab9:
    try:
        render_auto_insight()
    except Exception as _e9:
        st.error(f"❌ Error Auto-Insight: {_e9}")
        with st.expander("Detail error (klik untuk lihat)"):
            import traceback as _tb9
            st.code(_tb9.format_exc())