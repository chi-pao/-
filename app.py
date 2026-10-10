import re
import os
import unicodedata
from functools import lru_cache
import json
from datetime import datetime
from pathlib import Path
import base64
import joblib
import numpy as np
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from market_context import build_market_context, official_location_context
from address_master import get_municipalities, get_towns, make_town_point_map
from feedback_store import classify_feedback, save_feedback, load_feedback, update_review_status, export_approved_training_candidates


# =========================================================
# 基本設定
# =========================================================

st.set_page_config(
    page_title="中古住宅・中古物件・中古マンション・一戸建ての価格相場チェック｜お買い得物件チェッカー",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.html(
    """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-4DZ00D4915"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());

      gtag('config', 'G-4DZ00D4915');
    </script>
    """,
    unsafe_allow_javascript=True,
)

# =========================================================
# 2026年10月・ホームページ刷新（既存の予測処理は維持）
# =========================================================
from quality_guidance import assess_property_cautions

st.markdown(
    """
    <style>
    :root {
      --ink:#10283d; --navy:#0a263c; --teal:#0c857c;
      --gold:#f3bf63; --paper:#f7fafb; --line:#dce7e9;
    }
    .stApp { background:linear-gradient(180deg,#f3f9fa 0%,#ffffff 420px); color:var(--ink); }
    .block-container { max-width:1160px; padding-top:1.1rem; padding-bottom:4rem; }
    h1,h2,h3 { font-weight:780 !important; letter-spacing:-.022em; }
    .site-topbar { display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:13px; }
    .site-brand { font-weight:900;font-size:1.06rem;color:#0a3144;letter-spacing:-.02em; }
    .site-topbar a { color:#0e6f72;text-decoration:none;font-weight:700;font-size:.9rem; }
    .site-topbar a:hover { text-decoration:underline; }
    .hero { position:relative;overflow:hidden;display:flex;align-items:center;gap:1.9rem;
      padding:1.2rem 1.5rem;border-radius:18px;
      background:radial-gradient(circle at 94% 6%,rgba(71,205,190,.26),transparent 34%),
                 linear-gradient(126deg,#092337 0%,#0a3b51 58%,#0b706e 100%);
      color:white;box-shadow:0 16px 38px rgba(5,39,55,.16); }
    .hero:after { content:"";position:absolute;width:260px;height:260px;border:1px solid rgba(255,255,255,.12);
      border-radius:50%;right:-105px;bottom:-165px;pointer-events:none; }
    .hero-logo { width:88px;height:88px;object-fit:cover;border-radius:27px;flex:0 0 auto;
      border:3px solid rgba(255,255,255,.28);box-shadow:0 10px 25px rgba(0,0,0,.18); }
    .hero-copy { position:relative;z-index:1;min-width:0; }
    .hero-badge { display:inline-block;padding:.36rem .75rem;border:1px solid rgba(255,255,255,.35);
      border-radius:999px;background:rgba(255,255,255,.09);font-size:.82rem;letter-spacing:.03em;margin-bottom:.7rem; }
    .hero-title { color:white;font-size:clamp(1.5rem,2.5vw,2.1rem);line-height:1.23;
      font-weight:900;letter-spacing:-.04em;margin:0 0 .75rem; }
    .hero-subtitle { color:#e1f2f1;font-size:1rem;line-height:1.8;max-width:720px; }
    .hero-actions { display:flex;flex-wrap:wrap;gap:10px;margin-top:1.2rem; }
    .hero-action { display:inline-flex;align-items:center;justify-content:center;padding:11px 19px;
      background:#fff;color:#0a4451 !important;border-radius:12px;font-weight:850;text-decoration:none !important;
      box-shadow:0 6px 16px rgba(0,0,0,.08); }
    .hero-action.secondary { color:#f1ffff !important;background:rgba(255,255,255,.11);
      border:1px solid rgba(255,255,255,.33);box-shadow:none; }
    .trust-strip { display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 22px; }
    .trust-chip { flex:1 1 170px;background:#fff;border:1px solid var(--line);border-radius:14px;
      padding:12px 16px;color:#245268;font-weight:700;font-size:.9rem;text-align:center; }
    .steps { display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0 24px; }
    .step { background:white;border:1px solid var(--line);border-radius:18px;padding:1.05rem 1.1rem;
      box-shadow:0 4px 16px rgba(5,39,55,.035);height:100%; }
    .step-no { color:#08796e;font-size:.76rem;font-weight:900;letter-spacing:.08em; }
    .step-title { color:#143b4b;font-size:1rem;font-weight:850;margin:.2rem 0; }
    .step-text { color:#637988;font-size:.85rem;line-height:1.55; }
    .section-title { font-size:1.45rem;font-weight:850;color:#0a3548;margin-top:2rem;margin-bottom:.3rem; }
    .section-description { color:#617783;margin-bottom:1rem;line-height:1.7; }
    .property-card,.result-card,.info-card { border:1px solid var(--line);background:#fff;
      box-shadow:0 6px 18px rgba(15,53,68,.05); }
    .property-card { padding:1.35rem;border-radius:20px;margin:.6rem 0 1rem; }
    .property-title { font-size:1.15rem;font-weight:850;margin-bottom:.7rem; }
    .property-price { font-size:2rem;font-weight:900;margin-bottom:.7rem;color:#0b5d62; }
    .property-meta { color:#475569;font-size:.95rem;line-height:1.8; }
    .result-card { padding:1.6rem;border-radius:22px;margin:1rem 0;
      background:linear-gradient(150deg,#fff 55%,#eff9f7);border-top:4px solid #119387; }
    .result-label { color:#607682;font-size:.9rem;margin-bottom:.3rem;font-weight:650; }
    .result-price { font-size:2.25rem;font-weight:900;color:#0b5d62;letter-spacing:-.025em; }
    .result-note { color:#64748b;font-size:.9rem;margin-top:.3rem; }
    .info-card { padding:1rem 1.1rem;border-radius:14px;margin:.7rem 0; }
    .mini-label { color:#64748b;font-size:.82rem;margin-bottom:.2rem; }
    .stButton > button { border-radius:13px;min-height:48px;font-weight:850;border-width:1px; }
    .stButton > button[kind="primary"] { min-height:57px;font-size:1.06rem;
      background:linear-gradient(110deg,#0a6f72,#0c9687);border-color:#0c807b;
      box-shadow:0 8px 22px rgba(13,128,122,.18); }
    .stButton > button[kind="primary"]:hover { background:#095f63;border-color:#095f63; }
    .stDownloadButton > button { border-radius:13px;min-height:48px;font-weight:800; }
    [data-testid="stExpander"] { border:1px solid var(--line);border-radius:15px;background:#fff;overflow:hidden; }
    [data-testid="stForm"] { border-radius:18px; }
    .footer-feature { background:#f1f8f7;border:1px solid #d7e8e5;border-radius:20px;
      padding:20px 24px;margin:25px 0;line-height:1.85;color:#244c54; }
    .footer-feature strong { color:#0b5257; }
    @media(max-width:768px) {
      .block-container { padding-left:.85rem;padding-right:.85rem;padding-top:.75rem; }
      .hero { padding:1.4rem 1.2rem;gap:.9rem;border-radius:19px;align-items:flex-start; }
      .hero-logo { width:69px;height:69px;border-radius:16px; }
      .hero-title { font-size:1.55rem; }
      .hero-subtitle { font-size:.89rem;line-height:1.65; }
      .hero-badge { font-size:.71rem; }
      .hero-actions { gap:8px; }
      .hero-action { font-size:.86rem;padding:9px 12px; }
      .steps { grid-template-columns:1fr;gap:8px; }
      .step { padding:.8rem 1rem; }
      .result-price,.property-price { font-size:1.7rem; }
      .trust-chip { flex-basis:145px;font-size:.78rem;padding:10px 8px; }
      .site-topbar { align-items:flex-start; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

logo_path = Path(__file__).resolve().parent / "logo.png"

if logo_path.is_file():
    APP_ICON_BASE64 = base64.b64encode(
        logo_path.read_bytes()
    ).decode("ascii")
else:
    APP_ICON_BASE64 = ""

if logo_path.is_file():
    st.image(str(logo_path), width=110)

st.markdown(
    f"""
    <div class="site-topbar">
      <div class="site-brand">⌂ お買い得物件チェッカー</div>
      <div><a href="#step1-input">価格チェック</a> &nbsp; · &nbsp;
           <a href="#service-guide">サービスについて</a></div>
    </div>
    <section class="hero">

      <div class="hero-copy">
        <div class="hero-badge">住まい選びに、価格のものさしを。</div>
        <h1 class="hero-title">その物件価格、<br>AIで確かめよう。</h1>
        <div class="hero-subtitle">中古戸建て・マンションなどの参考価格をチェック。<br>
          物件の条件と地域情報をもとに、売出価格との差をわかりやすく表示します。</div>
        <div class="hero-actions">
          <a class="hero-action" href="#step1-input">無料で価格をチェック →</a>
          <a class="hero-action secondary" href="#service-guide">しくみを見る</a>
        </div>
      </div>
    </section>
    <div class="trust-strip">
      <div class="trust-chip">🏠 戸建て・マンションに対応</div>
      <div class="trust-chip">📍 地域・面積・築年を考慮</div>
      <div class="trust-chip">📊 取引データを活用した参考価格</div>
    </div>
    <div class="steps">
      <a href="#step1-input" style="text-decoration:none;color:inherit;display:block;">
        <div class="step"><div class="step-no">STEP 01</div><div class="step-title">物件情報を入力</div>
          <div class="step-text">種別・所在地・面積・販売価格を入力します。</div></div>
      </a>
      <a href="#step2-check" style="text-decoration:none;color:inherit;display:block;">
        <div class="step"><div class="step-no">STEP 02</div><div class="step-title">AIで参考価格を計算</div>
          <div class="step-text">学習済みモデルと利用可能な地域情報を参照します。</div></div>
      </a>
      <a href="#step3-result" style="text-decoration:none;color:inherit;display:block;">
        <div class="step"><div class="step-no">STEP 03</div><div class="step-title">価格差と注意点を確認</div>
          <div class="step-text">AIの推定結果と、購入前の確認事項を表示します。</div></div>
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 将来API・AI対応の共通データ設計
# =========================================================
# どの取得元（手入力 / URL / 国交省API / 将来の正式データ提供）でも、
# 最終的にこの項目名へ揃えてから価格判定へ渡します。

PROPERTY_TYPES = [
    "新築戸建て", "中古戸建て", "新築マンション", "中古マンション",
    "賃貸・借家", "アパート・収益物件", "その他"
]

COMMON_PROPERTY_FIELDS = [
    "物件ID", "物件名", "物件種別", "価格", "住所", "都道府県", "市区町村",
    "土地面積", "建物面積", "専有面積", "面積", "築年月", "築年", "間取り", "構造",
    "最寄り駅", "駅徒歩分", "用途地域", "接道", "住宅メーカー", "施工会社",
    "取引年月", "取引時地価", "現在地価", "緯度", "経度", "データ元", "元URL",
    "取得日時", "新築中古区分", "入居状況", "賃料", "年間収入", "管理費", "修繕積立金"
]

ML_FEATURE_FIELDS = [
    "物件種別", "価格", "都道府県", "市区町村", "土地面積", "建物面積", "専有面積",
    "築年", "間取り", "構造", "駅徒歩分", "用途地域", "接道", "住宅メーカー",
    "取引年月", "取引時地価", "現在地価", "新築中古区分", "賃料", "年間収入",
    "管理費", "修繕積立金"
]


def blank_property_record():
    """欠損を許容した共通物件レコードを作る。"""
    return {field: None for field in COMMON_PROPERTY_FIELDS}


def canonical_property_type(value):
    """取得元ごとの物件種別表記を共通分類へ寄せる。"""
    text = "" if value is None else str(value).strip()
    if not text:
        return "その他"
    if "マンション" in text:
        return "新築マンション" if "新築" in text else "中古マンション"
    if any(word in text for word in ["アパート", "一棟", "収益", "投資"]):
        return "アパート・収益物件"
    if any(word in text for word in ["賃貸", "借家"]):
        return "賃貸・借家"
    if any(word in text for word in ["戸建", "一戸建"]):
        return "新築戸建て" if "新築" in text else "中古戸建て"
    return text if text in PROPERTY_TYPES else "その他"


def choose_analysis_area(record):
    """物件種別に応じて価格比較に使う面積を選ぶ。"""
    ptype = canonical_property_type(record.get("物件種別"))
    candidates = ["面積"]
    if "マンション" in ptype:
        candidates = ["専有面積", "面積", "建物面積"]
    elif "戸建て" in ptype or "収益" in ptype or "賃貸" in ptype:
        candidates = ["建物面積", "面積", "専有面積", "土地面積"]
    for key in candidates:
        value = pd.to_numeric(pd.Series([record.get(key)]), errors="coerce").iloc[0]
        if not pd.isna(value) and float(value) > 0:
            return float(value)
    return None


def normalize_property_record(raw, source="manual", source_url=None):
    """任意の取得元データを共通スキーマへ変換する。"""
    record = blank_property_record()
    if raw:
        for key, value in dict(raw).items():
            if key in record:
                record[key] = value
    record["物件種別"] = canonical_property_type(record.get("物件種別"))
    if not record.get("面積"):
        record["面積"] = choose_analysis_area(record)
    record["データ元"] = source
    record["元URL"] = source_url
    record["取得日時"] = datetime.now().isoformat(timespec="seconds")
    return record


def normalize_property_dataframe(df, source="file"):
    """Excel/CSV/APIの表を共通列へ揃える。未知の列は残す。"""
    result = normalize_company_columns(df)
    if "物件種別" in result.columns:
        result["物件種別"] = result["物件種別"].apply(canonical_property_type)
    if "データ元" not in result.columns:
        result["データ元"] = source
    if "取得日時" not in result.columns:
        result["取得日時"] = datetime.now().isoformat(timespec="seconds")
    return result


def build_ml_training_dataframe(df):
    """将来の機械学習へそのまま渡しやすい列順に整える。"""
    work = normalize_property_dataframe(df, source="training")
    for field in ML_FEATURE_FIELDS:
        if field not in work.columns:
            work[field] = pd.NA
    return work[ML_FEATURE_FIELDS].copy()


# =========================================================
# 国土交通省 不動産情報ライブラリ API
# =========================================================
# XIT001: 不動産価格（取引価格・成約価格）情報取得API
# XIT002: 都道府県内市区町村一覧取得API
# APIキーはコードへ直書きせず、Streamlit Secrets の MLIT_API_KEY から取得します。

MLIT_API_ENABLED = True
MLIT_API_BASE_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001"
MLIT_CITY_API_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT002"

PREFECTURE_CODES = {
    "北海道": "01", "青森県": "02", "岩手県": "03", "宮城県": "04", "秋田県": "05",
    "山形県": "06", "福島県": "07", "茨城県": "08", "栃木県": "09", "群馬県": "10",
    "埼玉県": "11", "千葉県": "12", "東京都": "13", "神奈川県": "14", "新潟県": "15",
    "富山県": "16", "石川県": "17", "福井県": "18", "山梨県": "19", "長野県": "20",
    "岐阜県": "21", "静岡県": "22", "愛知県": "23", "三重県": "24", "滋賀県": "25",
    "京都府": "26", "大阪府": "27", "兵庫県": "28", "奈良県": "29", "和歌山県": "30",
    "鳥取県": "31", "島根県": "32", "岡山県": "33", "広島県": "34", "山口県": "35",
    "徳島県": "36", "香川県": "37", "愛媛県": "38", "高知県": "39", "福岡県": "40",
    "佐賀県": "41", "長崎県": "42", "熊本県": "43", "大分県": "44", "宮崎県": "45",
    "鹿児島県": "46", "沖縄県": "47"
}


def get_mlit_api_key():
    """Streamlit Secretsを最優先し、ローカル環境変数を予備として使う。"""
    # Streamlit Community Cloud の Secrets に
    # MLIT_API_KEY = "..." と登録した値を直接読み込む。
    try:
        secret_value = str(st.secrets["MLIT_API_KEY"]).strip()
        if secret_value:
            return secret_value
    except Exception:
        pass

    # 念のため [mlit] セクションで登録した場合にも対応する。
    try:
        mlit_section = st.secrets["mlit"]
        for key_name in ("MLIT_API_KEY", "API_KEY", "api_key"):
            try:
                secret_value = str(mlit_section[key_name]).strip()
                if secret_value:
                    return secret_value
            except Exception:
                continue
    except Exception:
        pass

    # ローカルPCで環境変数を設定している場合の予備経路。
    return os.environ.get("MLIT_API_KEY", "").strip()


def _mlit_headers():
    api_key = get_mlit_api_key()
    if not api_key:
        raise RuntimeError(
            "MLIT_API_KEY が見つかりません。Streamlit Community Cloud の Secrets に登録してください。"
        )
    return {
        "Ocp-Apim-Subscription-Key": api_key,
        "Accept": "application/json"
    }


def _to_number(value):
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    number = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    return None if pd.isna(number) else float(number)


def _year_from_text(value):
    if value is None:
        return None
    match = re.search(r"(19\d{2}|20\d{2})", str(value))
    return int(match.group(1)) if match else None


def _mlit_property_type(row):
    """XIT001の取引種類をアプリ共通の物件種別へ寄せる。"""
    type_text = str(row.get("Type") or "").strip()
    use_text = str(row.get("Use") or "").strip()

    if "マンション" in type_text:
        return "中古マンション"

    if "土地と建物" in type_text:
        # 明確に非住宅用途だけのものは戸建て比較から外す。
        non_residential_words = ["事務所", "店舗", "工場", "倉庫", "駐車場", "作業場"]
        if use_text and "住宅" not in use_text and any(word in use_text for word in non_residential_words):
            return "その他"

        building_year = _year_from_text(row.get("BuildingYear"))
        trade_year = _year_from_text(row.get("Period"))
        if building_year and trade_year and 0 <= trade_year - building_year <= 1:
            return "新築戸建て"
        return "中古戸建て"

    return "その他"


def convert_mlit_rows_to_dataframe(rows):
    """XIT001のJSON配列を価格チェック用の共通列へ変換する。"""
    records = []

    for index, row in enumerate(rows or []):
        price_yen = _to_number(row.get("TradePrice"))
        land_or_unit_area = _to_number(row.get("Area"))
        total_floor_area = _to_number(row.get("TotalFloorArea"))
        property_type = _mlit_property_type(row)
        prefecture = str(row.get("Prefecture") or "").strip()
        municipality = str(row.get("Municipality") or "").strip()
        district = str(row.get("DistrictName") or "").strip()
        address = f"{prefecture}{municipality}{district}".strip()
        building_year = _year_from_text(row.get("BuildingYear"))

        land_area = None
        building_area = None
        exclusive_area = None
        analysis_area = None

        if "マンション" in property_type:
            exclusive_area = land_or_unit_area
            analysis_area = exclusive_area
        elif "戸建て" in property_type:
            land_area = land_or_unit_area
            building_area = total_floor_area
            analysis_area = building_area or land_area
        else:
            land_area = land_or_unit_area
            building_area = total_floor_area
            analysis_area = building_area or land_area

        road_parts = [
            str(row.get("Direction") or "").strip(),
            str(row.get("Classification") or "").strip(),
        ]
        breadth = str(row.get("Breadth") or "").strip()
        if breadth:
            road_parts.append(f"幅員{breadth}m")
        road = " ".join(part for part in road_parts if part)

        raw_record = {
            "物件ID": f"MLIT-{row.get('DistrictCode') or row.get('MunicipalityCode') or 'NA'}-{index}",
            "物件名": f"{address} {row.get('Type') or ''}".strip(),
            "物件種別": property_type,
            # アプリ内の価格単位は「万円」。APIの円を万円へ変換する。
            "価格": (price_yen / 10000.0) if price_yen is not None else None,
            "住所": address,
            "都道府県": prefecture,
            "市区町村": municipality,
            "土地面積": land_area,
            "建物面積": building_area,
            "専有面積": exclusive_area,
            "面積": analysis_area,
            "築年": building_year,
            "間取り": row.get("FloorPlan") or None,
            "構造": row.get("Structure") or None,
            "用途地域": row.get("CityPlanning") or None,
            "接道": road or None,
            "取引年月": row.get("Period") or None,
            "データ元": "mlit_api",
            "取得日時": datetime.now().isoformat(timespec="seconds"),
        }
        record = normalize_property_record(raw_record, source="mlit_api")
        record["価格情報区分"] = row.get("PriceCategory") or None
        record["取引種類"] = row.get("Type") or None
        record["地区コード"] = row.get("DistrictCode") or None
        records.append(record)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def fetch_mlit_property_data(params=None):
    """XIT001から不動産取引価格・成約価格を取得する。"""
    if not MLIT_API_ENABLED:
        return pd.DataFrame()

    request_params = dict(params or {})
    request_params.setdefault("language", "ja")

    if not request_params.get("year"):
        raise ValueError("取引年を指定してください。")
    if not any(request_params.get(key) for key in ("area", "city", "station")):
        raise ValueError("都道府県・市区町村・駅のいずれかを指定してください。")

    response = requests.get(
        MLIT_API_BASE_URL,
        params=request_params,
        headers=_mlit_headers(),
        timeout=30
    )

    if response.status_code in (401, 403):
        raise RuntimeError("APIキーが認証されませんでした。Secrets の MLIT_API_KEY を確認してください。")

    response.raise_for_status()
    payload = response.json()

    if str(payload.get("status", "")).upper() not in ("", "OK"):
        raise RuntimeError(f"国土交通省APIからエラーが返されました：{payload.get('status')}")

    return convert_mlit_rows_to_dataframe(payload.get("data", []))
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_mlit_cities(area_code):
    """XIT002から指定都道府県の市区町村一覧を取得する。"""
    response = requests.get(
        MLIT_CITY_API_URL,
        params={"area": area_code},
        headers=_mlit_headers(),
        timeout=20
    )

    if response.status_code in (401, 403):
        raise RuntimeError("APIキーが認証されませんでした。")

    response.raise_for_status()
    payload = response.json()

    cities = []

    for item in payload.get("data", []) or []:
        city_id = str(item.get("id") or "").strip()
        city_name = str(item.get("name") or "").strip()

        if city_id and city_name:
            cities.append((city_id, city_name))

    return cities


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"

# V7は研究用の候補モデルです。既存の本番AIは自動変更しません。
V7_DIR = MODEL_DIR / "candidates"

@st.cache_data(ttl=60, show_spinner=False)
def read_v7_research_status():
    report_path = V7_DIR / "v7_report.json"
    state_path = V7_DIR / "v7_state.json"
    candidate_path = V7_DIR / "v7_used_house_candidate.joblib"
    result = {"candidate_exists": candidate_path.is_file(), "report": {}, "state": {}}
    for key, path in (("report", report_path), ("state", state_path)):
        if path.is_file():
            try:
                result[key] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
    return result


def show_v7_research_status():
    status = read_v7_research_status()
    report = status["report"]
    state = status["state"]
    with st.expander("🔬 AIの自動研究・精度改善状況", expanded=False):
        if not status["candidate_exists"]:
            st.info("V7の学習候補はまだ保存されていません。現在の本番AIで査定しています。")
            return
        st.success("V7の中古戸建て候補モデルを保存済み（本番AIは維持）")
        winner = report.get("winner") or {}
        selection = winner.get("selection_2024") or {}
        st.write("研究で選ばれた方法：", winner.get("name", "確認中"))
        if selection.get("median_ape_pct") is not None:
            st.metric("2024年の検証・誤差中央値", f"{selection['median_ape_pct']:.2f}%")
        if report.get("diagnostic_2025"):
            d = report["diagnostic_2025"]
            st.caption(f"2025年の参考検証：誤差中央値 {d.get('median_ape_pct', '不明')}%")
        if state.get("status") == "needs_comparable_validation":
            st.warning("新データを検出しました。同一条件での比較が済むまで、以前の候補を維持します。")
        st.caption("研究結果は候補です。既存の本番AIと同じ未学習データで比較し、改善が確認されるまでは査定に使用しません。")
        v8_report_path = V7_DIR / "v8_research_report.json"
        if v8_report_path.is_file():
            try:
                v8 = json.loads(v8_report_path.read_text(encoding="utf-8"))
                done = len(v8.get("experiments", []))
                st.write(f"V8追加研究：{done}実験完了 / 状態：{v8.get('status', '不明')}")
                best = v8.get("best_so_far") or v8.get("winner") or {}
                score = (best.get("2024") or {}).get("median_ape_pct")
                if score is not None:
                    st.write(f"V8の開発検証での最良誤差中央値：{score:.2f}%")
                st.caption("V8は研究候補です。実際の査定はV7（利用不可なら従来AI）を使用します。")
            except (OSError, ValueError, TypeError):
                pass


show_v7_research_status()


def show_v9_research_status():
    """Read the continuously updated research report without interrupting training."""
    import json as _json
    from datetime import datetime as _datetime
    report_path = MODEL_DIR / "candidates" / "adaptive_research_report.json"
    with st.expander("🔬 自動研究の進捗・査定AIの状態", expanded=False):
        active_mode = read_used_house_mode()
        st.write("中古戸建ての査定AI:", {"v7": "V7候補（従来の試験適用を維持）", "production": "本番AI"}.get(active_mode, active_mode))
        if not report_path.is_file():
            st.info("研究レポートはまだ作成されていません。")
            return
        try:
            report = _json.loads(report_path.read_text(encoding="utf-8"))
            experiments = report.get("experiments") or []
            best = report.get("best_2024") or {}
            updated = report.get("updated_at") or report.get("completed_at") or report.get("started_at")
            c1,c2,c3 = st.columns(3)
            c1.metric("実験回数", len(experiments))
            c2.metric("2024年の最良誤差中央値", str(best.get("median_ape_pct", "—")) + ("%" if best.get("median_ape_pct") is not None else ""))
            c3.metric("研究状態", str(report.get("status", "不明")))
            if updated:
                st.caption("最終レポート更新: " + str(updated))
            diagnostics = report.get("diagnostics") or []
            if diagnostics:
                st.write("誤差が大きい条件")
                st.dataframe(diagnostics[:8], use_container_width=True, hide_index=True)
            if experiments:
                st.write("直近の改善仮説と結果")
                rows = []
                for e in experiments[-15:]:
                    m=e.get("validation_2024") or {}
                    rows.append({"回": e.get("round"), "改善仮説": e.get("hypothesis"), "誤差中央値%": m.get("median_ape_pct"), "改善": "✓" if e.get("new_champion") else "", "状態": e.get("status")})
                st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption("2024年は研究で繰り返し使う開発検証データです。未知データでの精度は保証できません。研究候補は自動的に査定へ切り替わりません。")
        except (OSError, ValueError, TypeError) as exc:
            st.warning("研究レポートを読み込み中です。次回更新で再表示します。")


def read_used_house_mode():
    """Preserve the previous V7 trial unless explicitly configured otherwise."""
    import json as _json
    config_path = MODEL_DIR / "candidates" / "site_model_settings.json"
    try:
        mode = _json.loads(config_path.read_text(encoding="utf-8")).get("used_house_mode", "v7")
        return mode if mode in ("v7", "production") else "v7"
    except (OSError, ValueError, TypeError, AttributeError):
        return "v7"


show_v9_research_status()


@st.cache_resource(show_spinner=False)
def load_price_model(model_key):
    """models フォルダから学習済みAIモデルを読み込む。"""
    model_path = MODEL_DIR / f"{model_key}_model.joblib"

    if not model_path.exists():
        from urllib.request import urlopen
        from urllib.error import URLError, HTTPError
        import shutil as _shutil
        urls = {
            "new_house": "https://github.com/chi-pao/-/releases/download/models-v1/new_house_model.joblib",
            "used_condo": "https://github.com/chi-pao/-/releases/download/models-v1/used_condo_model.joblib",
            "used_house": "https://github.com/chi-pao/-/releases/download/models-v1/used_house_model.joblib",
        }
        url = urls.get(model_key)
        if not url:
            return None
        model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = model_path.with_suffix(".download")
        try:
            with st.spinner("AIモデルを準備しています。初回は数分かかる場合があります。"):
                with urlopen(url, timeout=90) as response, tmp_path.open("wb") as dest:
                    _shutil.copyfileobj(response, dest)
                if tmp_path.stat().st_size < 1000000:
                    raise RuntimeError("Downloaded model is unexpectedly small")
                tmp_path.replace(model_path)
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            st.error(f"AIモデルの取得に失敗しました: {type(e).__name__}: {e}")
            return None

    return joblib.load(model_path)


# V7試験運用: 中古戸建てだけ候補モデルを優先。失敗時は従来モデルへ戻す。
@st.cache_resource(show_spinner=False)
def _load_v7_candidate_cached(path_string, modified_ns):
    return joblib.load(path_string)


def load_v7_candidate():
    candidate_path = V7_DIR / "v7_used_house_candidate.joblib"
    if not candidate_path.is_file():
        return None
    return _load_v7_candidate_cached(str(candidate_path), candidate_path.stat().st_mtime_ns)


def get_model_key(property_type):
    return {
        "中古マンション": "used_condo",
        "新築マンション": "new_condo",
        "中古戸建て": "used_house",
        "新築戸建て": "new_house",
        "中古アパート・一棟収益物件": "used_income",
        "新築アパート・一棟収益物件": "new_income",
        "賃貸戸建て・借家": "rent_house",
        "賃貸・借家": "rent_house",
        "賃貸アパート": "rent_apartment",
        "賃貸マンション": "rent_condo",
    }.get(property_type)


def current_quarter():
    return ((datetime.now().month - 1) // 3) + 1
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_mlit_districts(city_code):
    if not city_code:
        return []

    districts = set()

    current_year = datetime.now().year
    years = [
        current_year,
        current_year - 1,
        current_year - 2,
        current_year - 3,
        current_year - 4,
    ]

    city_code = str(city_code)

    if city_code == "14100":
        city_codes = [
            "14101",
            "14102",
            "14103",
            "14104",
            "14105",
            "14106",
            "14107",
            "14108",
            "14109",
            "14110",
            "14111",
            "14112",
            "14113",
            "14114",
            "14115",
            "14116",
            "14117",
            "14118",
        ]
    else:
        city_codes = [city_code]

    for search_city_code in city_codes:
        for year in years:
            for quarter in (4, 3, 2, 1):
                try:
                    response = requests.get(
                        MLIT_API_BASE_URL,
                        params={
                            "year": year,
                            "quarter": quarter,
                            "city": search_city_code,
                            "language": "ja",
                        },
                        headers=_mlit_headers(),
                        timeout=20,
                    )

                    if not response.ok:
                        continue

                    payload = response.json()

                    for item in payload.get("data", []) or []:
                        district_name = str(
                            item.get("DistrictName") or ""
                        ).strip()

                        municipality = str(
                            item.get("Municipality") or ""
                        ).strip()

                        if district_name:
                            if municipality:
                                districts.add(
                                    f"{municipality} {district_name}"
                                )
                            else:
                                districts.add(district_name)

                except Exception:
                    continue

            if districts:
                break

    return sorted(districts)


def apply_verified_post_calibration(prediction_yen, bundle, prefecture_name):
    """Apply only a correction learned from held-out validation predictions."""
    config = bundle.get("post_calibration") or {}
    lookup = config.get("lookup") or {}
    mode = config.get("mode")
    if not lookup or not mode:
        return float(prediction_yen)
    if mode == "global":
        tag = "global"
    elif mode == "prefecture":
        tag = str(prefecture_name)
    elif mode == "predicted_band":
        key = bundle.get("model_key", "")
        cuts = ([50_000, 100_000, 200_000, 400_000] if key.startswith("rent_")
                else [1_000_000, 10_000_000, 50_000_000, 200_000_000])
        tag = str(int(np.searchsorted(cuts, prediction_yen, side="right")))
    else:
        return float(prediction_yen)
    return float(prediction_yen) * float(np.exp(float(lookup.get(tag, 0.0))))


def predict_ai_reference_price(
    property_type,
    municipality_code,
    district_name,
    building_year,
    structure,
    city_planning,
    floor_plan,
    renovation,
    condo_area=None,
    land_area=None,
    building_area=None,
    land_price_per_sqm=None,
    city_land_price_per_sqm=None,
    land_price_ratio=None,
    land_price_match_level=0.0,
    annual_income_yen=None,
    station_minutes=None,
):
    """学習済みAIモデルから参考価格（売買:万円、賃貸:万円/月）とモデル情報を返す。"""
    model_key = get_model_key(property_type)

    if model_key is None:
        raise ValueError(
            "この物件種別はAI価格予測に対応していません。"
        )

    bundle = load_price_model(model_key)

    if bundle is None:
        raise FileNotFoundError(
            f"models/{model_key}_model.joblib が見つかりません。"
        )

    now = datetime.now()
    transaction_year = now.year
    quarter = current_quarter()
    building_age = max(0, transaction_year - int(building_year))

    local_land_value = (
        float(land_price_per_sqm)
        if land_price_per_sqm is not None
        else np.nan
    )
    city_land_value = (
        float(city_land_price_per_sqm)
        if city_land_price_per_sqm is not None
        else np.nan
    )

    # 学習時の都道府県・建物用途と同じカテゴリを推定時にも渡す。
    pref_code = str(municipality_code or "").zfill(5)[:2]
    prefecture_name = next(
        (name for name, code in PREFECTURE_CODES.items() if str(code).zfill(2) == pref_code),
        np.nan,
    )
    model_use = "共同住宅" if model_key in ("used_income", "new_income") else (
        "住宅" if model_key in ("used_house", "new_house") else np.nan
    )
    common = {
        "Prefecture": prefecture_name,
        "Use": model_use,
        "MunicipalityCode": str(municipality_code) if municipality_code else np.nan,
        "DistrictName": district_name if district_name else np.nan,
        "Structure": structure if structure and structure != "不明" else np.nan,
        "CityPlanning": city_planning if city_planning and city_planning != "不明" else np.nan,
        "PriceCategory": np.nan,
        "BuildingYear": float(building_year),
        "BuildingAge": float(building_age),
        "TransactionYear": float(transaction_year),
        "Quarter": float(quarter),
        "AnnualIncome": annual_income_yen if annual_income_yen and annual_income_yen > 0 else np.nan,
        "StationMinutes": station_minutes if station_minutes and station_minutes > 0 else np.nan,
        "LandPricePerSqm": local_land_value,
        "CityLandPricePerSqm": city_land_value,
        "LandPriceRatio": (
            float(land_price_ratio)
            if land_price_ratio is not None
            else np.nan
        ),
        "LandPriceMatchLevel": float(land_price_match_level or 0.0),
        "LogLandPricePerSqm": (
            float(np.log1p(local_land_value))
            if np.isfinite(local_land_value) and local_land_value > 0
            else np.nan
        ),
        "LogCityLandPricePerSqm": (
            float(np.log1p(city_land_value))
            if np.isfinite(city_land_value) and city_land_value > 0
            else np.nan
        ),
    }

    if model_key in ("used_condo", "new_condo", "rent_house", "rent_apartment", "rent_condo"):
        common.update(
            {
                "Area": float(condo_area),
                "FloorPlan": floor_plan
                if floor_plan and floor_plan != "選択してください"
                else np.nan,
                "Renovation": renovation
                if renovation and renovation != "不明"
                else np.nan,
            }
        )
    else:
        common.update(
            {
                "Area": float(land_area),
                "TotalFloorArea": float(building_area),
            }
        )

    # 学習時の面積派生特徴量と同じ計算を利用する。
    area_num = common.get("Area", np.nan)
    floor_num = common.get("TotalFloorArea", np.nan)
    common["LogArea"] = float(np.log1p(area_num)) if np.isfinite(area_num) and area_num > 0 else np.nan
    common["LogTotalFloorArea"] = float(np.log1p(floor_num)) if np.isfinite(floor_num) and floor_num > 0 else np.nan
    common["LandBuildingRatio"] = float(area_num / floor_num) if np.isfinite(area_num) and np.isfinite(floor_num) and floor_num > 0 else np.nan
    revenue_num = common.get("AnnualIncome", np.nan)
    common["LogAnnualIncome"] = float(np.log1p(revenue_num)) if np.isfinite(revenue_num) and revenue_num > 0 else np.nan

    # V7 is available only for used detached houses. Original production models stay intact.
    if model_key == "used_house" and read_used_house_mode() == "v7":
        try:
            candidate = load_v7_candidate()
            if candidate is not None:
                if candidate.get("model_key") != "used_house" or candidate.get("target_unit") != "yen":
                    raise ValueError("V7候補の種別または価格単位が一致しません")
                candidate_features = candidate.get("feature_columns") or []
                if not candidate_features or "pipeline" not in candidate:
                    raise ValueError("V7候補の特徴量またはモデルがありません")
                candidate_frame = pd.DataFrame(
                    [{column: common.get(column, np.nan) for column in candidate_features}],
                    columns=candidate_features,
                )
                candidate_yen = float(candidate["pipeline"].predict(candidate_frame)[0])
                if not np.isfinite(candidate_yen) or candidate_yen <= 0:
                    raise ValueError("V7候補の予測値が不正です")
                active_bundle = dict(candidate)
                active_bundle["display_name"] = "V7研究候補（中古戸建て・試験適用）"
                active_bundle["metrics"] = {
                    "median_ape_pct": (candidate.get("validation_report", {}).get("winner", {})
                                       .get("selection_2024", {}).get("median_ape_pct"))
                }
                active_bundle["v7_active"] = True
                return candidate_yen / 10000.0, active_bundle
        except Exception as exc:
            st.warning(f"V7候補を読み込めなかったため従来AIを使用します：{exc}")

    ensemble_components = bundle.get("ensemble_components") or []

    if ensemble_components:
        weighted_logs = []
        total_weight = 0.0

        for component in ensemble_components:
            component_pipeline = component.get("pipeline")
            component_features = component.get("feature_columns") or []
            component_weight = float(component.get("weight", 0.0) or 0.0)

            if component_pipeline is None or component_weight <= 0:
                continue

            component_row = {
                column: common.get(column, np.nan)
                for column in component_features
            }
            component_df = pd.DataFrame(
                [component_row],
                columns=component_features,
            )
            component_prediction = float(
                component_pipeline.predict(component_df)[0]
            )
            component_prediction = max(component_prediction, 1.0)

            weighted_logs.append(
                component_weight * np.log(component_prediction)
            )
            total_weight += component_weight

        if total_weight > 0 and weighted_logs:
            prediction_yen = float(
                np.exp(sum(weighted_logs) / total_weight)
            )
            prediction_yen = apply_verified_post_calibration(prediction_yen, bundle, prefecture_name)
            return prediction_yen / 10000.0, bundle

    feature_columns = bundle.get("feature_columns", list(common.keys()))
    row = {
        column: common.get(column, np.nan)
        for column in feature_columns
    }

    input_df = pd.DataFrame([row], columns=feature_columns)
    prediction_yen = float(bundle["pipeline"].predict(input_df)[0])

    prediction_yen = apply_verified_post_calibration(prediction_yen, bundle, prefecture_name)
    return prediction_yen / 10000.0, bundle


def load_saved_mlit_data():
    data_dir = Path(__file__).resolve().parent / "data"
    files = sorted(data_dir.glob("mlit_*.csv"))

    frames = []

    for file in files:
        try:
            df = pd.read_csv(
                file,
                encoding="utf-8-sig",
                low_memory=False,
            )
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


def get_comparison_data(manual_data=None, api_params=None):
    """アップロードデータと国交省APIデータを共通形式で結合する。"""
    frames = []
    if manual_data is not None and len(manual_data) > 0:
        frames.append(normalize_property_dataframe(manual_data, source="upload"))
    if api_params:
        api_data = fetch_mlit_property_data(api_params)
        if api_data is not None and len(api_data) > 0:
            frames.append(normalize_property_dataframe(api_data, source="mlit_api"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def extract_municipality(address):
    if not address:
        return ""

    text = str(address).strip()

    match = re.search(
        r"[都道府県](.+?[市区町村])",
        text
    )

    if match:
        return match.group(1)

    return ""


def calculate_unit_price(df):
    result = df.copy()

    if "価格" in result.columns and "面積" in result.columns:

        price = pd.to_numeric(
            result["価格"],
            errors="coerce"
        )

        area = pd.to_numeric(
            result["面積"],
            errors="coerce"
        )

        result["㎡単価"] = (
            price /
            area.replace(0, pd.NA)
        )

    return result


def filter_comparison_properties(
    comparison_data,
    target_type,
    target_address,
    target_area,
    target_age
):

    data = comparison_data.copy()

    if "物件種別" in data.columns:

        data = data[
            data["物件種別"]
            .astype(str)
            .str.strip()
            == str(target_type).strip()
        ]

    if len(data) == 0:
        return data, "物件種別のみ"

    target_municipality = extract_municipality(
        target_address
    )

    if "住所" in data.columns and target_municipality:

        municipalities = data["住所"].apply(
            extract_municipality
        )

        same_municipality = data[
            municipalities == target_municipality
        ]

        if len(same_municipality) >= 3:
            data = same_municipality

    if "面積" in data.columns and target_area > 0:

        data = data.copy()

        data["_比較面積"] = pd.to_numeric(
            data["面積"],
            errors="coerce"
        )

        area_min = target_area * 0.70
        area_max = target_area * 1.30

        area_filtered = data[
            data["_比較面積"].between(
                area_min,
                area_max
            )
        ]

        if len(area_filtered) >= 3:
            data = area_filtered

    if "築年" in data.columns:

        data = data.copy()

        data["_比較築年"] = pd.to_numeric(
            data["築年"],
            errors="coerce"
        )

        age_filtered = data[
            data["_比較築年"].between(
                target_age - 15,
                target_age + 15
            )
        ]

        if len(age_filtered) >= 3:
            data = age_filtered

    for column in [
        "_比較面積",
        "_比較築年"
    ]:

        if column in data.columns:

            data = data.drop(
                columns=[column]
            )

    return data, "自動絞り込み"


def calculate_manufacturer_summary(data):

    required_columns = [
        "住宅メーカー",
        "価格",
        "面積"
    ]

    if not all(
        column in data.columns
        for column in required_columns
    ):
        return pd.DataFrame()

    work = data.copy()

    work["価格"] = pd.to_numeric(
        work["価格"],
        errors="coerce"
    )

    work["面積"] = pd.to_numeric(
        work["面積"],
        errors="coerce"
    )

    work["メーカー別㎡単価"] = (
        work["価格"]
        /
        work["面積"].replace(
            0,
            pd.NA
        )
    )

    work = work.dropna(
        subset=[
            "住宅メーカー",
            "メーカー別㎡単価"
        ]
    )

    if len(work) == 0:
        return pd.DataFrame()

    summary = (
        work[
            [
                "住宅メーカー",
                "メーカー別㎡単価"
            ]
        ]
        .groupby("住宅メーカー")
        .agg({
            "メーカー別㎡単価": [
                "mean",
                "count"
            ]
        })
        .reset_index()
    )

    summary.columns = [
        "住宅メーカー",
        "平均㎡単価",
        "件数"
    ]

    return summary


def calculate_manufacturer_factor(
    comparison_data,
    target_manufacturer
):

    if not target_manufacturer:
        return 1.0, 0

    summary = calculate_manufacturer_summary(
        comparison_data
    )

    if summary.empty:
        return 1.0, 0

    target_rows = summary[
        summary["住宅メーカー"]
        .astype(str)
        .str.strip()
        ==
        str(target_manufacturer).strip()
    ]

    if len(target_rows) == 0:
        return 1.0, 0

    target_count = int(
        target_rows.iloc[0]["件数"]
    )

    if target_count < 2:
        return 1.0, target_count

    overall_unit_price = (
        pd.to_numeric(
            comparison_data["価格"],
            errors="coerce"
        )
        /
        pd.to_numeric(
            comparison_data["面積"],
            errors="coerce"
        ).replace(
            0,
            pd.NA
        )
    )

    overall_median = (
        overall_unit_price.median()
    )

    if (
        pd.isna(overall_median)
        or
        overall_median <= 0
    ):
        return 1.0, target_count

    manufacturer_unit_price = float(
        target_rows.iloc[0]["平均㎡単価"]
    )

    factor = (
        manufacturer_unit_price
        /
        overall_median
    )

    factor = max(
        0.90,
        min(
            1.10,
            factor
        )
    )

    return factor, target_count


# =========================================================
# 一括チェック用の共通関数
# =========================================================

def normalize_company_columns(df):
    """よくある列名をアプリ内の標準列名へ自動変換する。"""
    aliases = {
        "物件名": ["物件名", "名称", "物件名称", "建物名"],
        "住所": ["住所", "所在地", "物件所在地", "所在地住所"],
        "価格": ["価格", "販売価格", "売価", "価格(万円)", "価格（万円）"],
        "面積": ["面積", "建物面積", "専有面積", "延床面積", "面積(㎡)", "面積（㎡）"],
        "築年": ["築年", "築年月", "建築年", "竣工年"],
        "物件種別": ["物件種別", "種別", "物件タイプ"],
        "住宅メーカー": ["住宅メーカー", "メーカー", "ハウスメーカー"],
        "施工会社": ["施工会社", "施工", "建築会社"],
        "土地面積": ["土地面積", "敷地面積"],
        "建物面積": ["建物面積", "延床面積"],
        "専有面積": ["専有面積", "専有面積(㎡)", "専有面積（㎡）"],
        "築年月": ["築年月", "建築年月", "竣工年月"],
        "最寄り駅": ["最寄り駅", "最寄駅", "駅"],
        "駅徒歩分": ["駅徒歩分", "徒歩分", "駅距離", "徒歩"],
        "構造": ["構造", "建物構造"],
        "用途地域": ["用途地域", "都市計画用途地域"],
        "接道": ["接道", "接道状況", "前面道路"],
        "取引年月": ["取引年月", "取引時期", "成約年月"],
        "取引時地価": ["取引時地価", "当時地価", "地価(取引時)"],
        "現在地価": ["現在地価", "地価", "最新地価"],
    }
    result = df.copy()
    existing = {str(c).strip(): c for c in result.columns}
    rename = {}
    for standard, candidates in aliases.items():
        if standard in result.columns:
            continue
        for candidate in candidates:
            if candidate in existing:
                rename[existing[candidate]] = standard
                break
    return result.rename(columns=rename)


def extract_year_value(value):
    if pd.isna(value):
        return 2005
    text = str(value)
    match = re.search(r"(19|20)\\d{2}", text)
    if match:
        return int(match.group(0))
    try:
        number = int(float(value))
        if 1900 <= number <= 2100:
            return number
    except Exception:
        pass
    return 2005


def evaluate_property_row(row, comparison_data):
    price = pd.to_numeric(pd.Series([row.get("価格")]), errors="coerce").iloc[0]
    area = pd.to_numeric(pd.Series([row.get("面積")]), errors="coerce").iloc[0]
    if pd.isna(price) or price <= 0:
        return None, "価格が未入力"
    if pd.isna(area) or area <= 0:
        return None, "面積が未入力"

    target_type = str(row.get("物件種別", "")).strip()
    address = str(row.get("住所", "")).strip()
    built_year = extract_year_value(row.get("築年", 2005))
    manufacturer = str(row.get("住宅メーカー", "")).strip()

    filtered, method = filter_comparison_properties(
        comparison_data, target_type, address, float(area), built_year
    )
    if len(filtered) < 2:
        filtered = comparison_data.copy()
        if target_type and "物件種別" in filtered.columns:
            same_type = filtered[
                filtered["物件種別"].astype(str).str.strip() == target_type
            ]
            if len(same_type) >= 2:
                filtered = same_type
        method = "物件種別中心"

    if "㎡単価" not in filtered.columns:
        filtered = calculate_unit_price(filtered)
    unit_prices = pd.to_numeric(filtered["㎡単価"], errors="coerce").replace(
        [float("inf"), -float("inf")], pd.NA
    ).dropna()
    if len(unit_prices) < 2:
        return None, "比較物件が不足"

    median_unit_price = float(unit_prices.median())
    building_age = max(0, datetime.now().year - built_year)
    age_adjustment = max(0.70, 1 - building_age * 0.01)
    estimated = median_unit_price * float(area) * age_adjustment

    factor, count = calculate_manufacturer_factor(filtered, manufacturer)
    if count >= 2:
        estimated *= factor

    ratio = float(price) / estimated if estimated > 0 else 1.0
    if ratio <= 0.90:
        grade, label = "A", "お買い得"
    elif ratio <= 1.10:
        grade, label = "B", "妥当な価格"
    else:
        grade, label = "C", "割高の可能性"

    return {
        "参考価格（万円）": round(estimated, 1),
        "差額（万円）": round(float(price) - estimated, 1),
        "価格評価": grade,
        "評価内容": label,
        "比較物件数": len(filtered),
        "比較方法": method,
    }, ""


# =========================================================
# 収益化・プラン設定（将来用）
# =========================================================
# 現在はすべてOFFです。公開初期は従来どおり無料・広告なしで利用できます。
# 利用者が増えたら、必要な機能だけTrueに変更して段階的に収益化できます。

MONETIZATION_ENABLED = False
VIDEO_ADS_ENABLED = False
AFFILIATE_ENABLED = True
PAYMENT_ENABLED = False
USAGE_LIMIT_ENABLED = False

DEFAULT_PLAN = "free"

PLAN_FEATURES = {
    "free": {
        "basic_evaluation": True,
        "detailed_analysis": True,
        "comparison_list": True,
        "manufacturer_adjustment": True,
        "report_export": True,
        "bulk_analysis": True,
        "ad_free": False,
    },
    "premium": {
        "basic_evaluation": True,
        "detailed_analysis": True,
        "comparison_list": True,
        "manufacturer_adjustment": True,
        "report_export": True,
        "bulk_analysis": True,
        "ad_free": True,
    },
    "pro": {
        "basic_evaluation": True,
        "detailed_analysis": True,
        "comparison_list": True,
        "manufacturer_adjustment": True,
        "report_export": True,
        "bulk_analysis": True,
        "ad_free": True,
    },
}


def get_current_plan():
    """将来ログイン・決済を接続する際の入口。現在は全員free扱い。"""
    if not MONETIZATION_ENABLED:
        return DEFAULT_PLAN

    return st.session_state.get(
        "user_plan",
        DEFAULT_PLAN
    )


def can_use_feature(feature_name, plan=None):
    """プランごとの機能利用可否を一か所で管理する。"""
    if not MONETIZATION_ENABLED:
        return True

    current_plan = plan or get_current_plan()

    return PLAN_FEATURES.get(
        current_plan,
        PLAN_FEATURES[DEFAULT_PLAN]
    ).get(feature_name, False)


def should_show_monetization(plan=None):
    """無料ユーザー向け収益化枠を表示するか判定する。"""
    if not MONETIZATION_ENABLED:
        return False

    current_plan = plan or get_current_plan()

    if PLAN_FEATURES.get(
        current_plan,
        PLAN_FEATURES[DEFAULT_PLAN]
    ).get("ad_free", False):
        return False

    return True


def show_video_monetization_slot():
    """将来の動画広告・動画アフィリエイト用スロット。"""
    if not should_show_monetization():
        return

    if not VIDEO_ADS_ENABLED:
        return

    st.markdown("#### 🎬 無料ユーザー向けおすすめ動画")
    st.caption("PR・広告を含む場合があります。")

    # 将来ここへ動画広告・動画アフィリエイトを接続します。
    # 例：st.video(VIDEO_URL)
    # 現在はVIDEO_ADS_ENABLED=Falseのためユーザー画面には表示されません。


def show_affiliate_slot():
    """A8.netの承認済み火災保険広告を査定後に表示。"""
    if not AFFILIATE_ENABLED:
        return

    # 将来、有料プランで広告非表示を有効にした場合はその設定を尊重する。
    if MONETIZATION_ENABLED and not should_show_monetization():
        return

    st.caption("PR｜火災保険の比較・無料診断（アフィリエイト広告）")
    st.markdown(
        """
        <div style="text-align:center;margin:4px 0 12px;">
            <a href="https://px.a8.net/svt/ejp?a8mat=4BECGX+3O692Q+3RU+6S6XRL" rel="nofollow">
                <img border="0" width="320" height="50" alt="火災保険の比較・無料診断（PR）"
                     style="max-width:100%;height:auto;"
                     src="https://www25.a8.net/svt/bgt?aid=261009825222&amp;wid=001&amp;eno=01&amp;mid=s00000000489041015000&amp;mc=1">
            </a>
            <img border="0" width="1" height="1" alt=""
                 src="https://www13.a8.net/0.gif?a8mat=4BECGX+3O692Q+3RU+6S6XRL">
            <div style="margin-top:12px;">
                <a href="https://px.a8.net/svt/ejp?a8mat=4BECGX+3O692Q+3RU+6S45GI" rel="nofollow">
                    あなたの家に最適な火災保険を無料診断！
                </a>
                <img border="0" width="1" height="1" alt=""
                     src="https://www12.a8.net/0.gif?a8mat=4BECGX+3O692Q+3RU+6S45GI">
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def register_usage_event(event_name):
    """将来のアクセス解析接続用。今はセッション内で件数だけ保持する。"""
    if "usage_events" not in st.session_state:
        st.session_state["usage_events"] = {}

    events = st.session_state["usage_events"]
    events[event_name] = events.get(event_name, 0) + 1


def usage_limit_reached(event_name, free_limit=None):
    """将来、無料版の回数制限を有効にするための入口。"""
    if not MONETIZATION_ENABLED:
        return False

    if not USAGE_LIMIT_ENABLED:
        return False

    if get_current_plan() != "free":
        return False

    if free_limit is None:
        return False

    current_count = st.session_state.get(
        "usage_events", {}
    ).get(event_name, 0)

    return current_count >= free_limit


# =========================================================
# 町名の五十音検索（行ボタンと連続ページ送り）
# =========================================================

KANA_GROUPS = {
    "あ": "あいうえおぁぃぅぇぉゔ",
    "か": "かきくけこがぎぐげご",
    "さ": "さしすせそざじずぜぞ",
    "た": "たちつてとだぢづでどっ",
    "な": "なにぬねの",
    "は": "はひふへほばびぶべぼぱぴぷぺぽ",
    "ま": "まみむめも",
    "や": "やゆよゃゅょ",
    "ら": "らりるれろ",
    "わ": "わをんゎ",
}


@lru_cache(maxsize=1)
def _get_town_kana_converter():
    # 町名の漢字を読み仮名に変換。
    try:
        from pykakasi import kakasi
        return kakasi()
    except ImportError:
        return None


@lru_cache(maxsize=30000)
def _town_name_reading(town_name):
    name = unicodedata.normalize("NFKC", str(town_name or "").strip())
    if not name:
        return ""
    converter = _get_town_kana_converter()
    reading = name
    if converter is not None:
        try:
            pieces = converter.convert(name)
            reading = "".join(p.get("hira", p.get("orig", "")) for p in pieces)
        except Exception:
            reading = name
    return unicodedata.normalize("NFKC", reading).strip()


@lru_cache(maxsize=30000)
def _town_name_kana_group(town_name):
    reading = _town_name_reading(town_name)
    if not reading:
        return "その他"
    initial = reading[0]
    # カタカナをひらがなへ変換
    if "ァ" <= initial <= "ヶ":
        initial = chr(ord(initial) - 0x60)
    for group, initials in KANA_GROUPS.items():
        if initial in initials:
            return group
    return "その他"


def _set_town_kana_group(group):
    st.session_state["town_kana_group"] = group
    st.session_state["town_visible_page"] = 0


def _select_visible_town(name):
    """候補ボタンを押したとき、従来の住所選択と同じ状態へ反映する。"""
    st.session_state["address_district"] = name


def _town_nearby_candidates(district_names, selected_group, count_each=5):
    """該当行が0件のとき、五十音順で直前・直後の町名を候補にする。"""
    groups = list(KANA_GROUPS)
    if selected_group not in groups:
        return []
    target_index = groups.index(selected_group)
    # 選択行の前後で、実際に町名がある最寄りの行を探す。
    previous = []
    following = []
    for index in range(target_index - 1, -1, -1):
        names = [n for n in district_names if _town_name_kana_group(n) == groups[index]]
        if names:
            previous = sorted(names, key=lambda n: (_town_name_reading(n), n))[-count_each:]
            break
    for index in range(target_index + 1, len(groups)):
        names = [n for n in district_names if _town_name_kana_group(n) == groups[index]]
        if names:
            following = sorted(names, key=lambda n: (_town_name_reading(n), n))[:count_each]
            break
    return previous + following


def _adjacent_town_kana_group(district_names, current_group, direction):
    """五十音の隣接する、町名が存在する行を探す（行をまたぐページ送り用）。"""
    groups = list(KANA_GROUPS)
    if current_group not in groups:
        return None
    start = groups.index(current_group)
    for index in range(start + direction, len(groups) if direction > 0 else -1, direction):
        candidate = groups[index]
        if any(_town_name_kana_group(name) == candidate for name in district_names):
            return candidate
    return None


def _move_town_page(direction, page, total_pages, district_names, search_mode, selected_group, page_size=12):
    """同じ行のページを移動し、端では前後の五十音行に続けて移動する。"""
    new_page = page + direction
    if 0 <= new_page < total_pages:
        st.session_state["town_visible_page"] = new_page
        return
    if search_mode != "五十音から探す":
        return
    other_group = _adjacent_town_kana_group(district_names, selected_group, direction)
    if other_group is None:
        return
    st.session_state["town_kana_group"] = other_group
    if direction < 0:
        count = sum(_town_name_kana_group(name) == other_group for name in district_names)
        st.session_state["town_visible_page"] = max(0, (count - 1) // page_size)
    else:
        st.session_state["town_visible_page"] = 0


def show_affiliate_precheck_banner():
    """査定入力の手前に、コンパクトなA8.net素材018だけを表示。"""
    if not AFFILIATE_ENABLED:
        return
    if MONETIZATION_ENABLED and not should_show_monetization():
        return
    st.caption("PR｜火災保険の広告")
    st.markdown(
        """
        <div style="text-align:center;margin:0 0 12px;">
          <a href="https://px.a8.net/svt/ejp?a8mat=4BECGX+3O692Q+3RU+6S7KWX" rel="nofollow">
            <img border="0" width="468" height="60" alt="火災保険の比較・無料診断（PR）"
                 style="display:block;max-width:100%;height:auto;margin:0 auto;"
                 src="https://www29.a8.net/svt/bgt?aid=261009825222&amp;wid=001&amp;eno=01&amp;mid=s00000000489041018000&amp;mc=1">
          </a>
          <img border="0" width="1" height="1" alt=""
               src="https://www12.a8.net/0.gif?a8mat=4BECGX+3O692Q+3RU+6S7KWX">
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# セッション初期化
# =========================================================

if "api_ready_architecture" not in st.session_state:
    st.session_state["api_ready_architecture"] = True

# 国土交通省APIは住所候補の取得など、裏側の処理にだけ使用します。
# 取引データ一覧やAPI取得ボタンは利用者画面には表示しません。


# =========================================================
# 査定前は素材018、査定後は細長い素材015・テキスト002を表示
# =========================================================


# =========================================================
# 入力方法：手入力を本体、URL自動入力を補助機能として扱う
# =========================================================

st.markdown(
    '<div class="section-title">🏠 1件の物件をチェック</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="section-description">'
    '必要な物件情報を入力して、最後に「価格チェック」を押すだけです。'
    'goo住宅のURLは貼り付けるだけで自動入力します。'
    '</div>',
    unsafe_allow_html=True
)

property_url = st.text_input(
    "物件URL（任意）",
    placeholder="goo住宅のURLなら貼り付けるだけで自動入力します",
    key="property_url"
)

fetch_url_button = False
url_fetch_succeeded = False

if property_url:
    current_url = property_url.strip()
    last_url_attempt = st.session_state.get("last_url_attempt", "")

    if current_url != last_url_attempt:
        fetch_url_button = True
        st.session_state["last_url_attempt"] = current_url

# =========================================================
# サイト判定
# =========================================================

detected_site = "その他のサイト"

if property_url:

    if "house.goo.ne.jp" in property_url:
        detected_site = "goo住宅・不動産"

    elif "suumo.jp" in property_url:
        detected_site = "SUUMO"

    elif "homes.co.jp" in property_url:
        detected_site = "LIFULL HOME'S"

    elif "athome.co.jp" in property_url:
        detected_site = "アットホーム"


# =========================================================
# URLから取得する物件データ
# =========================================================

property_data = {
    "物件名": None,
    "価格": None,
    "住所": None,
    "土地面積": None,
    "建物面積": None,
    "築年": None,
    "間取り": None,
    "物件種別": None,
    "最寄り駅": None,
    "住宅メーカー": None,
    "サイト": None
}

if property_url:
    property_data["サイト"] = detected_site

if fetch_url_button:

    register_usage_event("url_fetch_attempt")

    if property_url.strip() == "":
        st.warning("物件URLを入力してください。")

    elif not property_url.startswith(("http://", "https://")):
        st.warning("URLは http:// または https:// から始めてください。")

    elif detected_site != "goo住宅・不動産":
        st.info(
            "このURLは自動入力の対象外です。下の項目を入力して価格チェックしてください。"
        )

# =========================================================
# goo住宅
# =========================================================

if fetch_url_button and detected_site == "goo住宅・不動産":

    try:

        response = requests.get(
            property_url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        if response.status_code == 200:

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            page_text = soup.get_text(
                " ",
                strip=True
            )

            title = soup.find("h1")

            if title:

                property_data["物件名"] = (
                    title.get_text(
                        " ",
                        strip=True
                    )
                )

            price_match = re.search(
                r"価格\s*([0-9,]+)\s*万円",
                page_text
            )

            if price_match:

                property_data["価格"] = float(
                    price_match.group(1)
                    .replace(",", "")
                )

            address_match = re.search(
                r"所在地\s+(.+?)(?=\s+周辺地図|\s+交通|\s+間取り)",
                page_text
            )

            if address_match:

                property_data["住所"] = (
                    address_match.group(1).strip()
                )

            traffic_match = re.search(
                r"交通\s+(.+?)(?=\s+所在地|\s+間取り)",
                page_text
            )

            if traffic_match:

                property_data["最寄り駅"] = (
                    traffic_match.group(1).strip()
                )

            layout_match = re.search(
                r"間取り\s+([0-9]+[A-Za-zＡ-Ｚａ-ｚ0-9０-９]*[^\s|]*)",
                page_text
            )

            if layout_match:

                property_data["間取り"] = (
                    layout_match.group(1).strip()
                )

            building_area_match = re.search(
                r"建物面積\s+([0-9,.]+)\s*㎡",
                page_text
            )

            if building_area_match:

                property_data["建物面積"] = float(
                    building_area_match.group(1)
                    .replace(",", "")
                )

            land_area_match = re.search(
                r"土地面積\s+([0-9,.]+)\s*㎡",
                page_text
            )

            if land_area_match:

                property_data["土地面積"] = float(
                    land_area_match.group(1)
                    .replace(",", "")
                )

            built_match = re.search(
                r"築年月\s+([0-9]{4})年",
                page_text
            )

            if built_match:

                property_data["築年"] = int(
                    built_match.group(1)
                )

            if "中古一戸建て" in page_text:

                property_data["物件種別"] = (
                    "中古戸建て"
                )

            elif "新築一戸建て" in page_text:

                property_data["物件種別"] = (
                    "新築戸建て"
                )

            elif "中古マンション" in page_text:

                property_data["物件種別"] = (
                    "中古マンション"
                )

            elif "新築マンション" in page_text:

                property_data["物件種別"] = (
                    "新築マンション"
                )

            manufacturer_names = [
                "積水ハウス",
                "大和ハウス",
                "ダイワハウス",
                "セキスイハイム",
                "一条工務店",
                "住友林業",
                "タマホーム",
                "ミサワホーム",
                "パナソニック ホームズ",
                "ヘーベルハウス"
            ]

            for manufacturer in manufacturer_names:

                if manufacturer in page_text:

                    property_data[
                        "住宅メーカー"
                    ] = manufacturer

                    break

            register_usage_event("url_fetch_success")
            url_fetch_succeeded = True

            st.success(
                "物件情報を取得しました。下の入力欄に反映しました。"
            )

            st.markdown(
                '<div class="property-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="property-title">'
                f'{property_data.get("物件名") or "物件情報"}'
                '</div>',
                unsafe_allow_html=True
            )

            if property_data.get("価格") is not None:

                st.markdown(
                    '<div class="property-price">'
                    f'{property_data["価格"]:,.0f}万円'
                    '</div>',
                    unsafe_allow_html=True
                )

            meta_parts = []

            if property_data.get("間取り"):
                meta_parts.append(
                    f'間取り：{property_data["間取り"]}'
                )

            if property_data.get("建物面積") is not None:
                meta_parts.append(
                    f'建物：{property_data["建物面積"]:,.2f}㎡'
                )

            if property_data.get("築年"):
                meta_parts.append(
                    f'築年：{property_data["築年"]}年'
                )

            if property_data.get("物件種別"):
                meta_parts.append(
                    f'種別：{property_data["物件種別"]}'
                )

            st.markdown(
                '<div class="property-meta">'
                +
                "<br>".join(meta_parts)
                +
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.warning(
                "goo住宅のページを読み込めませんでした。"
                f"ステータスコード：{response.status_code}"
            )

    except Exception as e:

        st.error(
            f"ページの読み込み中にエラーが発生しました：{e}"
        )



# =========================================================
# URL情報を入力欄へ反映
# =========================================================

if url_fetch_succeeded:
    if st.session_state.get("last_fetched_url") != property_url:
        fetched_address = property_data.get("住所") or ""
        st.session_state["pending_auto_address"] = fetched_address
        st.session_state["target_price"] = float(property_data.get("価格") or 0)

        fetched_type = property_data.get("物件種別")

        if fetched_type in [
            "中古マンション",
            "新築マンション",
            "中古戸建て",
            "新築戸建て",
        ]:
            st.session_state["target_type"] = fetched_type

        fetched_building_area = float(property_data.get("建物面積") or 0)
        fetched_land_area = float(property_data.get("土地面積") or 0)

        if fetched_type == "中古マンション":
            st.session_state["condo_area"] = fetched_building_area
        else:
            st.session_state["house_building_area"] = fetched_building_area
            st.session_state["house_land_area"] = fetched_land_area

        fetched_manufacturer = property_data.get("住宅メーカー")

        if fetched_manufacturer in [
            "積水ハウス",
            "大和ハウス",
            "ダイワハウス",
            "セキスイハイム",
            "一条工務店",
            "住友林業",
            "タマホーム",
            "ミサワホーム",
            "パナソニック ホームズ",
            "ヘーベルハウス",
            "その他",
            "不明",
        ]:
            st.session_state["target_manufacturer"] = fetched_manufacturer

        fetched_layout = property_data.get("間取り")

        if fetched_layout in [
            "1R", "1K", "1DK", "1LDK",
            "2K", "2DK", "2LDK",
            "3K", "3DK", "3LDK",
            "4K", "4DK", "4LDK", "5LDK以上",
        ]:
            st.session_state["layout_choice"] = fetched_layout

        fetched_age = property_data.get("築年")

        if fetched_age:
            st.session_state["target_age"] = int(fetched_age)

        st.session_state["last_fetched_url"] = property_url


# =========================================================
# 物件情報入力
# =========================================================

st.markdown('<div id="step1-input"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">✏️ 物件情報を入力</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    '住所は一覧から選択できます。入力後に「AIで価格をチェック」を押してください。'
    '</div>',
    unsafe_allow_html=True
)

with st.container(border=True):
    st.markdown("#### 基本情報")

    if "target_type" not in st.session_state:
        st.session_state["target_type"] = "中古マンション"

    target_type = st.selectbox(
        "物件種別",
        [
            "中古マンション", "中古戸建て",
            "新築マンション", "新築戸建て",
            "中古アパート・一棟収益物件", "新築アパート・一棟収益物件",
            "賃貸戸建て・借家", "賃貸アパート", "賃貸マンション",
        ],
        key="target_type",
    )

    if load_price_model(get_model_key(target_type)) is None:
        st.info("この分類の学習済みモデルはまだありません。取引データが十分そろい、精度検証に合格すると利用できます。")

    st.markdown("#### 住所を選択")

    # 住所選択は取引実績とは切り離し、デジタル庁ABRの全国町字マスタを使う。
    # これにより、過去取引が0件の町字でも最初から選択できる。
    api_key_ready = bool(get_mlit_api_key())
    if not api_key_ready:
        st.warning(
            "価格計算用の国土交通省APIキーが見つかりません。"
            "住所選択はできますが、成約実績・公示地価の取得にはAPIキーが必要です。"
        )

    pending_address = str(
        st.session_state.get("pending_auto_address", "")
    )

    prefecture_names = ["選択してください"] + list(PREFECTURE_CODES.keys())

    default_prefecture = None
    for prefecture_name in PREFECTURE_CODES:
        if pending_address.startswith(prefecture_name):
            default_prefecture = prefecture_name
            break

    if default_prefecture and "address_prefecture" not in st.session_state:
        st.session_state["address_prefecture"] = default_prefecture

    selected_prefecture = st.selectbox(
        "都道府県",
        prefecture_names,
        key="address_prefecture",
    )

    selected_area_code = PREFECTURE_CODES.get(selected_prefecture)

    if st.session_state.get("_last_address_prefecture") != selected_prefecture:
        st.session_state.pop("address_city", None)
        st.session_state.pop("address_district", None)
        st.session_state["_last_address_prefecture"] = selected_prefecture

    city_pairs = []
    address_master_error = None
    if selected_area_code:
        try:
            with st.spinner("全国住所マスタから市区町村を読み込んでいます..."):
                city_pairs = get_municipalities(selected_area_code)
        except Exception as e:
            address_master_error = e
            st.error(
                "全国住所マスタを取得できませんでした。"
                "ネット接続を確認して再読み込みしてください。"
            )
            st.caption(str(e))

    city_names = ["選択してください"] + [name for _, name in city_pairs]

    if pending_address and "address_city" not in st.session_state:
        for _, city_name in city_pairs:
            if city_name and city_name in pending_address:
                st.session_state["address_city"] = city_name
                break

    selected_city_name = st.selectbox(
        "市区町村",
        city_names,
        key="address_city",
        disabled=(not bool(city_pairs)),
    )

    selected_city_code = next(
        (code for code, name in city_pairs if name == selected_city_name),
        "",
    )

    if st.session_state.get("_last_address_city") != selected_city_name:
        st.session_state.pop("address_district", None)
        st.session_state["_last_address_city"] = selected_city_name

    town_records = []
    if selected_area_code and selected_city_code:
        try:
            with st.spinner("全国住所マスタから町名を読み込んでいます..."):
                town_records = get_towns(
                    selected_area_code,
                    selected_city_code,
                )
        except Exception as e:
            st.warning(f"町名一覧を取得できませんでした：{e}")

    district_names = [str(x.get("town_name") or "").strip() for x in town_records]
    district_names = sorted({x for x in district_names if x})

    if pending_address and "address_district" not in st.session_state:
        for district_name in district_names:
            if district_name and district_name in pending_address:
                st.session_state["address_district"] = district_name
                break

    # 町名は五十音の行ボタンと前後のページ送りだけで選択する。
    # 選択した町名は address_district に保存し、既存の査定処理に渡す。
    if st.session_state.get("_town_visible_city") != selected_city_code:
        st.session_state["town_visible_page"] = 0
        st.session_state["town_kana_group"] = "あ"
        st.session_state["_town_visible_city"] = selected_city_code

    selected_district_model = st.session_state.get("address_district", "")
    if selected_district_model not in district_names:
        selected_district_model = ""
        st.session_state["address_district"] = ""

    if district_names:
        st.markdown("**地区・町名を選ぶ**")
        st.caption("五十音の行を押すと町名が表示されます。「前へ」「次へ」で行をまたいで探せます。")
        group_keys = list(KANA_GROUPS)
        for start in (0, 5):
            button_cols = st.columns(5)
            for col, group in zip(button_cols, group_keys[start:start + 5]):
                with col:
                    st.button(
                        f"{group}行",
                        key=f"town_kana_button_{group}",
                        use_container_width=True,
                        type=("primary" if st.session_state.get("town_kana_group", "あ") == group else "secondary"),
                        on_click=_set_town_kana_group,
                        args=(group,),
                    )

        selected_kana_group = st.session_state.get("town_kana_group", "あ")
        if _get_town_kana_converter() is None:
            st.warning("五十音検索には pykakasi が必要です。コマンドプロンプトで py -m pip install pykakasi を実行してください。")
            visible_districts = sorted(district_names)
        else:
            visible_districts = sorted(
                (name for name in district_names if _town_name_kana_group(name) == selected_kana_group),
                key=lambda name: (_town_name_reading(name), name),
            )
            st.caption(f"{selected_kana_group}行：{len(visible_districts)}件")
            if not visible_districts:
                nearby = _town_nearby_candidates(district_names, selected_kana_group)
                if nearby:
                    st.info("この行は0件です。前後の行の町名を表示しています。")
                    visible_districts = nearby
                else:
                    st.info("該当する町名がありません。別の行を選んでください。")

        if visible_districts:
            st.markdown("**町名をタップして選択**")
            page_size = 12
            total_pages = (len(visible_districts) + page_size - 1) // page_size
            page = min(max(int(st.session_state.get("town_visible_page", 0)), 0), total_pages - 1)
            st.session_state["town_visible_page"] = page
            page_towns = visible_districts[page * page_size:(page + 1) * page_size]
            for start in range(0, len(page_towns), 2):
                cols = st.columns(2)
                for offset, name in enumerate(page_towns[start:start + 2]):
                    with cols[offset]:
                        st.button(
                            name,
                            key=f"town_visible_{selected_city_code}_{selected_kana_group}_{page}_{start + offset}",
                            use_container_width=True,
                            type=("primary" if selected_district_model == name else "secondary"),
                            on_click=_select_visible_town,
                            args=(name,),
                        )
        else:
            page_size, total_pages, page = 12, 0, 0

        previous_group = _adjacent_town_kana_group(district_names, selected_kana_group, -1)
        next_group = _adjacent_town_kana_group(district_names, selected_kana_group, 1)
        can_previous = page > 0 or previous_group is not None
        can_next = page < total_pages - 1 or next_group is not None
        prev_col, count_col, next_col = st.columns([1, 1.2, 1])
        with prev_col:
            prev_label = f"◀ 前へ（{previous_group}行）" if page == 0 and previous_group else "◀ 前へ"
            if st.button(prev_label, disabled=not can_previous, key="town_visible_prev", use_container_width=True):
                _move_town_page(-1, page, total_pages, district_names, "五十音から探す", selected_kana_group, page_size)
                st.rerun()
        with count_col:
            st.caption(f"{selected_kana_group}行　{page + 1 if total_pages else 0} / {total_pages}ページ")
        with next_col:
            next_label = f"次へ（{next_group}行）▶" if page >= total_pages - 1 and next_group else "次へ ▶"
            if st.button(next_label, disabled=not can_next, key="town_visible_next", use_container_width=True):
                _move_town_page(1, page, total_pages, district_names, "五十音から探す", selected_kana_group, page_size)
                st.rerun()

    selected_district_model = st.session_state.get("address_district", "")
    if selected_district_model not in district_names:
        selected_district_model = ""
    if selected_district_model:
        st.success(f"選択中の町名：{selected_district_model}")
    elif district_names:
        st.caption("町名が未選択です。上の候補から選んでください。")

    selected_district = (
        f"{selected_city_name} {selected_district_model}"
        if selected_city_name != "選択してください" and selected_district_model
        else selected_district_model
    )

    selected_town_lat = None
    selected_town_lon = None
    for item in town_records:
        if str(item.get("town_name") or "").strip() == selected_district_model:
            selected_town_lat = item.get("lat")
            selected_town_lon = item.get("lon")
            break

    town_point_map = make_town_point_map(town_records)

    target_address = "".join(
        part
        for part in [
            selected_prefecture if selected_prefecture != "選択してください" else "",
            selected_city_name if selected_city_name != "選択してください" else "",
            selected_district_model,
        ]
        if part
    )

    if target_address:
        st.caption(f"選択した住所：{target_address}")
    col1, col2 = st.columns(2)

    with col1:
        target_price = st.number_input(
            "月額家賃（万円）" if target_type.startswith("賃貸") else "販売価格（万円）",
            min_value=0.0,
            step=0.5 if target_type.startswith("賃貸") else 100.0,
            value=None,
            key="target_price",
        )
        target_price = target_price or 0.0

    with col2:
        if target_type in ["新築マンション", "新築戸建て", "新築アパート・一棟収益物件"]:
            target_age = datetime.now().year
            st.text_input(
                "築年",
                value=f"{datetime.now().year}年（新築）",
                disabled=True,
            )
        else:
            year_options = list(
                range(datetime.now().year, 1899, -1)
            )

            if "target_age" not in st.session_state:
                st.session_state["target_age"] = datetime.now().year

            target_age = st.selectbox(
                "築年（西暦）",
                year_options,
                key="target_age",
            )

    st.markdown("#### 面積")


    area_unit = st.selectbox(
        "面積の単位",
        ["㎡", "坪"],
        key="area_unit",
    )

    target_condo_area = None
    target_land_area = None
    target_building_area = None

    if "マンション" in target_type or target_type.startswith("賃貸"):
        condo_area_input = st.number_input(
            f"{'床面積' if target_type.startswith('賃貸') and 'マンション' not in target_type else '専有面積'}（{area_unit}）",
            min_value=0.0,
            step=1.0,
            value=None,
            key="condo_area_input",
        )

        condo_area_value = condo_area_input or 0.0

        if area_unit == "坪":
            target_condo_area = (
                condo_area_value * 3.305785
            )
        else:
            target_condo_area = condo_area_value

        target_area = target_condo_area

    else:
        area_col1, area_col2 = st.columns(2)

        with area_col1:
            land_area_input = st.number_input(
                f"土地面積（{area_unit}）",
                min_value=0.0,
                step=1.0,
                value=None,
                key="house_land_area_input",
            )

        with area_col2:
            building_area_input = st.number_input(
                f"建物面積（{area_unit}）",
                min_value=0.0,
                step=1.0,
                value=None,
                key="house_building_area_input",
            )

        land_area_value = land_area_input or 0.0
        building_area_value = building_area_input or 0.0

        if area_unit == "坪":
            target_land_area = (
                land_area_value * 3.305785
            )
            target_building_area = (
                building_area_value * 3.305785
            )
        else:
            target_land_area = land_area_value
            target_building_area = building_area_value

        target_area = target_building_area
    

    st.markdown("#### 物件の詳細")

    detail_col1, detail_col2 = st.columns(2)

    structure_options = [
        "不明",
        "木造",
        "鉄骨造",
        "軽量鉄骨造",
        "RC",
        "SRC",
        "その他",
    ]

    with detail_col1:
        target_structure = st.selectbox(
            "構造",
            structure_options,
            key="target_structure",
        )

    target_city_planning = "不明"

    layout_options = [
        "選択してください",
        "1R", "1K", "1DK", "1LDK",
        "2K", "2DK", "2LDK",
        "3K", "3DK", "3LDK",
        "4K", "4DK", "4LDK",
        "5LDK以上",
    ]

    if "マンション" in target_type or target_type.startswith("賃貸"):
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            target_layout = st.selectbox(
                "間取り",
                layout_options,
                key="layout_choice",
            )

        with detail_col2:
            target_renovation = st.selectbox(
                "改装状況",
                ["不明", "改装済み", "未改装"],
                key="target_renovation",
            )
    else:
        target_layout = "選択してください"
        target_renovation = "不明"

    target_total_floors = None
    target_unit_floor = None

    if "マンション" in target_type or target_type.startswith("賃貸"):
        floor_col1, floor_col2 = st.columns(2)

        with floor_col1:
            target_total_floors = st.number_input(
                "建物の総階数（任意）",
                min_value=1,
                max_value=100,
                value=None,
                step=1,
                key="target_total_floors",
            )

        with floor_col2:
            target_unit_floor = st.number_input(
                "所在階（任意）",
                min_value=1,
                max_value=100,
                value=None,
                step=1,
                key="target_unit_floor",
            )

        if (
            target_total_floors is not None
            and target_unit_floor is not None
            and target_unit_floor > target_total_floors
        ):
            st.warning("所在階が建物の総階数を超えています。")

    manufacturer_options = [
        "選択してください",
        "積水ハウス",
        "大和ハウス",
        "ダイワハウス",
        "セキスイハイム",
        "一条工務店",
        "住友林業",
        "タマホーム",
        "ミサワホーム",
        "パナソニック ホームズ",
        "ヘーベルハウス",
        "その他",
        "不明",
    ]

    target_manufacturer = st.selectbox(
        "住宅メーカー（参考情報）",
        manufacturer_options,
        key="target_manufacturer",
    )

station_minutes_input = st.number_input(
    "最寄り駅まで徒歩（分・任意）",
    min_value=0.0,
    max_value=120.0,
    value=None,
    step=1.0,
    key="target_station_minutes",
)

target_station_minutes = station_minutes_input or 0.0

target_annual_income_yen = None
if "一棟収益物件" in target_type:
    annual_income_man = st.number_input(
        "年間家賃収入（万円・任意）", min_value=0.0, value=0.0, step=10.0,
        help="不明な場合は0のままで構いません。"
    )
    if annual_income_man > 0:
        target_annual_income_yen = annual_income_man * 10000.0

# =========================================================
# 地域相場・地価・駅距離
# =========================================================


st.caption(
    "不明なら0のままでOKです。"
    "入力した場合は駅距離も参考価格に反映します。"
)


# =========================================================
# 物件評価
# =========================================================

st.markdown('<div id="step2-check"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">🤖 AI価格チェック</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    '国土交通省の取引価格データで学習したAIモデルから参考価格を予測します。'
    '</div>',
    unsafe_allow_html=True
)
show_affiliate_precheck_banner()
evaluate_button = st.button(
    "🤖 AIで家賃をチェック" if target_type.startswith("賃貸") else "🤖 AIで価格をチェック",
    type="primary",
    use_container_width=True,
)

if evaluate_button:
    # 全国の売買6分類・賃貸3分類をすべて処理可能にする。
    # 未学習の分類は下のモデル読み込み時に正しく案内する。
    if get_model_key(target_type) is None:
        st.warning("未対応の物件種別です。")
        st.stop()

    if (
        selected_prefecture == "選択してください"
        or selected_city_name == "選択してください"
        or not selected_district_model
    ):
        st.warning("都道府県・市区町村・地区を選択してください。")
        st.stop()

    if target_price <= 0:
        st.warning("月額家賃を入力してください。" if target_type.startswith("賃貸") else "販売価格を入力してください。")
        st.stop()

    if target_type in ("中古マンション", "新築マンション") or target_type.startswith("賃貸"):
        if not target_condo_area or target_condo_area <= 0:
            st.warning("面積を入力してください。")
            st.stop()
    else:
        if not target_land_area or target_land_area <= 0:
            st.warning("土地面積を入力してください。")
            st.stop()

        if not target_building_area or target_building_area <= 0:
            st.warning("建物面積を入力してください。")
            st.stop()

    register_usage_event("property_evaluation")

    try:
        with st.spinner("AIが参考価格を計算しています..."):
            ai_land_context = {}

            try:
                ai_land_context = (
                    official_location_context(
                        api_key=get_mlit_api_key(),
                        area_code=str(selected_area_code),
                        city_code=str(selected_city_code),
                        district_name=selected_district,
                    )
                    or {}
                )
            except Exception:
                ai_land_context = {}

            ai_local_land = ai_land_context.get(
                "land_price_yen_per_sqm"
            )
            ai_city_land = ai_land_context.get(
                "city_land_price_yen_per_sqm"
            )

            if (
                ai_local_land
                and ai_city_land
                and ai_local_land > 0
                and ai_city_land > 0
            ):
                ai_land_ratio = (
                    float(ai_local_land)
                    / float(ai_city_land)
                )
            else:
                ai_land_ratio = None

            if ai_land_context.get("scope") == "地区・町名":
                ai_land_match_level = 2.0
            elif ai_city_land:
                ai_land_match_level = 1.0
            else:
                ai_land_match_level = 0.0

            estimated_price, model_bundle = predict_ai_reference_price(
                property_type=target_type,
                municipality_code=selected_city_code,
                district_name=selected_district_model,
                building_year=target_age,
                structure=target_structure,
                city_planning=target_city_planning,
                floor_plan=target_layout,
                renovation=target_renovation,
                condo_area=target_condo_area,
                land_area=target_land_area,
                building_area=target_building_area,
                land_price_per_sqm=ai_local_land,
                city_land_price_per_sqm=ai_city_land,
                land_price_ratio=ai_land_ratio,
                land_price_match_level=ai_land_match_level,
                annual_income_yen=target_annual_income_yen,
                station_minutes=target_station_minutes,
            )
    except Exception as e:
        st.error(f"AI価格予測中にエラーが発生しました：{e}")
        st.stop()


    market_context_result = {}

    try:
        if target_type.startswith("賃貸"):
            market_context_result = {"adjusted_price_man_yen": estimated_price}
        else:
            market_context_result = build_market_context(
            api_key=get_mlit_api_key(),
            base_ai_price_man_yen=estimated_price,
            property_type=target_type,
            area_code=selected_area_code,
            city_code=selected_city_code,
            district_name=selected_district,
            building_year=target_age,
            station_minutes=target_station_minutes,
            condo_area=target_condo_area,
            land_area=target_land_area,
            building_area=target_building_area,
            target_lat=selected_town_lat,
            target_lon=selected_town_lon,
            town_points=town_point_map,
            comparable_weight_max=model_bundle.get("comparable_weight_max", 0.0),
        )

        adjusted_market_price = (
            market_context_result.get(
                "adjusted_price_man_yen"
            )
        )

        if (
            adjusted_market_price
            and adjusted_market_price > 0
        ):
            estimated_price = float(
                adjusted_market_price
            )

    except Exception as market_error:
        market_context_result = {
            "error": str(
                market_error
            )
        }

    if estimated_price <= 0:
        st.warning("参考価格を計算できませんでした。")
        st.stop()

    price_difference = target_price - estimated_price
    target_ratio = target_price / estimated_price

    st.session_state["last_estimated_price_man_yen"] = float(estimated_price)
    st.session_state["last_evaluated_address"] = target_address
    st.session_state["last_property_snapshot"] = {
        "property_type": target_type,
        "municipality_code": selected_city_code,
        "district_name": selected_district_model,
        "building_year": target_age,
        "structure": target_structure,
        "city_planning": target_city_planning,
        "floor_plan": target_layout,
        "renovation": target_renovation,
        "condo_area": target_condo_area or "",
        "land_area": target_land_area or "",
        "building_area": target_building_area or "",
        "station_minutes": target_station_minutes or "",
        "total_floors": target_total_floors or "",
        "unit_floor": target_unit_floor or "",
    }

    if target_ratio <= 0.90:
        evaluation_label = "AI基準で割安の可能性"
    elif target_ratio <= 1.10:
        evaluation_label = "AI基準で相場に近い"
    else:
        evaluation_label = "割高の可能性"

    # 鑑定方式の参考評価（既存AIの価格・判定を変更しない）
    with st.expander("🏠 適正価格の根拠を詳しく見る（鑑定方式・試験版）", expanded=False):
        st.caption("既存AIの推定値を基準に、取引事例・原価・収益の観点を整理します。正式な不動産鑑定評価ではありません。")
        st.metric("既存AIの推定参考価格" if not target_type.startswith("賃貸") else "既存AIの推定参考家賃", f"{estimated_price:,.1f}万円" + ("/月" if target_type.startswith("賃貸") else ""))
        if target_type.startswith("賃貸"):
            st.info("賃貸は売買価格ではなく周辺賃料・立地・面積を重視します。売買の原価法・収益還元法は適用しません。")
        else:
            st.markdown("**① 取引事例比較法**")
            st.write("既存AIの学習済み取引価格を参考にしています。近隣の個別成約事例とその補正は、別途確認が必要です。")
            _col_a, _col_b = st.columns(2)
            with _col_a:
                _land_unit = st.number_input("土地の参考単価（万円/㎡・任意）", min_value=0.0, value=0.0, step=1.0, key="appraisal_land_unit")
                _build_unit = st.number_input("建物の再調達単価（万円/㎡・任意）", min_value=0.0, value=0.0, step=1.0, key="appraisal_build_unit")
            with _col_b:
                _useful_life = st.number_input("建物の想定耐用年数（年・仮定）", min_value=1, max_value=150, value=50, key="appraisal_life")
                _repair = st.number_input("必要な修繕・解体費（万円・任意）", min_value=0.0, value=0.0, step=10.0, key="appraisal_repair")
            st.markdown("**② 原価法（土地＋建物）**")
            _land_area = float(target_land_area or 0)
            _build_area = float(target_building_area or 0)
            _age = max(0.0, float(target_age or 0))
            if _land_unit > 0 and _land_area > 0 and _build_unit > 0 and _build_area > 0:
                _land_value = _land_unit * _land_area
                _building_value = _build_unit * _build_area * max(0.0, 1.0 - _age / _useful_life)
                _cost_value = max(0.0, _land_value + _building_value - _repair)
                st.metric("原価法による参考価格", f"{_cost_value:,.0f}万円")
                st.caption(f"土地 {_land_value:,.0f}万円 ＋ 建物残価 {_building_value:,.0f}万円 − 修繕等 {_repair:,.0f}万円。年数比例の簡易減価であり、実地調査の代わりにはなりません。")
            else:
                st.info("土地・建物の面積と参考単価が揃った場合に計算します。単価が未確認のまま架空の価格は表示しません。")
            st.markdown("**③ 収益還元法（収益物件向け）**")
            if "収益" in target_type or "アパート" in target_type:
                _rent = st.number_input("年間純収益（万円・任意）", min_value=0.0, value=0.0, step=10.0, key="appraisal_noi")
                _cap = st.number_input("還元利回り（%・任意）", min_value=0.1, max_value=50.0, value=6.0, step=0.5, key="appraisal_cap")
                if _rent > 0:
                    st.metric("収益還元法による参考価格", f"{_rent / (_cap / 100):,.0f}万円")
                    st.caption("年間純収益 ÷ 還元利回り。空室・維持費・修繕費を反映した純収益を入力してください。")
                else:
                    st.info("年間純収益を入力すると参考価格を表示します。")
            else:
                st.caption("通常の自宅用物件には原則として収益還元法を適用しません。")
            st.markdown("**④ お買い得度を判断する前に**")
            st.write(f"売出価格 {target_price:,.0f}万円 ／ 既存AIとの差 {estimated_price - target_price:+,.0f}万円")
            st.warning("価格差だけでお買い得とは断定できません。接道・再建築可否・劣化・事故歴・災害リスク・権利関係の確認が必要です。原価法と収益還元法の参考値は既存AIの予測や判定には自動合成しません。")

    st.markdown(
        '<div id="step3-result"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">📋 チェック結果</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="result-card">',
        unsafe_allow_html=True,
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.markdown(
            ('<div class="result-label">現在の月額家賃</div>' if target_type.startswith('賃貸') else '<div class="result-label">現在の販売価格</div>'),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="result-price">'
            + (f'{target_price:,.1f}万円/月' if target_type.startswith('賃貸') else f'{target_price:,.0f}万円')
            + '</div>',
            unsafe_allow_html=True,
        )

    with result_col2:
        st.markdown(
            ('<div class="result-label">AI参考家賃</div>' if target_type.startswith('賃貸') else '<div class="result-label">AI参考価格</div>'),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="result-price">'
            + (f'{estimated_price:,.1f}万円/月' if target_type.startswith('賃貸') else f'{estimated_price:,.0f}万円')
            + '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "AI参考家賃との差額" if target_type.startswith("賃貸") else "AI参考価格との差額",
            f"{price_difference:+,.1f}万円/月" if target_type.startswith("賃貸") else f"{price_difference:+,.0f}万円",
        )

    with result_col2:
        st.metric(
            "家賃評価" if target_type.startswith("賃貸") else "価格評価",
            evaluation_label,
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


    # V3/V4の研究から、入力条件による確認ポイントを表示。
    # これは誤差幅の統計的な推定ではなく、AIの予測値を変更しません。
    if not target_type.startswith("賃貸"):
        caution_items = assess_property_cautions(
            asking_price_man=target_price,
            building_year=target_age,
            land_area_sqm=target_land_area,
            property_type=target_type,
            station_minutes=target_station_minutes,
        )
        with st.expander("🔎 この物件で特に確認したいこと", expanded=bool(caution_items)):
            if caution_items:
                for item in caution_items:
                    st.markdown(f"**{item['title']}** — {item['description']}")
            else:
                st.write("大きな追加注意項目はありません。ただし、現地状況や権利関係の確認は必要です。")
            st.caption("研究で誤差が大きかった条件を参考にした注意喚起です。誤差の確率・信頼区間を示すものではありません。")

    st.markdown(
        "#### 📍 地域相場・地価・駅距離"
    )

    comparable_info = (
        market_context_result.get(
            "comparable"
        )
        or {}
    )

    official_info = (
        market_context_result.get(
            "official"
        )
        or {}
    )

    market_col1, market_col2, market_col3 = (
        st.columns(3)
    )

    with market_col1:
        comparable_estimate_yen = (
            comparable_info.get(
                "estimate_yen"
            )
        )

        if comparable_estimate_yen:
            comparable_text = (
                f"{comparable_estimate_yen / 10000:,.0f}万円"
            )
        else:
            comparable_text = "データ不足"

        st.metric(
            "近隣成約ベース推定価格",
            comparable_text,
        )

        if comparable_info:
            confidence = float(
                comparable_info.get("confidence", 0.0)
                or 0.0
            )
            st.caption(
                f"{comparable_info.get('scope', '地域')}・"
                f"{comparable_info.get('count', 0)}件を参考 / "
                f"信頼度 {confidence * 100:.0f}%"
            )

    with market_col2:
        land_price = (
            official_info.get(
                "land_price_yen_per_sqm"
            )
        )

        if land_price:
            land_price_text = (
                f"{land_price:,.0f}円/㎡"
            )
        else:
            land_price_text = (
                "データ不足"
            )

        st.metric(
            "公示地価・基準地価",
            land_price_text,
        )

        if official_info.get(
            "year"
        ):
            st.caption(
                f"{official_info.get('year')}年・"
                f"{official_info.get('scope', '地域')}の目安"
            )

    with market_col3:
        if (
            target_station_minutes
            and target_station_minutes > 0
        ):
            st.metric(
                "最寄り駅まで",
                f"徒歩 {target_station_minutes:.0f}分",
            )

        else:
            station_distance = (
                official_info.get(
                    "station_distance_m"
                )
            )

            if station_distance:
                station_text = (
                    f"{station_distance:,.0f}m"
                )
            else:
                station_text = (
                    "データ不足"
                )

            st.metric(
                "地域の交通距離目安",
                station_text,
            )

    base_ai_price = (
        market_context_result.get(
            "base_ai_price_man_yen"
        )
    )

    if base_ai_price:
        st.caption(
            "AI単体の参考価格："
            f"{base_ai_price:,.0f}万円"
        )

        st.caption(
            "最終価格はAIを基準にし、過去の検証で有効だった範囲内でのみ近隣成約を反映します。"
            f"今回の近隣成約反映率：{market_context_result.get('comparable_weight', 0.0)*100:.1f}%"
        )

    if market_context_result.get(
        "error"
    ):
        st.caption(
            "地域相場データを取得できなかったため、"
            "今回はAIモデル単体の価格を使用しました。"
        )

    metrics = model_bundle.get("metrics", {})
    training_rows = model_bundle.get("training_rows")

    st.markdown("#### 🧠 AIモデル情報")
    if model_bundle.get("v7_active"):
        st.info("中古戸建てには研究候補V7を試験適用中です。2024年の検証値は本番精度を保証するものではありません。")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        median_error = metrics.get("median_ape_pct")
        st.metric(
            "テスト時の中央誤差率",
            f"{median_error:.1f}%" if median_error is not None else "—",
        )

    with metric_col2:
        within_20 = metrics.get("within_20pct_pct")
        st.metric(
            "±20%以内だった割合",
            f"{within_20:.1f}%" if within_20 is not None else "—",
        )

    with metric_col3:
        st.metric(
            "学習件数",
            f"{training_rows:,}件" if training_rows else "—",
        )

    if target_ratio <= 0.90:
        difference_pct = (1 - target_ratio) * 100
        evaluation_reason = (
            f"AI参考価格は約{estimated_price:,.0f}万円です。"
            f"現在の販売価格はAI参考価格より"
            f"{difference_pct:.1f}%低い水準です。"
        )
    elif target_ratio <= 1.10:
        evaluation_reason = (
            f"AI参考価格は約{estimated_price:,.0f}万円です。"
            f"現在の販売価格はAI参考価格に対して"
            f"{target_ratio * 100:.1f}%の水準です。"
        )
    else:
        difference_pct = (target_ratio - 1) * 100
        evaluation_reason = (
            f"AI参考価格は約{estimated_price:,.0f}万円です。"
            f"現在の販売価格はAI参考価格より"
            f"{difference_pct:.1f}%高い水準です。"
        )

    st.markdown("#### 💡 評価の内容")
    st.info(evaluation_reason)

    st.caption(
        "※AI参考価格は統計的な予測値です。"
        "個別物件の状態、眺望、リフォーム、管理状態、接道、"
        "特殊事情などを完全には反映できない場合があります。"
    )

    show_video_monetization_slot()
    show_affiliate_slot()

st.caption(
    "出典：国土交通省「不動産情報ライブラリ」。"
    "当サービスでは取得・公開された取引情報を学習・加工し、独自のAI参考価格を算出しています。"
)

st.caption(
    "このサービスは、国土交通省の不動産情報ライブラリのAPI機能を使用していますが、"
    "提供情報の最新性、正確性、完全性等が保証されたものではありません。"
)



# =========================================================
# 公開サイトの説明・研究の透明性
# =========================================================
st.markdown('<div id="service-guide"></div>', unsafe_allow_html=True)
st.markdown("## このサービスで分かること")
st.markdown(
    """
    <div class="footer-feature">
      <strong>参考価格を知る</strong> — 学習済みAIが物件条件から参考価格を計算します。<br>
      <strong>売出価格と比較する</strong> — 入力した販売価格との差を可視化します。<br>
      <strong>見落としを減らす</strong> — 築古・低価格・広い土地など、注意したい条件を案内します。<br>
      <small>※表示価格は統計的な目安であり、正式な鑑定評価・買取保証ではありません。</small>
    </div>
    """,
    unsafe_allow_html=True,
)
with st.expander("🧪 AI研究V4の進捗と公開版の違い"):
    st.write(
        "2024年の研究比較では、V3の比較対象モデルの誤差中央値22.70%に対し、"
        "V4の追加特徴量モデルは20.88%でした。"
    )
    st.warning(
        "これは2024年のモデル選定用データにおける研究結果です。"
        "2025年の最終検証と公開用モデルの準備が終わるまで、"
        "V4の数値を現在の公開サイトの精度として表示することはできません。"
        "このサイトの予測は既存の学習済みモデルを使用します。"
    )
with st.expander("❓ よくある質問"):
    st.markdown(
        """
        **AI参考価格は売れる価格ですか？**

        いいえ。入力条件に基づく統計的な参考価格です。実際の成約価格や買取額を保証しません。

        **「割安」と表示されたら買ってもよいですか？**

        いいえ。修繕費、再建築可否、接道、災害リスク、権利関係などを別途確認してください。

        **V4の研究結果はもう反映されていますか？**

        いいえ。2025年の検証結果と公開用モデルの準備が整うまで、自動では切り替わりません。
        """
    )

# =========================================================
# お問い合わせ・改善フィードバック
# =========================================================
st.divider()
st.markdown("### 💬 お問い合わせ・改善フィードバック")
st.caption(
    "一般のご意見・価格差の報告・不具合・法人や提携のご相談まで、"
    "窓口は1つにまとめています。内容は裏側で分類し、学習候補は管理者確認後にのみ利用します。"
)

with st.form("feedback_form", clear_on_submit=True):
    fb_category = st.selectbox(
        "お問い合わせ種別",
        [
            "価格精度について",
            "住所・地域データについて",
            "不具合・エラー",
            "改善要望",
            "利用相談",
            "法人・提携",
            "広告・取材",
            "その他",
        ],
    )
    default_fb_address = st.session_state.get("last_evaluated_address", target_address if 'target_address' in globals() else "")
    fb_address = st.text_input("対象住所（任意）", value=str(default_fb_address or ""))

    c1, c2 = st.columns(2)
    with c1:
        last_ai = st.session_state.get("last_estimated_price_man_yen")
        fb_ai_price = st.number_input(
            "このサイトで表示された参考価格（万円・任意）",
            min_value=0.0,
            value=float(last_ai) if last_ai else None,
            step=100.0,
        )
    with c2:
        fb_actual_price = st.number_input(
            "実際に分かっている価格（万円・任意）",
            min_value=0.0,
            value=None,
            step=100.0,
        )

    fb_basis = st.selectbox(
        "実際の価格の根拠（任意）",
        [
            "不明・感想",
            "実際の成約価格",
            "売買契約書・成約資料",
            "査定書・業者資料",
            "売出価格・広告価格",
            "その他",
        ],
    )
    fb_actual_period = st.text_input(
        "実際の価格の時期（学習候補にする場合）",
        placeholder="例：2026年第2四半期",
    )
    fb_message = st.text_area(
        "内容",
        placeholder="価格が違う、住所がない、改善してほしい点、法人相談などをご記入ください。",
        height=140,
    )

    c3, c4 = st.columns(2)
    with c3:
        fb_name = st.text_input("お名前（任意）")
        fb_company = st.text_input("会社名（任意）")
    with c4:
        fb_email = st.text_input("メールアドレス（返信希望の場合）")
        fb_reply = st.checkbox("返信を希望する")

    fb_submit = st.form_submit_button("送信する", use_container_width=True)

if fb_submit:
    if not str(fb_message or "").strip():
        st.warning("内容を入力してください。")
    elif fb_reply and not str(fb_email or "").strip():
        st.warning("返信を希望する場合はメールアドレスを入力してください。")
    else:
        auto = classify_feedback(
            fb_category,
            fb_message,
            fb_actual_price,
            fb_basis,
            fb_actual_period,
        )
        result = save_feedback(
            {
                "category": fb_category,
                "priority": auto["priority"],
                "address": fb_address,
                "ai_price_man_yen": fb_ai_price or "",
                "actual_price_man_yen": fb_actual_price or "",
                "actual_price_basis": fb_basis,
                "message": fb_message,
                "name": fb_name,
                "email": fb_email,
                "company": fb_company,
                "reply_requested": "はい" if fb_reply else "いいえ",
                **(st.session_state.get("last_property_snapshot", {}) or {}),
                "actual_price_period": fb_actual_period,
                "learning_candidate": "候補" if auto["learning_candidate"] else "対象外",
                "review_status": auto["review_status"],
            }
        )
        st.success("送信しました。ありがとうございます。")
        if auto["learning_candidate"]:
            st.caption(
                "実価格の根拠があるため学習候補として記録しました。"
                "自動学習には入れず、管理者確認後にのみ利用します。"
            )

# 管理者画面は ?admin=1 を付けたときだけ表示する。
try:
    admin_mode = str(st.query_params.get("admin", "")) == "1"
except Exception:
    admin_mode = False

if admin_mode:
    st.divider()
    st.markdown("### 🔐 管理者：お問い合わせ確認")
    try:
        configured_admin_password = str(st.secrets.get("ADMIN_PASSWORD", "")).strip()
    except Exception:
        configured_admin_password = os.environ.get("ADMIN_PASSWORD", "").strip()

    if not configured_admin_password:
        st.warning("ADMIN_PASSWORD が未設定のため管理画面は無効です。")
    else:
        entered_admin_password = st.text_input("管理者パスワード", type="password")
        if entered_admin_password == configured_admin_password:
            feedback_df = load_feedback()
            if feedback_df.empty:
                st.info("お問い合わせはまだありません。")
            else:
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    category_filter = st.selectbox(
                        "種別で絞り込み",
                        ["すべて"] + sorted(feedback_df["category"].dropna().unique().tolist()),
                    )
                with filter_col2:
                    status_filter = st.selectbox(
                        "対応状況で絞り込み",
                        ["すべて"] + sorted(feedback_df["review_status"].dropna().unique().tolist()),
                    )

                shown = feedback_df.copy()
                if category_filter != "すべて":
                    shown = shown[shown["category"] == category_filter]
                if status_filter != "すべて":
                    shown = shown[shown["review_status"] == status_filter]

                st.dataframe(
                    shown.sort_values("submitted_at", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "お問い合わせCSVをダウンロード",
                    feedback_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="feedback.csv",
                    mime="text/csv",
                )

                st.markdown("#### 対応状況を更新")
                selected_record_id = st.selectbox(
                    "問い合わせID",
                    feedback_df["record_id"].astype(str).tolist(),
                )
                new_review_status = st.selectbox(
                    "新しい状態",
                    ["未対応", "要確認", "対応中", "学習承認", "学習不可", "返信済み", "完了"],
                )
                if st.button("状態を更新"):
                    if update_review_status(selected_record_id, new_review_status):
                        st.success("更新しました。")
                    else:
                        st.error("更新できませんでした。")

                if st.button("学習承認済みデータを書き出す"):
                    exported = export_approved_training_candidates()
                    if exported:
                        st.success(f"学習用CSVを作成しました：{exported}")
                        st.caption("次回 py train_model.py 実行時に自動で読み込まれます。")
                    else:
                        st.info("学習承認済みの候補データはありません。")
        elif entered_admin_password:
            st.error("パスワードが違います。")

# =========================================================
# 公開サイト用フッター・規約
# =========================================================
st.divider()

with st.expander("📜 利用規約"):
    st.markdown("""
### 利用規約

本サービス「お買い得物件チェッカー」は、入力された物件情報や比較データをもとに、参考情報を提供するサービスです。

- 本サービスの判定・参考価格は、実際の査定価格、成約価格、将来の資産価値を保証するものではありません。
- 不動産の購入・売却・投資等の最終判断は、利用者ご自身の責任で行ってください。
- 本サービスの画面、文章、ロゴ、画像、プログラムその他のコンテンツについて、権利者の許可なく複製、転載、再配布、販売、改変して公開することを禁止します。
- 本サービスの内容は、予告なく変更・停止する場合があります。
- 法令上必要な事項や正式な事業者情報は、サービス公開・運営開始時に追記します。
""")

with st.expander("🔒 プライバシーポリシー"):
    st.markdown("""
### プライバシーポリシー

本サービスでは、サービス提供・改善に必要な範囲で、利用者が入力またはアップロードした情報を取り扱う場合があります。

- 入力・アップロードされた情報は、価格チェック等の機能を提供する目的で利用します。
- A8.netのアフィリエイト広告を掲載しています。広告画像の読み込みやリンクのクリック時に、広告配信・成果計測のため外部サービスへ通信が発生する場合があります。
- 個人情報を取り扱う機能、アクセス解析、広告、アフィリエイト、決済、会員登録等を導入する場合は、利用目的や第三者提供等について必要な内容を明示します。
- パスワード、本人確認書類、不要な個人情報など、価格チェックに必要のない機密情報は入力・アップロードしないでください。
- 正式公開時には、実際に利用する外部サービスや保存方法に合わせて本ポリシーを更新します。
""")

st.markdown(
    """
    <div style="text-align:center; padding:1.5rem 0 2.5rem 0; color:#64748b; font-size:0.88rem; line-height:1.8;">
        <strong>お買い得物件チェッカー</strong><br>
        © 2026 お買い得物件チェッカー. All Rights Reserved.<br>
        無断転載・複製・再配布を禁止します。
    </div>
    """,
    unsafe_allow_html=True,
)
