from __future__ import annotations

import io
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data" / "address_master"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ABR_PREF_URL = (
    "https://data.address-br.digital.go.jp/mt_town_fullset/pref/"
    "mt_town_fullset_pref{pref_code}.csv.zip"
)
REQUEST_TIMEOUT = 60


def _norm_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _digits(value) -> str:
    return re.sub(r"\D", "", _norm_text(value))


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        hit = lower_map.get(name.lower())
        if hit is not None:
            return hit
    return None


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    last_error = None
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"住所マスタCSVを読み込めませんでした: {last_error}")


def _read_zip_bytes(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError("住所マスタZIPにCSVがありません。")
        with zf.open(names[0]) as f:
            return _read_csv_bytes(f.read())


def _download_prefecture(pref_code: str) -> bytes:
    url = ABR_PREF_URL.format(pref_code=str(pref_code).zfill(2))
    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "PropertyPriceChecker/1.0"},
    )
    response.raise_for_status()
    return response.content


def _load_raw_prefecture(pref_code: str) -> pd.DataFrame:
    pref_code = str(pref_code).zfill(2)
    zip_path = CACHE_DIR / f"mt_town_fullset_pref{pref_code}.csv.zip"
    csv_path = CACHE_DIR / f"mt_town_fullset_pref{pref_code}.csv"

    if csv_path.exists():
        return pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)

    if zip_path.exists():
        return _read_zip_bytes(zip_path.read_bytes())

    raw = _download_prefecture(pref_code)
    zip_path.write_bytes(raw)
    return _read_zip_bytes(raw)


def _canonicalize(raw: pd.DataFrame, pref_code: str) -> pd.DataFrame:
    base_columns = [
        "prefecture_code",
        "prefecture_name",
        "municipality_code",
        "municipality_name",
        "town_name",
        "lat",
        "lon",
    ]

    if raw.empty:
        return pd.DataFrame(columns=base_columns)

    lg_col = _first_existing(
        raw,
        [
            "lg_code",
            "local_government_code",
            "市区町村コード",
            "全国地方公共団体コード",
        ],
    )
    pref_col = _first_existing(raw, ["pref", "pref_name", "都道府県名"])
    county_col = _first_existing(raw, ["county", "county_name", "郡名", "郡"])
    city_col = _first_existing(raw, ["city", "city_name", "市区町村名", "市町村名"])
    ward_col = _first_existing(raw, ["ward", "ward_name", "区名", "政令市区名"])
    oaza_col = _first_existing(
        raw,
        [
            "oaza_cho",
            "oaza_cho_name",
            "大字・町名",
            "大字町名",
            "町字名",
            "町名",
        ],
    )
    chome_col = _first_existing(raw, ["chome", "chome_name", "丁目名", "丁目"])
    koaza_col = _first_existing(raw, ["koaza", "koaza_name", "小字名", "小字"])
    lat_col = _first_existing(
        raw,
        ["rep_lat", "latitude", "lat", "代表点緯度", "緯度"],
    )
    lon_col = _first_existing(
        raw,
        ["rep_lon", "longitude", "lon", "lng", "代表点経度", "経度"],
    )

    if lg_col is None:
        raise RuntimeError("住所マスタに市区町村コード列が見つかりません。")

    def val(row, col):
        return _norm_text(row.get(col)) if col else ""

    rows = []

    for _, row in raw.iterrows():
        lg_digits = _digits(row.get(lg_col))
        if len(lg_digits) < 5:
            continue

        municipality_code = lg_digits[:5]

        county = val(row, county_col)
        city = val(row, city_col)
        ward = val(row, ward_col)

        municipality_name = "".join(x for x in [county, city, ward] if x)
        if not municipality_name:
            municipality_name = city or ward
        if not municipality_name:
            continue

        oaza = val(row, oaza_col)
        chome = val(row, chome_col)
        koaza = val(row, koaza_col)
        town_name = "".join(x for x in [oaza, chome, koaza] if x)
        if not town_name:
            continue

        lat = (
            pd.to_numeric(row.get(lat_col), errors="coerce")
            if lat_col
            else float("nan")
        )
        lon = (
            pd.to_numeric(row.get(lon_col), errors="coerce")
            if lon_col
            else float("nan")
        )

        rows.append(
            {
                "prefecture_code": str(pref_code).zfill(2),
                "prefecture_name": val(row, pref_col),
                "municipality_code": municipality_code,
                "municipality_name": municipality_name,
                "town_name": town_name,
                "lat": lat,
                "lon": lon,
            }
        )

    if not rows:
        return pd.DataFrame(columns=base_columns)

    out = pd.DataFrame(rows, columns=base_columns)

    # ここが今回の修正点。
    # 緯度・経度がすべて欠損でも列そのものは必ず残す。
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")

    key_columns = [
        "prefecture_code",
        "prefecture_name",
        "municipality_code",
        "municipality_name",
        "town_name",
    ]

    # numeric_only=True によって lat/lon 列が消えるケースを避け、
    # 列を明示した agg で町字単位にまとめる。
    grouped = (
        out.groupby(
            key_columns,
            dropna=False,
            as_index=False,
        )
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        )
    )

    # 念のため、将来データ形式が変わっても列が必ず存在するようにする。
    for column in ("lat", "lon"):
        if column not in grouped.columns:
            grouped[column] = float("nan")

    return grouped.sort_values(
        ["municipality_name", "town_name"],
        kind="stable",
    ).reset_index(drop=True)


@lru_cache(maxsize=64)
def load_prefecture_master(pref_code: str) -> pd.DataFrame:
    pref_code = str(pref_code).zfill(2)
    raw = _load_raw_prefecture(pref_code)
    return _canonicalize(raw, pref_code)


def get_municipalities(pref_code: str) -> list[tuple[str, str]]:
    df = load_prefecture_master(pref_code)
    if df.empty:
        return []

    pairs = (
        df[["municipality_code", "municipality_name"]]
        .drop_duplicates()
        .sort_values("municipality_name", kind="stable")
    )

    return [
        (str(row.municipality_code), str(row.municipality_name))
        for row in pairs.itertuples(index=False)
    ]


def get_towns(pref_code: str, municipality_code: str) -> list[dict]:
    df = load_prefecture_master(pref_code)
    code = str(municipality_code or "").strip()[:5]

    work = df[df["municipality_code"].astype(str) == code].copy()
    if work.empty:
        return []

    # lat/lon が無いデータでも町名一覧自体は返せるようにする。
    for column in ("lat", "lon"):
        if column not in work.columns:
            work[column] = float("nan")

    work = (
        work.drop_duplicates("town_name")
        .sort_values("town_name", kind="stable")
    )

    records = work[["town_name", "lat", "lon"]].to_dict("records")

    # NaN は扱いやすいよう None に変換する。
    for item in records:
        if pd.isna(item.get("lat")):
            item["lat"] = None
        if pd.isna(item.get("lon")):
            item["lon"] = None

    return records


def make_town_point_map(
    towns: list[dict],
) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}

    for item in towns or []:
        name = _norm_text(item.get("town_name"))
        lat = item.get("lat")
        lon = item.get("lon")

        if not name or lat is None or lon is None:
            continue

        try:
            point = (float(lat), float(lon))
        except Exception:
            continue

        result[name] = point

        compact = re.sub(r"\s+", "", name)
        result.setdefault(compact, point)

    return result
