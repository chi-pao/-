from __future__ import annotations

import re
import math
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
from pathlib import Path
import requests

XIT001_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001"
XIT002_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT002"
XCT001_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external/XCT001"

REQUEST_TIMEOUT = 30
LOOKBACK_QUARTERS = 12


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Ocp-Apim-Subscription-Key": api_key,
        "Accept": "application/json",
    }


def _to_float(value):
    if value is None or pd.isna(value):
        return np.nan

    text = str(value).replace(",", "").strip()

    if not text:
        return np.nan

    if "億" in text:
        multiplier = 100_000_000.0
    elif "万" in text:
        multiplier = 10_000.0
    else:
        multiplier = 1.0

    cleaned = re.sub(
        r"[^0-9.\-]",
        "",
        text,
    )

    if not cleaned:
        return np.nan

    try:
        return float(cleaned) * multiplier
    except Exception:
        return np.nan


def _year_from_building(value):
    if value is None or pd.isna(value):
        return np.nan

    match = re.search(
        r"(19\d{2}|20\d{2})",
        str(value),
    )

    if match:
        return float(
            match.group(1)
        )

    return np.nan


def _previous_quarters(
    count=LOOKBACK_QUARTERS,
):
    now = datetime.now()

    year = now.year
    quarter = (
        (now.month - 1) // 3
    ) + 1

    for _ in range(count):
        yield year, quarter

        quarter -= 1

        if quarter == 0:
            year -= 1
            quarter = 4


def _request_json(
    url,
    api_key,
    params,
):
    response = requests.get(
        url,
        headers=_headers(
            api_key
        ),
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code == 404:
        return {}

    response.raise_for_status()

    return response.json()
@lru_cache(
    maxsize=128
)
def _fetch_municipalities_cached(
    api_key: str,
    area_code: str,
):
    payload = _request_json(
        XIT002_URL,
        api_key,
        {
            "area": str(area_code),
            "language": "ja",
        },
    )

    result = []

    for item in payload.get("data", []) or []:
        code = str(item.get("id") or "").strip()
        name = str(item.get("name") or "").strip()

        if code and name:
            result.append((code, name))

    return result


def _resolve_search_city_codes(
    api_key: str,
    area_code: str,
    city_code: str,
):
    """
    政令指定都市などで市コードの配下に区コードがある場合、
    XIT002の市区町村一覧から区コードへ自動展開する。
    通常の市区町村や東京23区などは、そのコードをそのまま使う。
    """

    city_code = str(city_code or "").strip()

    if not city_code:
        return []

    try:
        municipalities = sorted(
            _fetch_municipalities_cached(
                api_key,
                str(area_code),
            ),
            key=lambda x: x[0],
        )
    except Exception:
        return [city_code]

    selected_index = None
    selected_name = ""

    for index, (code, name) in enumerate(municipalities):
        if code == city_code:
            selected_index = index
            selected_name = name
            break

    if selected_index is None:
        return [city_code]

    if not selected_name.endswith("市"):
        return [city_code]

    child_codes = []

    for code, name in municipalities[selected_index + 1:]:
        if name.endswith("区"):
            child_codes.append(code)
        else:
            break

    return child_codes or [city_code]
@lru_cache(
    maxsize=256
)
def _fetch_transactions_cached(
    api_key: str,
    area_code: str,
    city_code: str,
):
    rows = []

    search_city_codes = _resolve_search_city_codes(
        api_key,
        area_code,
        city_code,
    )

    for search_city_code in search_city_codes:
        for year, quarter in _previous_quarters():
            try:
                payload = _request_json(
                    XIT001_URL,
                    api_key,
                    {
                        "year": year,
                        "quarter": quarter,
                        "city": search_city_code,
                        "language": "ja",
                    },
                )
            except Exception:
                continue

            for row in payload.get("data", []) or []:
                item = dict(row)

                item["_SourceYear"] = year
                item["_SourceQuarter"] = quarter

                rows.append(item)

    return rows

def fetch_recent_transactions(
    api_key: str,
    area_code: str,
    city_code: str,
) -> pd.DataFrame:
    data_dir = Path(__file__).resolve().parent / "data"
    city = str(city_code or "").strip().replace(".0", "")

    if city and data_dir.exists():
        frames = []

        for file in sorted(data_dir.glob("mlit_*.csv")):
            try:
                for chunk in pd.read_csv(
                    file,
                    encoding="utf-8-sig",
                    dtype={"MunicipalityCode": str},
                    chunksize=10000,
                    low_memory=False,
                ):
                    if "MunicipalityCode" not in chunk.columns:
                        continue

                    codes = (
                        chunk["MunicipalityCode"]
                        .fillna("")
                        .astype(str)
                        .str.replace(r"\.0$", "", regex=True)
                        .str.strip()
                        .str.zfill(5)
                    )

                    matched = chunk.loc[codes == city.zfill(5)].copy()

                    if not matched.empty:
                        if "_SourceYear" not in matched.columns:
                            matched["_SourceYear"] = matched.get("取得年", np.nan)

                        if "_SourceQuarter" not in matched.columns:
                            matched["_SourceQuarter"] = matched.get("取得四半期", np.nan)

                        frames.append(matched)

            except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
                continue

        if frames:
            return pd.concat(frames, ignore_index=True)

    if not api_key or not area_code or not city_code:
        return pd.DataFrame()

    return pd.DataFrame(
        _fetch_transactions_cached(
            api_key,
            str(area_code),
            str(city_code),
        )
    )

def _normalize_transactions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if df.empty:
        return df
    
    df = df.copy()

    for column in (
        "TradePrice",
        "Area",
        "TotalFloorArea",
        "BuildingYear",
        "_SourceYear",
        "_SourceQuarter",
    ):
        if column not in df.columns:
            df[column] = np.nan

    out = pd.DataFrame(
        index=df.index
    )

    out["Type"] = df.get(
        "Type",
        "",
    )
    out["Use"] = df.get("Use", "")
    out["MunicipalityCode"] = df.get(
        "MunicipalityCode",
        "",
    )

    out["Municipality"] = df.get(
        "Municipality",
        "",
    )

    out["DistrictName"] = df.get(
        "DistrictName",
        "",
    )

    out["TradePrice"] = (
        df.get(
            "TradePrice",
            np.nan,
        )
        .map(
            _to_float
        )
    )

    out["Area"] = (
        df.get(
            "Area",
            np.nan,
        )
        .map(
            _to_float
        )
    )

    out[
        "TotalFloorArea"
    ] = (
        df.get(
            "TotalFloorArea",
            np.nan,
        )
        .map(
            _to_float
        )
    )

    out[
        "BuildingYear"
    ] = (
        df.get(
            "BuildingYear",
            np.nan,
        )
        .map(
            _year_from_building
        )
    )

    out["_SourceYear"] = pd.to_numeric(
        df.get("_SourceYear", np.nan),
        errors="coerce",
    )
    out["_SourceQuarter"] = pd.to_numeric(
        df.get("_SourceQuarter", np.nan),
        errors="coerce",
    )

    return out


def _type_mask(series: pd.Series, property_type: str, usage=None, building_year=None, source_year=None):
    """近隣成約は建物用途と新築/中古を混同しない。"""
    types = series.fillna("").astype(str)
    uses = (usage if isinstance(usage, pd.Series) else pd.Series("", index=series.index)).fillna("").astype(str)
    income = uses.str.contains("共同住宅|アパート|一棟|収益", regex=True) | types.str.contains("アパート|一棟収益", regex=True)
    other = uses.str.contains("事務所|店舗|工場|倉庫|駐車場|作業場", regex=True) & ~uses.str.contains("住宅", regex=False)
    house = (types.str.contains("土地と建物|戸建", regex=True)
             & ~income & ~other)
    year = pd.to_numeric(building_year, errors="coerce") if isinstance(building_year, pd.Series) else pd.Series(np.nan, index=series.index)
    when = pd.to_numeric(source_year, errors="coerce") if isinstance(source_year, pd.Series) else pd.Series(np.nan, index=series.index)
    age = when - year
    explicitly_new = types.str.contains("新築", regex=False)
    if property_type == "中古戸建て":
        return house & ~explicitly_new & (~age.between(0, 1) | age.isna())
    if property_type == "新築戸建て":
        return house & (explicitly_new | age.between(0, 1))
    if property_type == "中古マンション":
        return types.str.contains("中古マンション|マンション等", regex=True) & ~explicitly_new
    if property_type == "新築マンション":
        return types.str.contains("マンション", regex=False) & explicitly_new
    # 一棟収益の成約例は売買AI側の特徴量で扱い、現状ではここで混合しない。
    return pd.Series(False, index=series.index)


def _normalize_town_key(value) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("ヶ", "ケ").replace("ヵ", "カ")
    return text


def _haversine_km(lat1, lon1, lat2, lon2):
    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except Exception:
        return np.nan
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _point_for_town(name: str, town_points: dict | None):
    if not town_points:
        return None
    raw = str(name or "").strip()
    candidates = [raw, _normalize_town_key(raw)]
    # 国交省と住所マスタで「○丁目」の表記差があるケースを少し吸収する。
    base = re.sub(r"[0-9０-９一二三四五六七八九十]+丁目$", "", raw)
    if base and base != raw:
        candidates.extend([base, _normalize_town_key(base)])
    for key in candidates:
        if key in town_points:
            return town_points[key]
    # 完全一致がない場合は、同じ大字名から始まる町字の代表点を使う。
    norm = _normalize_town_key(raw)
    prefix_hits = []
    for key, point in town_points.items():
        key_norm = _normalize_town_key(key)
        if norm and (key_norm.startswith(norm) or norm.startswith(key_norm)):
            prefix_hits.append(point)
    if prefix_hits:
        try:
            return (
                float(np.mean([float(x[0]) for x in prefix_hits])),
                float(np.mean([float(x[1]) for x in prefix_hits])),
            )
        except Exception:
            return None
    return None


def comparable_estimate(
    api_key: str,
    area_code: str,
    city_code: str,
    district_name: str,
    property_type: str,
    building_year: float,
    condo_area=None,
    land_area=None,
    building_area=None,
    target_lat=None,
    target_lon=None,
    town_points=None,
):
    df = _normalize_transactions(
        fetch_recent_transactions(
            api_key,
            area_code,
            city_code,
        )
    )

    if df.empty:
        return None

    df = df[
        _type_mask(
            df["Type"],
            property_type,
            df["Use"],
            df["BuildingYear"],
            df["_SourceYear"],
        )
    ].copy()

    df = df[
        df[
            "TradePrice"
        ].notna()
        & (
            df[
                "TradePrice"
            ]
            > 0
        )
    ].copy()

    if df.empty:
        return None

    district_text = str(
        district_name
        or ""
    ).strip()

    district_parts = district_text.split()

    district_key = (
        district_parts[-1]
        if district_parts
        else ""
    )

    municipality_key = (
        district_parts[0]
        if len(district_parts) >= 2
        else ""
    )

    exact = df[
        df[
            "DistrictName"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        == district_key
    ].copy()

    municipality_rows = pd.DataFrame()

    if municipality_key:
        municipality_rows = df[
            df[
                "Municipality"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            == municipality_key
        ].copy()

    if (
        municipality_rows.empty
        and not exact.empty
        and "MunicipalityCode" in exact.columns
    ):
        exact_codes = (
            exact["MunicipalityCode"]
            .dropna()
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.strip()
            .unique()
            .tolist()
        )

        if exact_codes:
            municipality_rows = df[
                df["MunicipalityCode"]
                .fillna("")
                .astype(str)
                .str.replace(".0", "", regex=False)
                .str.strip()
                .isin(exact_codes)
            ].copy()

    # 町名データが不足しても、市区町村平均へ一気に広げない。
    # 住所マスタの代表点を使い、物理的に近い町字の成約から順に広げる。
    if len(exact) >= 3:
        work = exact.copy()
        work["_geo_distance_km"] = 0.0
        scope = "地区・町名"

    else:
        nearby_source = municipality_rows.copy()
        if nearby_source.empty:
            # 市区町村名がAPI表記と合わない場合でも、選択した市区町村コードの
            # 取引だけに限定する。全国/都道府県平均には広げない。
            code_text = str(city_code or "").strip()
            if code_text and "MunicipalityCode" in df.columns:
                normalized_codes = (
                    df["MunicipalityCode"]
                    .fillna("")
                    .astype(str)
                    .str.replace(".0", "", regex=False)
                    .str.strip()
                )
                nearby_source = df[normalized_codes == code_text].copy()

        geo_ready = (
            target_lat is not None
            and target_lon is not None
            and bool(town_points)
            and not nearby_source.empty
        )

        if geo_ready:
            distances = []
            for district_value in nearby_source["DistrictName"].fillna("").astype(str):
                point = _point_for_town(district_value, town_points)
                if point is None:
                    distances.append(np.nan)
                else:
                    distances.append(
                        _haversine_km(
                            target_lat, target_lon,
                            point[0], point[1],
                        )
                    )
            nearby_source["_geo_distance_km"] = distances
            mapped = nearby_source[nearby_source["_geo_distance_km"].notna()].copy()

            work = pd.DataFrame()
            scope = "近隣データ"
            for radius in (2.0, 5.0, 10.0, 20.0, 40.0):
                candidate = mapped[mapped["_geo_distance_km"] <= radius].copy()
                if len(candidate) >= 3:
                    work = candidate
                    scope = f"近隣{radius:g}km"
                    break

            if work.empty and len(mapped) >= 3:
                work = mapped.sort_values("_geo_distance_km").head(20).copy()
                farthest = float(work["_geo_distance_km"].max())
                scope = f"近隣約{farthest:.1f}km"

            if work.empty:
                return None
        else:
            # 距離情報が取れないときは、市区町村平均で価格を作らずAI側へ任せる。
            # これにより「際波の実績がない→宇部市平均」のような飛び方を防ぐ。
            return None

    current_year = (
        datetime.now().year
    )

    target_age = max(
        0,
        current_year
        - int(
            building_year
        ),
    )

    if (
        property_type
        in ("中古マンション", "新築マンション")
    ):
        target_area = float(
            condo_area
            or 0
        )

        if target_area <= 0:
            return None

        close = work[
            work[
                "Area"
            ].notna()
            & (
                work[
                    "Area"
                ]
                >= target_area
                * 0.80
            )
            & (
                work[
                    "Area"
                ]
                <= target_area
                * 1.25
            )
        ].copy()
       
        

        close_age = (
            current_year
            - close[
                "BuildingYear"
            ]
        )

        close = close[
            close[
                "BuildingYear"
            ].isna()
            | (
                (
                    close_age
                    - target_age
                ).abs()
                <= 8
            )
        ]

        if len(close) >= 3:
            work = close

        work = work.copy()

        work["_area_diff"] = (
            work["Area"] - target_area
        ).abs() / target_area

        work["_age_diff"] = (
            (
                current_year
                - work["BuildingYear"]
            )
            - target_age
        ).abs()

        work["_similarity_score"] = (
            work["_area_diff"] * 2.0
            + work["_age_diff"] / 20.0
        )

        if "_geo_distance_km" in work.columns:
            work["_similarity_score"] += (
                work["_geo_distance_km"].fillna(20.0).clip(lower=0.0, upper=20.0)
                / 5.0
                * 0.35
            )

        work = (
            work.sort_values(
                "_similarity_score"
            )
            .head(10)
            .copy()
        )

        unit = (
            work[
                "TradePrice"
            ]
            / work[
                "Area"
            ].replace(
                0,
                np.nan,
            )
        )

        unit = (
            unit.replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        if len(unit) < 3:
            return None

        if len(unit) >= 5:
            lower = unit.quantile(0.10)
            upper = unit.quantile(0.90)

            trimmed_unit = unit[
                (unit >= lower)
                & (unit <= upper)
            ]

            if len(trimmed_unit) >= 3:
                unit = trimmed_unit

        similarity_used = (
            work.loc[unit.index, "_similarity_score"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(2.0)
        )

        # 対象物件に近い成約ほど強く反映する。
        # 類似度スコア0に近いほど重みが大きく、離れるほど急速に弱くする。
        weights = 1.0 / (1.0 + similarity_used) ** 3
        # 新しい成約ほど重い。四半期あたり緩やかに減衰させる。
        now = datetime.now()
        quarter_idx = now.year * 4 + ((now.month - 1) // 3 + 1)
        source_idx = work.loc[unit.index, "_SourceYear"].fillna(now.year) * 4 + work.loc[unit.index, "_SourceQuarter"].fillna(1)
        weights = weights * np.exp(-np.maximum(0, quarter_idx - source_idx) / 16.0)

        weight_sum = float(weights.sum())
        if weight_sum > 0:
            weighted_unit_price = float(
                (unit * weights).sum() / weight_sum
            )
        else:
            weighted_unit_price = float(unit.median())

        median_unit_price = float(unit.median())
        estimate = (
            weighted_unit_price
            * target_area
        )

    else:
        target_land = float(
            land_area
            or 0
        )

        target_building = float(
            building_area
            or 0
        )

        if (
            target_land <= 0
            or target_building <= 0
        ):
            return None
        close = work[
            work[
                "Area"
            ].notna()
            & work[
                "TotalFloorArea"
            ].notna()
            & (
                work[
                    "Area"
                ]
                >= target_land
                * 0.75
            )
            & (
                work[
                    "Area"
                ]
                <= target_land
                * 1.30
            )
            & (
                work[
                    "TotalFloorArea"
                ]
                >= target_building
                * 0.75
            )
            & (
                work[
                    "TotalFloorArea"
                ]
                <= target_building
                * 1.30
            )
        ].copy()

        close_age = (
            current_year
            - close[
                "BuildingYear"
            ]
        )

        close = close[
            close[
                "BuildingYear"
            ].isna()
            | (
                (
                    close_age
                    - target_age
                ).abs()
                <= 10
            )
        ]

        if len(close) >= 3:
            work = close

        work = work.copy()

        work["_land_diff"] = (
            work["Area"] - target_land
        ).abs() / target_land

        work["_building_diff"] = (
            work["TotalFloorArea"] - target_building
        ).abs() / target_building

        work["_age_diff"] = (
            (
                current_year
                - work["BuildingYear"]
            )
            - target_age
        ).abs()

        work["_similarity_score"] = (
            work["_land_diff"] * 1.5
            + work["_building_diff"] * 1.5
            + work["_age_diff"] / 20.0
        )

        if "_geo_distance_km" in work.columns:
            work["_similarity_score"] += (
                work["_geo_distance_km"].fillna(20.0).clip(lower=0.0, upper=20.0)
                / 5.0
                * 0.35
            )

        work = (
            work.sort_values(
                "_similarity_score"
            )
            .head(10)
            .copy()
        )

        equivalent_area = (
            work["Area"]
            + 0.50
            * work[
                "TotalFloorArea"
            ]
        )

        unit = (
            work[
                "TradePrice"
            ]
            / equivalent_area.replace(
                0,
                np.nan,
            )
        )
        unit = (
            unit.replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        if len(unit) < 3:
            return None

        if len(unit) >= 5:
            lower = unit.quantile(0.10)
            upper = unit.quantile(0.90)

            trimmed_unit = unit[
                (unit >= lower)
                & (unit <= upper)
            ]

            if len(trimmed_unit) >= 3:
                unit = trimmed_unit

        similarity_used = (
            work.loc[unit.index, "_similarity_score"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(2.0)
        )

        # 土地面積・建物面積・築年が近い成約を強く反映する。
        weights = 1.0 / (1.0 + similarity_used) ** 3
        # 新しい成約ほど重い。四半期あたり緩やかに減衰させる。
        now = datetime.now()
        quarter_idx = now.year * 4 + ((now.month - 1) // 3 + 1)
        source_idx = work.loc[unit.index, "_SourceYear"].fillna(now.year) * 4 + work.loc[unit.index, "_SourceQuarter"].fillna(1)
        weights = weights * np.exp(-np.maximum(0, quarter_idx - source_idx) / 16.0)

        weight_sum = float(weights.sum())
        if weight_sum > 0:
            weighted_unit_price = float(
                (unit * weights).sum() / weight_sum
            )
        else:
            weighted_unit_price = float(unit.median())

        median_unit_price = float(unit.median())

        estimate = (
            weighted_unit_price
            * (
                target_land
                + 0.50
                * target_building
            )
        )

    used_count = int(len(unit))

    similarity_values = (
        work.loc[unit.index, "_similarity_score"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    median_similarity = (
        float(similarity_values.median())
        if len(similarity_values)
        else 1.5
    )
    similarity_confidence = _clip(
        1.0 / (1.0 + median_similarity),
        0.0,
        1.0,
    )

    q1 = float(unit.quantile(0.25))
    q3 = float(unit.quantile(0.75))
    iqr_ratio = (
        (q3 - q1) / median_unit_price
        if median_unit_price > 0
        else 1.0
    )
    dispersion_confidence = _clip(
        1.0 - (iqr_ratio / 0.80),
        0.0,
        1.0,
    )

    count_confidence = _clip(
        used_count / 10.0,
        0.0,
        1.0,
    )

    if scope == "地区・町名":
        scope_confidence = 1.00
    elif scope.startswith("近隣2"):
        scope_confidence = 0.95
    elif scope.startswith("近隣5"):
        scope_confidence = 0.90
    elif scope.startswith("近隣10"):
        scope_confidence = 0.82
    elif scope.startswith("近隣20"):
        scope_confidence = 0.72
    elif scope.startswith("近隣40") or scope.startswith("近隣約"):
        scope_confidence = 0.62
    else:
        scope_confidence = 0.55

    recency_confidence = 0.70
    source_year = work.loc[unit.index, "_SourceYear"].dropna()
    source_quarter = work.loc[unit.index, "_SourceQuarter"].dropna()
    if len(source_year) and len(source_quarter):
        now = datetime.now()
        current_quarter = ((now.month - 1) // 3) + 1
        current_index = now.year * 4 + current_quarter
        source_index = (
            source_year.astype(float) * 4
            + source_quarter.astype(float)
        )
        median_quarters_old = max(
            0.0,
            float(current_index - source_index.median()),
        )
        recency_confidence = _clip(
            1.0 - median_quarters_old / 16.0,
            0.35,
            1.0,
        )

    confidence = _clip(
        0.30 * similarity_confidence
        + 0.25 * dispersion_confidence
        + 0.20 * count_confidence
        + 0.15 * scope_confidence
        + 0.10 * recency_confidence,
        0.25,
        0.95,
    )

    return {
        "estimate_yen": float(estimate),
        "count": used_count,
        "scope": scope,
        "median_unit_price_yen": float(median_unit_price),
        "weighted_unit_price_yen": float(weighted_unit_price),
        "confidence": float(confidence),
        "similarity_confidence": float(similarity_confidence),
        "dispersion_confidence": float(dispersion_confidence),
        "recency_confidence": float(recency_confidence),
        "iqr_ratio": float(iqr_ratio),
        "median_similarity_score": float(median_similarity),
        "median_geo_distance_km": (
            float(work.loc[unit.index, "_geo_distance_km"].dropna().median())
            if "_geo_distance_km" in work.columns
            and len(work.loc[unit.index, "_geo_distance_km"].dropna())
            else 0.0 if scope == "地区・町名" else None
        ),
    }


def _pick(
    row: dict,
    *names,
):
    for name in names:
        if name in row:
            value = row.get(
                name
            )

            if (
                value is not None
                and str(
                    value
                ).strip()
                != ""
            ):
                return value

    return None


def _make_city_code(
    row: dict,
):
    direct = _pick(
        row,
        "city_code",
        "市区町村コード",
    )

    if direct:
        text = re.sub(
            r"\D",
            "",
            str(
                direct
            ),
        )

        if len(text) >= 5:
            return text[:5]

    pref = _pick(
        row,
        "標準地番号　市区町村コード　県コード",
        "標準地番号 市区町村コード 県コード",
    )

    city = _pick(
        row,
        "標準地番号　市区町村コード　市区町村コード",
        "標準地番号 市区町村コード 市区町村コード",
    )

    if (
        pref is not None
        and city is not None
    ):
        pref_text = re.sub(
            r"\D",
            "",
            str(pref),
        ).zfill(2)

        city_text = re.sub(
            r"\D",
            "",
            str(city),
        ).zfill(3)

        return (
            pref_text
            + city_text
        )[-5:]

    return ""


@lru_cache(
    maxsize=128
)
def _fetch_appraisal_cached(
    api_key: str,
    area_code: str,
    year: int,
):
    payload = _request_json(
        XCT001_URL,
        api_key,
        {
            "year": year,
            "area": area_code,
            "division": "00",
        },
    )

    if not isinstance(
        payload,
        dict,
    ):
        return []

    return (
        payload.get(
            "data",
            [],
        )
        or []
    )


def _fetch_latest_appraisals(
    api_key: str,
    area_code: str,
):
    current_year = (
        datetime.now().year
    )

    for year in range(
        current_year,
        max(
            2021,
            current_year - 5,
        ),
        -1,
    ):
        try:
            rows = (
                _fetch_appraisal_cached(
                    api_key,
                    area_code,
                    year,
                )
            )

            if rows:
                return (
                    year,
                    rows,
                )

        except Exception:
            continue

    return (
        None,
        [],
    )


def official_location_context(
    api_key: str,
    area_code: str,
    city_code: str,
    district_name: str,
):
    if (
        not api_key
        or not area_code
        or not city_code
    ):
        return None

    year, rows = (
        _fetch_latest_appraisals(
            api_key,
            str(
                area_code
            ),
        )
    )

    if not rows:
        return None

    search_city_codes = set(
        _resolve_search_city_codes(
            api_key,
            str(area_code),
            str(city_code),
        )
    )

    city_rows = [
        row
        for row in rows
        if _make_city_code(
            row
        )
        in search_city_codes
    ]

    if not city_rows:
        return None

    def row_text(row):
        parts = [
            _pick(
                row,
                "標準地番号　地域名",
                "標準地番号 地域名",
            ),
            _pick(
                row,
                "標準地　所在地　所在地番",
                "標準地 所在地 所在地番",
            ),
            _pick(
                row,
                "標準地　所在地　住居表示",
                "標準地 所在地 住居表示",
            ),
            _pick(
                row,
                "location",
                "所在地",
            ),
        ]

        return " ".join(
            str(x)
            for x in parts
            if x
        )

    district_text = str(
        district_name
        or ""
    ).strip()

    district = (
        district_text.split()[-1]
        if district_text
        else ""
    )

    district_rows = [
        row
        for row in city_rows
        if (
            district
            and district
            in row_text(row)
        )
    ]

    if len(
        district_rows
    ) >= 2:
        chosen = (
            district_rows
        )

        scope = (
            "地区・町名"
        )

    else:
        chosen = (
            city_rows
        )

        scope = (
            "市区町村"
        )

    def land_price(
        row,
    ):
        return _to_float(
            _pick(
                row,
                "1㎡当たりの価格",
                "公示価格",
                "u_current_years_price_ja",
            )
        )

    def station_distance(
        row,
    ):
        return _to_float(
            _pick(
                row,
                "標準地　交通施設の状況　距離",
                "標準地 交通施設の状況 距離",
                "u_road_distance_to_nearest_station_name_ja",
            )
        )

    chosen_land = (
        pd.Series(
            [
                land_price(x)
                for x in chosen
            ],
            dtype=float,
        )
        .dropna()
    )

    city_land = (
        pd.Series(
            [
                land_price(x)
                for x in city_rows
            ],
            dtype=float,
        )
        .dropna()
    )

    chosen_station = (
        pd.Series(
            [
                station_distance(x)
                for x in chosen
            ],
            dtype=float,
        )
        .dropna()
    )

    city_station = (
        pd.Series(
            [
                station_distance(x)
                for x in city_rows
            ],
            dtype=float,
        )
        .dropna()
    )

    return {
        "year": year,
        "scope": scope,
        "land_price_yen_per_sqm": (
            float(
                chosen_land.median()
            )
            if len(
                chosen_land
            )
            else None
        ),
        "city_land_price_yen_per_sqm": (
            float(
                city_land.median()
            )
            if len(
                city_land
            )
            else None
        ),
        "station_distance_m": (
            float(
                chosen_station.median()
            )
            if len(
                chosen_station
            )
            else None
        ),
        "city_station_distance_m": (
            float(
                city_station.median()
            )
            if len(
                city_station
            )
            else None
        ),
        "point_count": int(
            len(chosen)
        ),
    }


def _clip(
    value,
    low,
    high,
):
    return max(
        low,
        min(
            high,
            value,
        ),
    )


def build_market_context(
    *,
    api_key: str,
    base_ai_price_man_yen: float,
    property_type: str,
    area_code: str,
    city_code: str,
    district_name: str,
    building_year: float,
    station_minutes: float = 0.0,
    condo_area=None,
    land_area=None,
    building_area=None,
    target_lat=None,
    target_lon=None,
    town_points=None,
    comparable_weight_max: float = 0.0,
):
    base_yen = float(base_ai_price_man_yen) * 10_000.0

    comparable = comparable_estimate(
        api_key=api_key,
        area_code=str(area_code),
        city_code=str(city_code),
        district_name=district_name,
        property_type=property_type,
        building_year=building_year,
        condo_area=condo_area,
        land_area=land_area,
        building_area=building_area,
        target_lat=target_lat,
        target_lon=target_lon,
        town_points=town_points,
    ) if property_type in ("中古戸建て", "中古マンション") else None

    official = official_location_context(
        api_key=api_key,
        area_code=str(area_code),
        city_code=str(city_code),
        district_name=district_name,
    )

    # 公示地価は新AIモデルが学習特徴量として既に使うため、
    # ここでは二重に掛けない。表示・信頼性確認用として保持する。
    land_factor = 1.0
    station_factor = 1.0

    if official:
        city_station = official.get("city_station_distance_m")
        local_station = official.get("station_distance_m")

        if (
            station_minutes
            and station_minutes > 0
            and city_station
            and city_station > 0
        ):
            target_station_m = max(
                80.0,
                float(station_minutes) * 80.0,
            )
            station_factor = _clip(
                (city_station / target_station_m) ** 0.06,
                0.94,
                1.06,
            )
        elif (
            local_station
            and city_station
            and local_station > 0
            and city_station > 0
        ):
            station_factor = _clip(
                (city_station / local_station) ** 0.04,
                0.96,
                1.04,
            )

    # 駅距離の追加補正は未校正のため、AIモデルの推定値へは掛けない。
    ai_adjusted_yen = base_yen

    if comparable and comparable.get("estimate_yen", 0) > 0:
        confidence = float(comparable.get("confidence", 0.50))

        # 近隣成約の混合率は学習時に検証した上限だけ許可する。
        # 旧モデルや検証未実施の分類は0%=AIを守る。大幅な食い違いも抑制。
        comparable_yen = float(comparable["estimate_yen"])
        validated_cap = _clip(float(comparable_weight_max or 0.0), 0.0, 0.7)
        if property_type not in ("中古戸建て", "中古マンション", "新築戸建て", "新築マンション"):
            validated_cap = 0.0
        disagreement = abs(math.log(max(comparable_yen, 1) / max(ai_adjusted_yen, 1)))
        agreement_factor = min(1.0, 0.30 / max(disagreement, 0.30))
        comparable_weight = validated_cap * _clip(confidence, 0.0, 1.0) * agreement_factor
        adjusted_yen = float(math.exp(
            comparable_weight * math.log(max(comparable_yen, 1))
            + (1.0 - comparable_weight) * math.log(max(ai_adjusted_yen, 1))
        ))
    else:
        confidence = 0.0
        comparable_weight = 0.0
        adjusted_yen = ai_adjusted_yen

    return {
        "base_ai_price_man_yen": base_yen / 10_000.0,
        "adjusted_price_man_yen": adjusted_yen / 10_000.0,
        "comparable": comparable,
        "official": official,
        "comparable_weight": comparable_weight,
        "comparable_confidence": confidence,
        "land_factor": land_factor,
        "station_factor": station_factor,
    }
