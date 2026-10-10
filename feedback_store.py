from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_FILE = DATA_DIR / "feedback.csv"

FIELDS = [
    "submitted_at",
    "category",
    "priority",
    "address",
    "ai_price_man_yen",
    "actual_price_man_yen",
    "actual_price_basis",
    "message",
    "name",
    "email",
    "company",
    "reply_requested",
    "property_type",
    "municipality_code",
    "district_name",
    "building_year",
    "structure",
    "city_planning",
    "floor_plan",
    "renovation",
    "condo_area",
    "land_area",
    "building_area",
    "station_minutes",
    "actual_price_period",
    "learning_candidate",
    "review_status",
    "record_id",
]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def classify_feedback(category: str, message: str, actual_price=None, actual_price_basis: str = "", actual_price_period: str = "") -> dict:
    """AI学習へ自動投入せず、一次分類だけを行う。最終判断は管理者が行う。"""
    category = _text(category)
    message = _text(message)
    basis = _text(actual_price_basis)

    priority = "通常"
    high_words = ("不具合", "エラー", "表示されない", "価格が違う", "成約", "法人", "提携", "取材", "広告")
    if category in {"法人・提携", "広告・取材", "重大な不具合"} or any(w in message for w in high_words):
        priority = "高"

    learning_candidate = False
    try:
        has_actual = actual_price is not None and float(actual_price) > 0
    except Exception:
        has_actual = False

    period = _text(actual_price_period)
    if (
        has_actual
        and period
        and basis in {"実際の成約価格", "売買契約書・成約資料", "査定書・業者資料"}
    ):
        learning_candidate = True

    return {
        "priority": priority,
        "learning_candidate": learning_candidate,
        "review_status": "要確認" if learning_candidate else "未対応",
    }


def _webhook_url() -> str:
    return os.environ.get("FEEDBACK_WEBHOOK_URL", "").strip()


def save_feedback(record: dict) -> dict:
    row = {key: "" for key in FIELDS}
    for key in FIELDS:
        if key in record:
            row[key] = record.get(key)

    if not row["submitted_at"]:
        row["submitted_at"] = datetime.now().isoformat(timespec="seconds")

    seed = "|".join(
        [
            _text(row["submitted_at"]),
            _text(row["email"]),
            _text(row["address"]),
            _text(row["message"]),
        ]
    )
    row["record_id"] = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

    exists = FEEDBACK_FILE.exists()
    with FEEDBACK_FILE.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})

    webhook_status = "未設定"
    url = _webhook_url()
    if url:
        try:
            response = requests.post(url, json=row, timeout=15)
            response.raise_for_status()
            webhook_status = "送信済み"
        except Exception as exc:
            webhook_status = f"送信失敗: {exc}"

    return {
        "record_id": row["record_id"],
        "webhook_status": webhook_status,
    }


def load_feedback() -> pd.DataFrame:
    if not FEEDBACK_FILE.exists():
        return pd.DataFrame(columns=FIELDS)
    try:
        df = pd.read_csv(FEEDBACK_FILE, encoding="utf-8-sig", dtype=str).fillna("")
    except Exception:
        return pd.DataFrame(columns=FIELDS)
    for field in FIELDS:
        if field not in df.columns:
            df[field] = ""
    return df[FIELDS].copy()


def update_review_status(record_id: str, new_status: str) -> bool:
    df = load_feedback()
    if df.empty or "record_id" not in df.columns:
        return False
    mask = df["record_id"].astype(str) == str(record_id)
    if not mask.any():
        return False
    df.loc[mask, "review_status"] = str(new_status)
    df.to_csv(FEEDBACK_FILE, index=False, encoding="utf-8-sig")
    return True



def export_approved_training_candidates(output_path: Path | None = None) -> Path | None:
    """管理者が「学習承認」にした実価格報告だけを学習用CSVへ書き出す。"""
    df = load_feedback()
    if df.empty:
        return None
    approved = df[
        (df["review_status"] == "学習承認")
        & (df["learning_candidate"] == "候補")
    ].copy()
    if approved.empty:
        return None

    def num(series):
        return pd.to_numeric(series, errors="coerce")

    out = pd.DataFrame()
    out["Type"] = approved["property_type"].map(
        lambda x: "中古マンション" if "マンション" in str(x) else "土地と建物"
    )
    out["MunicipalityCode"] = approved["municipality_code"]
    out["DistrictName"] = approved["district_name"]
    out["TradePrice"] = num(approved["actual_price_man_yen"]) * 10000.0
    out["FloorPlan"] = approved["floor_plan"]
    out["Area"] = num(approved["condo_area"]).where(
        approved["property_type"].astype(str).str.contains("マンション"),
        num(approved["land_area"]),
    )
    out["TotalFloorArea"] = num(approved["building_area"])
    out["BuildingYear"] = num(approved["building_year"])
    out["Structure"] = approved["structure"]
    out["CityPlanning"] = approved["city_planning"]
    out["Period"] = approved["actual_price_period"]
    out["Renovation"] = approved["renovation"]
    out["PriceCategory"] = "ユーザー提供・管理者承認"
    out = out[out["TradePrice"].notna() & (out["TradePrice"] > 0)].copy()
    if out.empty:
        return None

    if output_path is None:
        output_path = DATA_DIR / "approved_feedback_training.csv"
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
