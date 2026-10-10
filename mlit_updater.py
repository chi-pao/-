
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
API_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001"

PREFECTURES = {
    "北海道": "01", "青森県": "02", "岩手県": "03", "宮城県": "04",
    "秋田県": "05", "山形県": "06", "福島県": "07", "茨城県": "08",
    "栃木県": "09", "群馬県": "10", "埼玉県": "11", "千葉県": "12",
    "東京都": "13", "神奈川県": "14", "新潟県": "15", "富山県": "16",
    "石川県": "17", "福井県": "18", "山梨県": "19", "長野県": "20",
    "岐阜県": "21", "静岡県": "22", "愛知県": "23", "三重県": "24",
    "滋賀県": "25", "京都府": "26", "大阪府": "27", "兵庫県": "28",
    "奈良県": "29", "和歌山県": "30", "鳥取県": "31", "島根県": "32",
    "岡山県": "33", "広島県": "34", "山口県": "35", "徳島県": "36",
    "香川県": "37", "愛媛県": "38", "高知県": "39", "福岡県": "40",
    "佐賀県": "41", "長崎県": "42", "熊本県": "43", "大分県": "44",
    "宮崎県": "45", "鹿児島県": "46", "沖縄県": "47",
}


def get_api_key():
    key = os.environ.get("MLIT_API_KEY", "").strip()
    if key:
        return key

    try:
        import tomllib

        path = BASE_DIR / ".streamlit" / "secrets.toml"

        if path.exists():
            with path.open("rb") as f:
                settings = tomllib.load(f)

            key = str(settings.get("MLIT_API_KEY") or "").strip()
            if key:
                return key

            section = settings.get("mlit", {})
            return str(
                section.get("MLIT_API_KEY")
                or section.get("API_KEY")
                or section.get("api_key")
                or ""
            ).strip()

    except Exception as error:
        print("APIキー読み込みエラー:", error)

    return ""


def fetch_data(api_key, area, year, quarter):
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Accept": "application/json",
    }

    params = {
        "area": area,
        "year": year,
        "quarter": quarter,
        "language": "ja",
    }

    for attempt in range(3):
        try:
            response = requests.get(
                API_URL,
                params=params,
                headers=headers,
                timeout=60,
            )

            if response.status_code in (401, 403):
                raise RuntimeError("APIキーの認証エラー")

            if response.status_code == 404:
                return []

            response.raise_for_status()
            payload = response.json()

            if not isinstance(payload, dict):
                raise RuntimeError("API応答形式が不正です")

            status = str(payload.get("status", "")).upper()
            if status not in ("", "OK"):
                raise RuntimeError(f"APIステータス: {status}")

            rows = payload.get("data", [])

            if not isinstance(rows, list):
                raise RuntimeError("データ形式が不正です")

            return rows

        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))

    raise RuntimeError("データ取得に失敗しました")


def save_data(rows, prefecture, year, quarter):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    path = DATA_DIR / f"mlit_{prefecture}_{year}_Q{quarter}.csv"

    if not rows:
        return 0, str(path), False

    frame = pd.DataFrame(rows)

    if frame.empty:
        return 0, str(path), False

    if "TradePrice" not in frame.columns:
        raise ValueError("TradePrice列がありません")

    frame = frame.drop_duplicates().copy()

    frame["_SourceYear"] = int(year)
    frame["_SourceQuarter"] = int(quarter)

    frame["取得都道府県"] = prefecture
    frame["取得年"] = int(year)
    frame["取得四半期"] = int(quarter)
    frame["取得日時"] = datetime.now().isoformat(
        timespec="seconds"
    )

    temporary = path.with_suffix(".tmp")

    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return len(frame), str(path), True


def recent_quarters():
    now = datetime.now()

    current_index = now.year * 4 + (now.month - 1) // 3

    targets = []

    # 公表遅延に対応するため、直近の複数四半期を確認する。
    for offset in (4, 3, 2, 1):
        index = current_index - offset
        year = index // 4
        quarter = index % 4 + 1
        targets.append((year, quarter))

    return targets


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--year", type=int, default=None)
    parser.add_argument(
        "--quarter",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
    )
    parser.add_argument(
        "--prefecture",
        choices=list(PREFECTURES.keys()),
        default=None,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="保存済みファイルも再取得する",
    )

    args = parser.parse_args()

    if args.quarter is not None and args.year is None:
        parser.error("--quarter には --year も必要です")

    api_key = get_api_key()

    if not api_key:
        print("MLIT_API_KEY が見つかりません")
        sys.exit(1)

    if args.year is None:
        targets = recent_quarters()
    elif args.quarter is None:
        targets = [
            (args.year, q)
            for q in (1, 2, 3, 4)
        ]
    else:
        targets = [(args.year, args.quarter)]

    prefectures = (
        {args.prefecture: PREFECTURES[args.prefecture]}
        if args.prefecture
        else PREFECTURES
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    failed = []
    empty = []
    skipped = []

    print("全国不動産取引データ更新を開始します")
    print("対象都道府県:", len(prefectures))
    print("対象四半期:", targets)

    for prefecture, code in prefectures.items():
        for year, quarter in targets:
            label = f"{prefecture} {year}年 Q{quarter}"
            path = DATA_DIR / f"mlit_{prefecture}_{year}_Q{quarter}.csv"

            if path.exists() and not args.force:
                try:
                    existing = pd.read_csv(
                        path,
                        encoding="utf-8-sig",
                        nrows=1,
                    )
                    if (
                        "_SourceYear" in existing.columns
                        and "_SourceQuarter" in existing.columns
                        and "TradePrice" in existing.columns
                    ):
                        skipped.append(label)
                        print(f"保存済み: {label}")
                        continue
                except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
                    pass

            try:
                rows = fetch_data(
                    api_key, code, year, quarter
                )

                count, saved_path, saved = save_data(
                    rows, prefecture, year, quarter
                )

                if saved:
                    total_saved += count
                    print(f"成功: {label} / {count}件")
                else:
                    empty.append(label)
                    print(f"データなし: {label}")

            except Exception as error:
                failed.append(label)
                print(f"失敗: {label} / {error}")

            time.sleep(1)

    print()
    print("処理終了")
    print("今回保存した件数:", total_saved)
    print("保存済み:", len(skipped))
    print("データなし:", len(empty))
    print("取得失敗:", len(failed))
    print("保存先:", DATA_DIR)

    if failed:
        print("失敗した対象:")
        for item in failed:
            print(" -", item)
        sys.exit(1)


if __name__ == "__main__":
    main()
