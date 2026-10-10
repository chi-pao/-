"""研究で苦手と分かった条件について、過度な確信を避けるための表示補助。

予測値、AIモデル、判定閾値は変更しない。
研究の「1,000万円以下」は成約価格での診断だが、公開アプリには
売出価格しかないため、同一集団と見なさず注意喚起に限る。
"""
from __future__ import annotations
from datetime import datetime


def assess_property_cautions(
    *, asking_price_man=None, building_year=None, land_area_sqm=None,
    property_type="", station_minutes=None, current_year=None,
):
    """中古戸建てについて、入力情報から確認事項を返す（精度確率ではない）。"""
    if property_type != "中古戸建て":
        return []
    year = int(current_year or datetime.now().year)
    items = []
    try:
        if asking_price_man is not None and 0 < float(asking_price_man) <= 1000:
            items.append({
                "title": "低価格帯の物件",
                "description": "研究では低価格の成約物件で誤差が大きい傾向がありました。"
                "売出価格は成約価格とは異なりますが、修繕費・解体費・再建築可否も確認してください。",
            })
    except (ValueError, TypeError):
        pass
    try:
        if building_year is not None and 1800 < float(building_year) <= year - 36:
            items.append({
                "title": "築36年以上",
                "description": "築古物件は建物状態や設備更新歴で価格が変わりやすくなります。"
                "耐震性・雨漏り・修繕履歴を確認してください。",
            })
    except (ValueError, TypeError):
        pass
    try:
        if land_area_sqm is not None and float(land_area_sqm) >= 351:
            items.append({
                "title": "土地面積351㎡以上",
                "description": "広い土地では形状・接道・用途地域・利用可能面積の差が重要です。"
                "実際の土地条件を確認してください。",
            })
    except (ValueError, TypeError):
        pass
    try:
        if station_minutes is None or float(station_minutes) <= 0:
            items.append({
                "title": "駅までの所要時間が未入力",
                "description": "交通条件が不明なため、最寄り駅・バス・車でのアクセスも別途確認してください。",
            })
    except (ValueError, TypeError):
        pass
    return items
