# 公開サイトのスクショ自動入力テスト（Windows）
# 実行: py auto_test.py
import sys, subprocess, json, time, pathlib, tkinter as tk
from tkinter import filedialog

URL = "https://okaidokucyekka.streamlit.app/"
def ensure_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
        from playwright.sync_api import sync_playwright
    return sync_playwright

root = tk.Tk()
root.withdraw()
print("テストする物件スクリーンショットを選んでください。")
filename = filedialog.askopenfilename(title="物件スクショを選択", filetypes=[("画像", "*.png *.jpg *.jpeg *.webp"), ("すべて", "*.*")])
root.destroy()
if not filename:
    sys.exit("画像が選択されなかったため終了しました。")

sync_playwright = ensure_playwright()
output = pathlib.Path.home() / "Desktop" / "property_test_result"
output.mkdir(parents=True, exist_ok=True)
report = {"url": URL, "image": filename, "checks": {}, "errors": []}
try:
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=False)
        except Exception:
            print("初回のみブラウザをインストールします。")
            subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
            browser = p.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1360, "height": 950})
        page.goto(URL, wait_until="domcontentloaded", timeout=120000)
        page.get_by_text("スクショを入れるだけで自動入力").wait_for(timeout=120000)
        # Streamlitのiframe titleはバージョンで変わるため、実際のファイル入力を探索する。
        deadline = time.monotonic() + 45
        uploader = None
        while time.monotonic() < deadline and uploader is None:
            for frame in page.frames:
                try:
                    if frame.locator('input[type="file"]').count() and frame.locator('#zone').count():
                        uploader = frame.locator('input[type="file"]').first
                        break
                except Exception:
                    pass
            if uploader is None:
                page.wait_for_timeout(700)
        if uploader is None:
            raise RuntimeError("OCRの画像アップロード欄が見つかりません。")
        uploader.set_input_files(filename)
        print("OCRと自動入力を待っています（最大120秒）。")
        try:
            page.get_by_text("自動反映：", exact=False).wait_for(timeout=120000)
        except Exception:
            report["errors"].append("自動反映メッセージを確認できません")
        page.wait_for_timeout(4000)
        report["checks"]["page_text"] = page.locator("body").inner_text()[-12000:]
        for label in ("販売価格（万円）", "築年", "土地面積", "建物面積", "専有面積", "間取り", "都道府県", "市区町村", "物件種別"):
            try:
                control = page.get_by_label(label, exact=False).first
                if control.count():
                    report["checks"][label] = control.input_value(timeout=3000)
            except Exception as e:
                report["checks"][label] = "取得不可: " + str(e)[:100]
        # 実際の選択値はStreamlitのカスタムselectboxからも取得する。
        for label in ("都道府県", "市区町村", "物件種別", "間取り"):
            try:
                box = page.locator('[data-testid="stSelectbox"]').filter(has_text=label).first
                if box.count():
                    report["checks"][label + "_表示"] = box.inner_text(timeout=3000)
            except Exception:
                pass
        report["checks"]["ocr_status"] = [
            frame.locator("#status").inner_text()
            for frame in page.frames if frame.locator("#status").count()
        ]
        page.screenshot(path=str(output / "result.png"), full_page=True, timeout=30000)
        browser.close()
except Exception as e:
    report["errors"].append(repr(e))
(output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n結果:", output / "result.json")
print("画面:", output / "result.png")
print("取得した入力値:", json.dumps({k:v for k,v in report["checks"].items() if k!="page_text"}, ensure_ascii=False, indent=2))
if report["errors"]:
    print("エラー:", report["errors"])
input("Enterキーで終了")
