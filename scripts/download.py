"""
青空文庫から夏目漱石の作品をダウンロードして前処理するスクリプト
"""

import re
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# ── ダウンロードする作品リスト ───────────────────────────────────────────────
WORKS = [
    {
        "name": "wagahai",
        "title": "吾輩は猫である",
        "url": "https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip",
    },
    # {
    #     "name": "botchan",
    #     "title": "坊っちゃん",
    #     "url": "https://www.aozora.gr.jp/cards/000148/files/752_ruby_2438.zip",
    # },
    # {
    #     "name": "kokoro",
    #     "title": "こころ",
    #     "url": "https://www.aozora.gr.jp/cards/000148/files/773_ruby_5968.zip",
    # },
    # {
    #     "name": "sanshiro",
    #     "title": "三四郎",
    #     "url": "https://www.aozora.gr.jp/cards/000148/files/794_ruby_4237.zip",
    # },
]

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def download_work(name: str, title: str, url: str) -> Path:
    """1作品をダウンロードして解凍し、生テキストのパスを返す"""
    work_dir = RAW_DIR / name
    work_dir.mkdir(parents=True, exist_ok=True)

    zip_path = work_dir / f"{name}.zip"
    print(f"ダウンロード中: {title}")
    urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(work_dir)
    zip_path.unlink()

    txt_files = list(work_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"{work_dir} にテキストファイルが見つかりません")
    return txt_files[0]


def clean_text(text: str) -> str:
    """青空文庫テキストの前処理"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    separator = re.compile(r"^-{10,}$")
    separators = [i for i, line in enumerate(lines) if separator.match(line.strip())]
    if not separators:
        raise ValueError("本文の区切り線が見つかりませんでした")

    start_idx = separators[-1] + 1
    end_idx = len(lines)
    footer = re.compile(r"^底本[：:]")
    for i in range(start_idx, len(lines)):
        if footer.match(lines[i]):
            end_idx = i
            break

    text = "\n".join(lines[start_idx:end_idx])
    text = re.sub(r"《[^》]*》", "", text)  # ルビ削除
    text = re.sub(r"［＃[^］]*］", "", text)  # 注記削除
    text = text.replace("|", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for work in WORKS:
        name, title, url = work["name"], work["title"], work["url"]
        output_path = PROCESSED_DIR / f"{name}_cleaned.txt"

        if output_path.exists():
            print(f"スキップ (既存): {title}")
            continue

        try:
            raw_path = download_work(name, title, url)
            with raw_path.open(encoding="shift_jis") as f:
                text = f.read()
            cleaned = clean_text(text)
            output_path.write_text(cleaned, encoding="utf-8")
            print(f"保存完了: {output_path}  ({len(cleaned):,}文字)")
        except Exception as e:
            print(f"エラー ({title}): {e}")


if __name__ == "__main__":
    main()
