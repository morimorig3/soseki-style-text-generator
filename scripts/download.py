"""
青空文庫から夏目漱石「吾輩は猫である」をダウンロードして前処理するスクリプト
"""

import re
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


def download_aozora_text():
    """青空文庫からテキストをダウンロード"""
    # 吾輩は猫である（夏目漱石）のURL
    url = "https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip"

    # data/rawディレクトリを作成
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "wagahai.zip"

    print(f"ダウンロード中: {url}")
    urlretrieve(url, zip_path)
    print(f"保存完了: {zip_path}")

    # zipファイルを解凍
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)

    print(f"解凍完了: {raw_dir}")

    # zipファイルを削除
    zip_path.unlink()

    return raw_dir


def clean_text(text):
    """
    青空文庫のテキストから不要な記号を除去
    - ヘッダー・フッター（注記部分）を削除
    - ルビ(《》)を削除
    - 注記(［＃〜］)を削除
    - 縦書き記号を削除
    """
    # 改行コードを正規化してから行分割
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    # 青空文庫フォーマット:
    #   タイトル・著者 → 区切り線 → 記号説明 → 区切り線 → 本文 → 底本フッター
    # 区切り線: ハイフン10文字以上で構成される行
    # 本文開始: 最後の区切り線の次の行
    # 本文終了: 「底本：」で始まるフッター行（青空文庫の書式規約）
    separator = re.compile(r"^-{10,}$")
    separators = [i for i, line in enumerate(lines) if separator.match(line.strip())]

    if not separators:
        raise ValueError(
            "本文の区切り線が見つかりませんでした（ハイフン10文字以上の行がありません）"
        )

    start_idx = separators[-1] + 1  # 最後の区切り線の次の行から本文開始

    # フッター開始行（底本：）を本文終了とする
    end_idx = len(lines)
    footer = re.compile(r"^底本[：:]")
    for i in range(start_idx, len(lines)):
        if footer.match(lines[i]):
            end_idx = i
            break

    # 本文のみ抽出
    text = "\n".join(lines[start_idx:end_idx])

    # ルビを削除: 《ふりがな》
    text = re.sub(r"《[^》]*》", "", text)

    # 注記を削除: ［＃〜］
    text = re.sub(r"［＃[^］]*］", "", text)

    # | (ルビの開始記号)を削除
    text = text.replace("|", "")

    # 行末の余分な空白を削除
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # 連続する空行を1行にまとめる
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def preprocess_text(raw_dir):
    """テキストファイルを読み込んで前処理"""
    # 解凍されたテキストファイルを探す（.txtファイル）
    txt_files = list(raw_dir.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"{raw_dir}にテキストファイルが見つかりません")

    # 最初のテキストファイルを使用
    txt_file = txt_files[0]
    print(f"読み込み中: {txt_file}")

    # Shift-JISで読み込み（青空文庫はShift-JIS）
    with open(txt_file, encoding="shift_jis") as f:
        text = f.read()

    print(f"元テキスト文字数: {len(text)}")

    # 前処理
    cleaned_text = clean_text(text)
    print(f"前処理後文字数: {len(cleaned_text)}")

    # data/processedディレクトリに保存
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / "wagahai_cleaned.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"保存完了: {output_path}")

    return output_path


def main():
    """メイン処理"""
    print("=" * 50)
    print("青空文庫テキストダウンロード & 前処理")
    print("=" * 50)

    # ダウンロード
    raw_dir = download_aozora_text()

    # 前処理
    output_path = preprocess_text(raw_dir)

    print("\n✅ 完了!")
    print(f"前処理済みテキスト: {output_path}")


if __name__ == "__main__":
    main()
