import json
import pickle
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import MeCab
import numpy as np

# 漢字・ひらがな・カタカナを1文字以上含む単語だけを対象にする
_RE_JAPANESE = re.compile(r"[\u3040-\u9FFF\u30A0-\u30FF]")
# ひらがな・カタカナのみ1文字 → 文脈なしでの品詞誤判定が多いため除外
_RE_KANA_SINGLE = re.compile(r"^[\u3040-\u30FF]$")

TRAIN_DATA_PATH = Path("data/processed/train_data.npz")
VOCAB_PATH = Path("data/processed/vocab.pkl")
OUTPUT_PATH = Path("data/processed/frequent_words.json")

TOP_N = 100
TARGET_POS = {"名詞", "動詞"}


def build_tagger() -> MeCab.Tagger:
    def run(arg: str) -> str:
        return subprocess.check_output(["mecab-config", arg]).decode().strip()

    candidates = [
        "",
        f"-r {run('--sysconfdir')}/mecabrc",
        f"-d {run('--dicdir')}/ipadic -r {run('--sysconfdir')}/mecabrc",
    ]
    for args in candidates:
        try:
            return MeCab.Tagger(args)
        except RuntimeError:
            continue

    print("MeCabの初期化に失敗しました。", file=sys.stderr)
    sys.exit(1)


def get_pos_and_form(tagger: MeCab.Tagger, word: str) -> tuple[str, str]:
    """
    単語の (品詞, 活用形) を返す。判定できない場合は ('', '')。
    feature 形式: 品詞,品詞細分類1,...,活用型,活用形,原形,...
    """
    node = tagger.parseToNode(word)
    while node:
        if node.surface == word:
            parts = node.feature.split(",")
            pos = parts[0]
            # 活用形は index 5 (足りない場合は空文字)
            form = parts[5] if len(parts) > 5 else ""
            return pos, form
        node = node.next
    return "", ""


def main() -> None:
    for path in (TRAIN_DATA_PATH, VOCAB_PATH):
        if not path.exists():
            print(f"ファイルが見つかりません: {path}", file=sys.stderr)
            sys.exit(1)

    # 語彙辞書読み込み
    with VOCAB_PATH.open("rb") as f:
        vocab = pickle.load(f)
    id_to_word: dict = vocab["id_to_word"]

    # 学習データから単語の出現回数をカウント
    data = np.load(TRAIN_DATA_PATH)
    # inputs と targets で全単語IDが揃っているが、重複するので inputs だけ使う
    # (最後の1単語は targets[-1] にしか現れないため両方マージ)
    all_ids = np.concatenate([data["inputs"], data["targets"][-1:]])
    word_counts: Counter = Counter(int(i) for i in all_ids)

    # MeCabで品詞フィルタ
    tagger = build_tagger()

    # 除外する活用形 (途中の形は原形として意味をなさない)
    exclude_forms = {"連用形", "連用タ接続", "仮定形", "命令ｅ"}
    # 除外する名詞細分類 (数・非自立など)
    exclude_noun_subtypes = {"数", "非自立", "接尾", "特殊"}

    filtered: list[tuple[str, int]] = []
    for word_id, count in word_counts.items():
        word = id_to_word[word_id]

        # 日本語文字(漢字・仮名)を含まない単語は除外
        if not _RE_JAPANESE.search(word):
            continue
        # ひらがな・カタカナのみの1文字は文脈なし誤判定が多いため除外
        if _RE_KANA_SINGLE.match(word):
            continue

        pos, form = get_pos_and_form(tagger, word)

        if pos not in TARGET_POS:
            continue
        # 活用形フィルタ (連用形など途中の形を除外)
        if pos == "動詞" and form in exclude_forms:
            continue
        # 名詞の細分類フィルタ
        node = tagger.parseToNode(word)
        while node:
            if node.surface == word:
                subtype = node.feature.split(",")[1] if "," in node.feature else ""
                if subtype in exclude_noun_subtypes:
                    break
                filtered.append((word, count))
                break
            node = node.next

    # 出現回数で降順ソートして TOP_N を取得
    filtered.sort(key=lambda x: x[1], reverse=True)
    top_words = filtered[:TOP_N]

    # コンソール出力
    print(f"TOP{TOP_N} 名詞・動詞:")
    for rank, (word, count) in enumerate(top_words, 1):
        print(f"{rank:2d}. {word} ({count:,}回)")

    # JSON保存
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    word_list = [word for word, _ in top_words]
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(word_list, f, ensure_ascii=False, indent=2)

    print(f"\n保存完了: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
