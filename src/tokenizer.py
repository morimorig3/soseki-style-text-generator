import pickle
import sys
from pathlib import Path

import MeCab
import numpy as np

INPUT_PATH = Path("data/processed/wagahai_cleaned.txt")
VOCAB_PATH = Path("data/processed/vocab.pkl")
TRAIN_DATA_PATH = Path("data/processed/train_data.npz")


def _build_tagger() -> MeCab.Tagger:
    """MeCab Tagger を初期化して返す。失敗時は sys.exit(1)"""
    import subprocess

    def run(arg: str) -> str:
        return subprocess.check_output(["mecab-config", arg]).decode().strip()

    candidates = [
        "-Owakati",
        f"-Owakati -r {run('--sysconfdir')}/mecabrc",
        f"-Owakati -d {run('--dicdir')}/ipadic -r {run('--sysconfdir')}/mecabrc",
    ]
    for args in candidates:
        try:
            return MeCab.Tagger(args)
        except RuntimeError:
            continue

    print("MeCabの初期化に失敗しました。辞書・設定ファイルのパスを確認してください。", file=sys.stderr)
    sys.exit(1)


def tokenize_text(text: str) -> list[str]:
    """MeCab分かち書きで単語リストを返す"""
    tagger = _build_tagger()
    return tagger.parse(text).split()


def create_vocab(words: list[str]) -> tuple[dict, dict]:
    """語彙辞書を作成して返す (word_to_id, id_to_word)"""
    unique_words = sorted(set(words))
    word_to_id = {word: idx for idx, word in enumerate(unique_words)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    return word_to_id, id_to_word


def create_dataset(
    words: list[str], word_to_id: dict
) -> tuple[np.ndarray, np.ndarray]:
    """連続する単語ペアから inputs / targets 配列を作成して返す"""
    ids = [word_to_id[word] for word in words]
    inputs = np.array(ids[:-1], dtype=np.int32)
    targets = np.array(ids[1:], dtype=np.int32)
    return inputs, targets


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"ファイルが見つかりません: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    text = INPUT_PATH.read_text(encoding="utf-8")

    print("形態素解析中...")
    words = tokenize_text(text)
    print(f"単語数: {len(words)}")

    word_to_id, id_to_word = create_vocab(words)
    print(f"語彙数: {len(word_to_id)}")

    inputs, targets = create_dataset(words, word_to_id)
    print(f"学習データ数: {len(inputs)}")

    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with VOCAB_PATH.open("wb") as f:
        pickle.dump({"word_to_id": word_to_id, "id_to_word": id_to_word}, f)

    np.savez_compressed(TRAIN_DATA_PATH, inputs=inputs, targets=targets)


if __name__ == "__main__":
    main()
