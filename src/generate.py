import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

from src.model import NextWordPredictor

MODEL_PATH = Path("models/final_model.pkl")
VOCAB_PATH = Path("models/vocab.pkl")


def load_model(model_path: Path, vocab_size: int) -> NextWordPredictor:
    with model_path.open("rb") as f:
        ckpt = pickle.load(f)
    model = NextWordPredictor(
        vocab_size=ckpt["vocab_size"],
        embedding_dim=ckpt["embedding_dim"],
        hidden_dim=ckpt["hidden_dim"],
    )
    for p, saved in zip(model.params, ckpt["params"]):
        p[...] = saved
    return model


def sample(probs: np.ndarray, temperature: float) -> int:
    """temperature サンプリングで次の単語IDを返す"""
    p = probs ** (1.0 / temperature)
    p = p / p.sum()
    return int(np.random.choice(len(p), p=p))


def generate(
    model: NextWordPredictor,
    word_to_id: dict,
    id_to_word: dict,
    start_word: str,
    length: int,
    temperature: float,
) -> list[str]:
    if start_word not in word_to_id:
        print(f"未知語: '{start_word}'", file=sys.stderr)
        sys.exit(1)

    words = [start_word]
    current_id = word_to_id[start_word]

    for _ in range(length - 1):
        x = np.array([current_id])
        probs = model.predict(x)[0]
        current_id = sample(probs, temperature)
        words.append(id_to_word[current_id])

    return words


def main() -> None:
    parser = argparse.ArgumentParser(description="漱石風テキスト生成")
    parser.add_argument("--input", default="吾輩", help="開始単語 (デフォルト: 吾輩)")
    parser.add_argument("--length", type=int, default=20, help="生成する単語数 (デフォルト: 20)")
    parser.add_argument("--temperature", type=float, default=1.0, help="サンプリング温度 (デフォルト: 1.0)")
    args = parser.parse_args()

    for path in (MODEL_PATH, VOCAB_PATH):
        if not path.exists():
            print(f"ファイルが見つかりません: {path}", file=sys.stderr)
            sys.exit(1)

    with VOCAB_PATH.open("rb") as f:
        vocab = pickle.load(f)
    word_to_id: dict = vocab["word_to_id"]
    id_to_word: dict = vocab["id_to_word"]

    model = load_model(MODEL_PATH, len(word_to_id))

    print(f"開始単語: {args.input}")
    print("生成中...\n")

    words = generate(
        model, word_to_id, id_to_word,
        start_word=args.input,
        length=args.length,
        temperature=args.temperature,
    )

    print(" ".join(words))


if __name__ == "__main__":
    main()
