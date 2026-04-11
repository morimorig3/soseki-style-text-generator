import json
import pickle
from pathlib import Path

import numpy as np
import streamlit as st

from src.model import NextWordPredictor

MODEL_PATH = Path("models/final_model.pkl")
VOCAB_PATH = Path("models/vocab.pkl")
FREQUENT_WORDS_PATH = Path("data/processed/frequent_words.json")


# ── モデル読み込み (起動時に1回だけ) ────────────────────────────────────────
@st.cache_resource
def load_resources() -> tuple[NextWordPredictor, dict, dict, list[str]]:
    with MODEL_PATH.open("rb") as f:
        ckpt = pickle.load(f)
    with VOCAB_PATH.open("rb") as f:
        vocab = pickle.load(f)
    with FREQUENT_WORDS_PATH.open(encoding="utf-8") as f:
        frequent_words: list[str] = json.load(f)

    model = NextWordPredictor(
        vocab_size=ckpt["vocab_size"],
        embedding_dim=ckpt["embedding_dim"],
        hidden_dim=ckpt["hidden_dim"],
    )
    for p, saved in zip(model.params, ckpt["params"]):
        p[...] = saved

    return model, vocab["word_to_id"], vocab["id_to_word"], frequent_words


def sample_next(
    model: NextWordPredictor,
    word_id: int,
    temperature: float,
) -> int:
    """temperature サンプリングで次の単語IDを返す"""
    x = np.array([word_id])
    probs = model.predict(x)[0].astype(np.float64)
    probs = probs ** (1.0 / temperature)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


# ── Session State 初期化 ─────────────────────────────────────────────────────
def init_state() -> None:
    if "current_text" not in st.session_state:
        st.session_state.current_text: list[str] = []
    if "last_word_id" not in st.session_state:
        st.session_state.last_word_id: int | None = None


def reset() -> None:
    st.session_state.current_text = []
    st.session_state.last_word_id = None


def set_start_word(word: str, word_to_id: dict) -> None:
    st.session_state.current_text = [word]
    st.session_state.last_word_id = word_to_id[word]


# ── メイン UI ────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="漱石風テキスト生成AI", layout="centered")

    st.title("夏目漱石風テキスト生成AI")
    st.caption("「吾輩は猫である」を学習したAIが次の単語を予測します")

    model, word_to_id, id_to_word, frequent_words = load_resources()
    init_state()

    # ── 開始単語の選択 ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("開始単語を選択")

    # frequent_words を先頭20語 + ランダムボタンで表示
    display_words = [w for w in frequent_words[:20] if w in word_to_id]

    cols = st.columns(5)
    for i, word in enumerate(display_words):
        if cols[i % 5].button(word, key=f"word_{word}"):
            set_start_word(word, word_to_id)
            st.rerun()

    if st.button("🎲 ランダム", key="random"):
        candidates = [w for w in frequent_words if w in word_to_id]
        word = np.random.choice(candidates)
        set_start_word(word, word_to_id)
        st.rerun()

    # ── 現在の文章 ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("現在の文章")

    text = (
        " ".join(st.session_state.current_text)
        if st.session_state.current_text
        else "（開始単語を選択してください）"
    )
    st.text_area(
        "生成テキスト",
        value=text,
        height=120,
        disabled=True,
        label_visibility="collapsed",
    )

    # ── 操作ボタン ───────────────────────────────────────────────────────────
    col_add, col_reset = st.columns([2, 1])

    with col_add:
        add_disabled = st.session_state.last_word_id is None
        if st.button(
            "次の単語を追加",
            disabled=add_disabled,
            type="primary",
            use_container_width=True,
        ):
            temperature = st.session_state.get("temperature", 1.0)
            next_id = sample_next(model, st.session_state.last_word_id, temperature)
            next_word = id_to_word[next_id]
            st.session_state.current_text.append(next_word)
            st.session_state.last_word_id = next_id
            st.rerun()

    with col_reset:
        if st.button("リセット", use_container_width=True):
            reset()
            st.rerun()

    # ── 設定 ─────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("設定")

    temperature = st.slider(
        "Temperature",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="低いほど確率の高い単語を選びやすくなります。高いほどランダム性が増します。",
    )
    st.session_state["temperature"] = temperature

    col_lo, col_hi = st.columns(2)
    col_lo.caption("← 0.5: 保守的 (高頻度語に集中)")
    col_hi.caption("2.0: 多様 (ランダム性が強い) →")


if __name__ == "__main__":
    main()
