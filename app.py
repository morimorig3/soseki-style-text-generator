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
        st.session_state.current_text = []
    if "last_word_id" not in st.session_state:
        st.session_state.last_word_id = None
    if "selected_word" not in st.session_state:
        st.session_state.selected_word = None


def reset() -> None:
    st.session_state.current_text = []
    st.session_state.last_word_id = None


def set_start_word(word: str, word_to_id: dict) -> None:
    st.session_state.current_text = [word]
    st.session_state.last_word_id = word_to_id[word]


# ── メイン UI ────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="漱石風テキスト生成AI", layout="centered")

    st.title("夏目漱石風テキスト生成AIくん")
    st.caption(
        "ゼロから作るDeep Learning 1をベースに、NumPyのみで実装したシンプルなニューラルネットワークモデルです。"
    )
    st.caption(
        "「吾輩は猫である」を学習データとして次の単語を予測します。RNNやTransformerを使わないため、文脈を長く保持することはできませんが、単純な構造でどこまで漱石っぽい文章が生成できるか試してみてください！"
    )

    model, word_to_id, id_to_word, frequent_words = load_resources()
    init_state()

    # ── 開始単語の選択 ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("開始単語を選択")

    candidates = [w for w in frequent_words if w in word_to_id]

    # ランダムボタンで session_state を更新 → selectbox の index に反映
    if st.session_state.selected_word not in candidates:
        st.session_state.selected_word = candidates[0]

    col_select, col_random = st.columns([4, 1])

    with col_random:
        if st.button("🎲", use_container_width=True, help="ランダム選択"):
            word = str(np.random.choice(candidates))
            st.session_state.selected_word = word
            set_start_word(word, word_to_id)
            st.rerun()

    with col_select:
        selected = st.selectbox(
            "開始単語",
            candidates,
            index=candidates.index(st.session_state.selected_word),
            label_visibility="collapsed",
        )

    if selected != st.session_state.selected_word:
        st.session_state.selected_word = selected
        set_start_word(selected, word_to_id)
        st.rerun()

    # ── 現在の文章 ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("現在の文章")

    if st.session_state.current_text:
        text = "".join(st.session_state.current_text)
        st.markdown(
            f"<p style='font-size:1.1rem; line-height:2.0;'>{text}</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<p style='color:gray;'>（開始単語を選択してください）</p>",
            unsafe_allow_html=True,
        )

    # ── 設定 ─────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("設定")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        n_words = st.slider(
            "追加する単語数",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="ボタン1回で追加する単語の数",
        )
    with col_s2:
        temperature = st.slider(
            "Temperature",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="低いほど確率の高い単語を選びやすくなります。高いほどランダム性が増します。",
        )
        col_lo, col_hi = st.columns(2)
        col_lo.caption("← 保守的")
        col_hi.caption("多様 →")

    # ── 操作ボタン ───────────────────────────────────────────────────────────
    add_disabled = st.session_state.last_word_id is None
    col_add, col_sentence, col_reset = st.columns([3, 3, 2])

    with col_add:
        if st.button(
            f"単語を {n_words} 個追加",
            disabled=add_disabled,
            type="primary",
            use_container_width=True,
        ):
            current_id = st.session_state.last_word_id
            for _ in range(n_words):
                current_id = sample_next(model, current_id, temperature)
                st.session_state.current_text.append(id_to_word[current_id])
            st.session_state.last_word_id = current_id
            st.rerun()

    with col_sentence:
        if st.button(
            "1文生成",
            disabled=add_disabled,
            use_container_width=True,
        ):
            current_id = st.session_state.last_word_id
            for _ in range(100):  # 無限ループ防止の上限
                current_id = sample_next(model, current_id, temperature)
                word = id_to_word[current_id]
                st.session_state.current_text.append(word)
                if word == "。":
                    break
            st.session_state.last_word_id = current_id
            st.rerun()

    with col_reset:
        if st.button("リセット", use_container_width=True):
            reset()
            st.rerun()


if __name__ == "__main__":
    main()
