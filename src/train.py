import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

from src.model import NextWordPredictor

# ── ハイパーパラメータ ───────────────────────────────────────────────────────
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.01
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
SAVE_CHECKPOINT = True  # False にするとエポックごとの保存をスキップ

# ── パス ────────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH = Path("data/processed/train_data.npz")
VOCAB_PATH = Path("data/processed/vocab.pkl")
MODELS_DIR = Path("models")


# ── 最適化手法 (ゼロつく1 第6章: Adam) ────────────────────────────────────
class Adam:
    """Adam optimizer"""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m: list[np.ndarray] = []
        self.v: list[np.ndarray] = []

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        if not self.m:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.iter += 1
        # バイアス補正済み学習率
        lr_t = (
            self.lr
            * np.sqrt(1.0 - self.beta2**self.iter)
            / (1.0 - self.beta1**self.iter)
        )

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] += (1.0 - self.beta1) * (grad - self.m[i])
            self.v[i] += (1.0 - self.beta2) * (grad**2 - self.v[i])
            param -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)


# ── チェックポイント保存 ─────────────────────────────────────────────────────
def save_checkpoint(model: NextWordPredictor, epoch: int, vocab_size: int) -> None:
    path = MODELS_DIR / f"checkpoint_epoch{epoch}.pkl"
    with path.open("wb") as f:
        pickle.dump(
            {
                "params": [p.copy() for p in model.params],
                "vocab_size": vocab_size,
                "embedding_dim": EMBEDDING_DIM,
                "hidden_dim": HIDDEN_DIM,
                "epoch": epoch,
            },
            f,
        )
    print(f"保存完了: {path}")


def save_final(model: NextWordPredictor, vocab_size: int) -> None:
    path = MODELS_DIR / "final_model.pkl"
    with path.open("wb") as f:
        pickle.dump(
            {
                "params": [p.copy() for p in model.params],
                "vocab_size": vocab_size,
                "embedding_dim": EMBEDDING_DIM,
                "hidden_dim": HIDDEN_DIM,
            },
            f,
        )
    print(f"保存完了: {path}")

    vocab_dst = MODELS_DIR / "vocab.pkl"
    shutil.copy(VOCAB_PATH, vocab_dst)
    print(f"保存完了: {vocab_dst}")


# ── メイン ──────────────────────────────────────────────────────────────────
def main() -> None:
    for path in (TRAIN_DATA_PATH, VOCAB_PATH):
        if not path.exists():
            print(f"ファイルが見つかりません: {path}", file=sys.stderr)
            sys.exit(1)

    MODELS_DIR.mkdir(exist_ok=True)

    # データ読み込み
    data = np.load(TRAIN_DATA_PATH)
    inputs: np.ndarray = data["inputs"]
    targets: np.ndarray = data["targets"]
    n_samples = len(inputs)

    with VOCAB_PATH.open("rb") as f:
        vocab = pickle.load(f)
    vocab_size: int = len(vocab["word_to_id"])

    print(f"学習データ数: {n_samples}")
    print(f"語彙数: {vocab_size}")

    # モデル・オプティマイザ初期化
    model = NextWordPredictor(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
    )
    optimizer = Adam(lr=LEARNING_RATE)

    n_iter = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE  # 1エポックのイテレーション数

    # 学習ループ
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        # シャッフル
        perm = np.random.permutation(n_samples)
        inputs_s = inputs[perm]
        targets_s = targets[perm]

        total_loss = 0.0

        for it in range(n_iter):
            start = it * BATCH_SIZE
            x_batch = inputs_s[start : start + BATCH_SIZE]
            t_batch = targets_s[start : start + BATCH_SIZE]

            loss = model.forward(x_batch, t_batch)
            model.backward()
            optimizer.update(model.params, model.grads)

            total_loss += loss

            if (it + 1) % 100 == 0:
                print(f"[{it + 1}/{n_iter}] loss: {loss:.3f}")

        avg_loss = total_loss / n_iter
        print(f"Epoch {epoch} 平均loss: {avg_loss:.3f}")

        if SAVE_CHECKPOINT:
            save_checkpoint(model, epoch, vocab_size)

    save_final(model, vocab_size)


if __name__ == "__main__":
    main()
