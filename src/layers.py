import numpy as np


class Embedding:
    """単語ID → 埋め込みベクトル (重み行列から対応行を抽出)"""

    def __init__(self, W: np.ndarray) -> None:
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx: np.ndarray | None = None

    def forward(self, idx: np.ndarray) -> np.ndarray:
        (W,) = self.params
        self.idx = idx
        return W[idx]

    def backward(self, dout: np.ndarray) -> None:
        assert self.idx is not None, "forward を先に呼んでください"
        (dW,) = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)


class Affine:
    """全結合層: out = x @ W + b"""

    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        W, b = self.params
        self.x = x
        return x @ W + b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self.x is not None, "forward を先に呼んでください"
        W, _ = self.params
        dW, db = self.grads
        dW[...] = self.x.T @ dout
        db[...] = dout.sum(axis=0)
        return dout @ W.T


class ReLU:
    """ReLU活性化関数"""

    def __init__(self) -> None:
        self.params: list = []
        self.grads: list = []
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self.mask is not None, "forward を先に呼んでください"
        return dout * self.mask


def _softmax(x: np.ndarray) -> np.ndarray:
    """数値安定な Softmax (内部ユーティリティ)"""
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


class SoftmaxWithLoss:
    """
    Softmax + Cross Entropy Loss を一体化した層

    合体させることで backward が (y - t) / batch_size に簡略化される。

    forward(x, t) → スカラー損失
    backward()    → shape (batch_size, vocab_size) の勾配
    """

    def __init__(self) -> None:
        self.params: list = []
        self.grads: list = []
        self.y: np.ndarray | None = None   # Softmax 出力
        self.t: np.ndarray | None = None   # 正解ラベル (ID の配列)

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Parameters
        ----------
        x : ロジット, shape (batch_size, vocab_size)
        t : 正解単語 ID, shape (batch_size,)

        Returns
        -------
        クロスエントロピー損失 (スカラー)
        """
        self.t = t
        self.y = _softmax(x)
        batch_size = x.shape[0]
        # 正解ラベルに対応する確率だけ取り出して対数
        log_p = np.log(self.y[np.arange(batch_size), t] + 1e-7)
        return float(-log_p.sum() / batch_size)

    def backward(self) -> np.ndarray:
        """
        Returns
        -------
        shape (batch_size, vocab_size) の勾配
        """
        assert self.y is not None and self.t is not None, "forward を先に呼んでください"
        batch_size = self.y.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        return dx / batch_size
