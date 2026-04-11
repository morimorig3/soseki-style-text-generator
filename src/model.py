import numpy as np

from src.layers import Affine, Embedding, ReLU, SoftmaxWithLoss, _softmax


class NextWordPredictor:
    """
    次の単語を予測するニューラルネットワーク

    構造 (学習時): Embedding → Affine → ReLU → Affine → SoftmaxWithLoss
    構造 (推論時): Embedding → Affine → ReLU → Affine → Softmax

    Parameters
    ----------
    vocab_size    : 語彙数
    embedding_dim : 埋め込み次元 (デフォルト 100)
    hidden_dim    : 隠れ層の次元 (デフォルト 128)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
    ) -> None:
        # He 初期化 (ReLU 用)
        W_emb = np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.01
        W1 = (
            np.random.randn(embedding_dim, hidden_dim).astype(np.float32)
            * np.sqrt(2.0 / embedding_dim)
        )
        b1 = np.zeros(hidden_dim, dtype=np.float32)
        W2 = (
            np.random.randn(hidden_dim, vocab_size).astype(np.float32)
            * np.sqrt(2.0 / hidden_dim)
        )
        b2 = np.zeros(vocab_size, dtype=np.float32)

        self.embedding = Embedding(W_emb)
        self.affine1 = Affine(W1, b1)
        self.relu = ReLU()
        self.affine2 = Affine(W2, b2)
        self.loss_layer = SoftmaxWithLoss()

        # Softmax は loss_layer に含まれるためここには入れない
        self.layers = [
            self.embedding,
            self.affine1,
            self.relu,
            self.affine2,
        ]

        # 全レイヤーのパラメータ・勾配を一元管理
        self.params: list[np.ndarray] = []
        self.grads: list[np.ndarray] = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def _forward_logits(self, x: np.ndarray) -> np.ndarray:
        """Embedding〜最終 Affine までの順伝播 (ロジットを返す)"""
        out: np.ndarray = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        学習用の順伝播: クロスエントロピー損失を返す

        Parameters
        ----------
        x : 単語IDの配列, shape (batch_size,)
        t : 正解単語IDの配列, shape (batch_size,)

        Returns
        -------
        クロスエントロピー損失 (スカラー)
        """
        logits = self._forward_logits(x)
        return self.loss_layer.forward(logits, t)

    def backward(self) -> None:
        """
        誤差逆伝播 (勾配は各レイヤーの grads に蓄積される)

        SoftmaxWithLoss の backward が (y-t)/batch_size を返すので
        呼び出し側から勾配を渡す必要はない。
        """
        dout = self.loss_layer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        推論: 確率分布を返す

        Parameters
        ----------
        x : 単語IDの配列, shape (batch_size,)

        Returns
        -------
        確率分布, shape (batch_size, vocab_size)
        """
        return _softmax(self._forward_logits(x))
