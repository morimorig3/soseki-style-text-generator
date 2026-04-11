# 夏目漱石風のテキスト生成AIくん

ゼロから作るDeepLearning1 の演習としてシンプルなニューラルネットワークの予測モデルを実装しました。
漱石の「吾輩は猫である」を形態素解析して学習データとしてモデルに学習させています。

ゼロつく1の演習なので、RNNやTransformerは使っていません。
そのため生成精度には限界がありますが単語の予測のモデルとしてシンプルなものにできたのではないかと思います。

**ハイパーパラメータ**

| 項目 | 値 |
|---|---|
| エポック | 10 |
| ミニバッチサイズ | 128 |
| 学習率 | 0.01 |
| 更新手法 | Adam (β1=0.9, β2=0.999) |
| Embedding dim | 100 |
| 隱れ層 | 128 |
| 語彙数 | 13,604 |
| 学習データ数 | 206,387 |

学習データは青空文庫から漱石の「吾輩は猫である」を利用しています。
語彙数に関してはMeCabを利用して、それぞれの文章を形態素解析して分解しています。

# develop

## セットアップ

### 1. 必要なシステムツールのインストール

**MeCab**(形態素解析器)が必要です。

#### macOS (Homebrew)
```bash
brew install mecab mecab-ipadic
```

#### Ubuntu/Debian
```bash
sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8
```

### 2. Python環境のセットアップ

```bash
# uvのインストール(まだの場合)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 仮想環境の作成
uv venv

# 仮想環境の有効化
source .venv/bin/activate  # macOS/Linux
# または
.venv\Scripts\activate  # Windows

# 依存関係のインストール
uv pip install -r requirements.txt
```

### 3. データのダウンロードと前処理

```bash
# 青空文庫からテキストをダウンロード
python scripts/download.py

# 形態素解析してデータセット作成
python -m src.tokenizer
```

---

## 使い方

### 学習

```bash
python -m src.train
```

### テキスト生成

```bash
python -m src.generate --input "吾輩"
```

---

## プロジェクト構成

```
soseki-style-text-generator/
├── data/
│   ├── raw/              # 青空文庫の生テキスト
│   └── processed/        # 前処理済みデータ
├── scripts/
│   └── download.py       # データダウンロードスクリプト
├── src/
│   ├── tokenizer.py      # 形態素解析
│   ├── model.py          # モデル定義
│   ├── train.py          # 学習
│   └── generate.py       # テキスト生成
├── models/               # 学習済みモデル
├── requirements.txt      # Python依存関係
└── README.md
```

---

## 技術スタック

- **Python 3.11+**
- **NumPy**: 数値計算
- **MeCab**: 形態素解析
- **Deep Learning**: ゼロから実装(フレームワーク不使用)

---

## ライセンス

MIT License

学習データは青空文庫(著作権保護期間満了)を使用しています。