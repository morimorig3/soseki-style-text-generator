# soseki-style-text-generator

夏目漱石風のテキスト生成AI

「吾輩は猫である」をDeep Learningで学習し、単語を入力すると次の単語を予測するモデルです。

---

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
python src/tokenize.py
```

---

## 使い方

### 学習

```bash
python src/train.py
```

### テキスト生成

```bash
python src/generate.py --input "吾輩"
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
│   ├── tokenize.py       # 形態素解析
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