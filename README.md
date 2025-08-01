# 🤝 Multi-Agent Chat Demo

Python 3.10 × Gradio で動く **マルチエージェント型チャット UI**
（Market → Planning → Dev → Debug の 4 段階パイプラインを自動化）

---

## 📖 概要
`ui.py` では **OpenAI GPT-4o** を使った 4 つのエージェントを構成し、
`RoundRobinGroupChat` でラウンドロビンに会話を進行します。

| エージェント | 役割 |
| ------------ | ---- |
| **MarketResearchAgent** | ユーザーのヒアリング結果を分析し、市場調査レポートを作成 |
| **ProductPlanningAgent** | 市場調査を受けてプロダクト企画案を作成し、開発要件を定義 |
| **DemoDevelopmentAgent** | 企画案を基に最小デモ（Streamlit & Dockerfile）を設計・実装 |
| **DebuggerAgent** | 実装コードをレビューし、バグを修正または問題なしを宣言 |

各エージェントは **`TERMINATE`** の検知、または最大 **50 メッセージ** で終了します。

---

## 🗂️ ディレクトリ構成
```text
.
├─ Dockerfile
├─ requirements.txt
├─ ui.py
├─ .env.example
└─ README.md
```

---

## 🚀 クイックスタート

### 1. `.env` を準備
`.env.example` をコピーし、OpenAI API キーを設定します。

```bash
cp .env.example .env
echo "OPENAI_API_KEY=sk-..." >> .env
```

### 2. Docker イメージをビルド
```bash
docker build -t multi-agent-chat .
```

### 3. コンテナを起動
```bash
docker run --rm \
  --env-file .env \
  -p 7860:7860 \
  multi-agent-chat
```

ブラウザで <http://localhost:7860> を開き、
「楽しいゲームを考えてください」などと入力してみましょう。

---

## 🛠️ ローカル実行（Docker 不使用）
```bash
python -m venv .venv
source .venv/bin/activate           # Windows は .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install autogen gradio python-dotenv
export OPENAI_API_KEY=sk-...
python ui.py
```
---

## 🧑‍💻 開発ガイド

### エージェントの追加・変更
`create_agents()` に `AssistantAgent(...)` を追加するだけで
会話ループに組み込まれます。`TextMentionTermination` を使えば
**「TERMINATE」** などのキーワードで個別に停止させることも可能です。

### ロギング
詳細ログを得るには先頭で:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

を有効化してください。

### ホットリロード
開発時は:
```bash
gradio ui.py --dev --server-name 0.0.0.0 --server-port 7860
```
でソース変更時に自動リロードされます。



## 🙌 コントリビュート
Issue・PR 大歓迎です！ 機能追加やバグ報告など、ぜひご協力ください。
