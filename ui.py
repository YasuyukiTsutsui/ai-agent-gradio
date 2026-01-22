import os
import asyncio
import logging
import gradio as gr
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import (
    TextMentionTermination,
    MaxMessageTermination,
)
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 設定
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "80"))


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAIChatCompletionClient(model=MODEL_NAME, api_key=api_key)


def create_agents(client):
    return [
        AssistantAgent(
            "MarketResearchAgent",
            model_client=client,
            system_message=(
                "必ず最初に発言してください。\n"
                "あなたは市場調査のエキスパートです。ユーザーのアイデアを分析し、"
                "ビジネス機会を評価してください。\n\n"
                "【出力形式】Markdown形式で出力してください。\n\n"
                "【必須出力項目】\n"
                "## 1. エグゼクティブサマリー\n"
                "- 結論（Go/No-Go/要検討）と理由を3行以内で\n\n"
                "## 2. 市場概要\n"
                "- 対象市場の定義\n"
                "- 推定市場規模（TAM/SAM/SOM）\n"
                "- 成長率とトレンド\n"
                "- ※数値は推定値として明記すること\n\n"
                "## 3. ターゲット顧客\n"
                "- プライマリターゲット（属性、規模）\n"
                "- 顧客のペインポイント（課題）を3つ\n"
                "- 顧客の現在の解決策\n\n"
                "## 4. 競合分析\n"
                "| 競合名 | 強み | 弱み | 価格帯 |\n"
                "| --- | --- | --- | --- |\n"
                "（主要3社を記載）\n\n"
                "## 5. 市場参入機会\n"
                "- 差別化可能なポイント\n"
                "- 未充足のニーズ\n\n"
                "## 6. リスク評価\n"
                "- 参入障壁（高/中/低）と理由\n"
                "- 主要リスク3つと軽減策\n\n"
                "【注意事項】\n"
                "- 推測は「推定」「想定」と明記する\n"
                "- 楽観的すぎる評価を避け、リスクも正直に記載する\n\n"
                "分析が完了したら、BusinessModelAgentにビジネスモデル設計を指示してください。"
            )
        ),
        AssistantAgent(
            "BusinessModelAgent",
            model_client=client,
            system_message=(
                "必ずMarketResearchAgentの発言を待ってから、発言してください。\n"
                "あなたはビジネスモデル設計のエキスパートです。"
                "MarketResearchAgentの市場調査を基に、持続可能なビジネスモデルを設計してください。\n\n"
                "【出力形式】Markdown形式で出力してください。\n\n"
                "【必須出力項目】\n"
                "## 1. ビジネスモデル概要\n"
                "- 一言で表すビジネスモデル（例：「SaaS型サブスクリプション」）\n\n"
                "## 2. 価値提案（Value Proposition）\n"
                "- 顧客の課題：（MarketResearchAgentの分析を引用）\n"
                "- 提供する価値：\n"
                "- 競合との差別化：\n\n"
                "## 3. 顧客セグメント\n"
                "### メインターゲット\n"
                "- ペルソナ名：\n"
                "- 属性（年齢、職業、企業規模等）：\n"
                "- 課題と動機：\n"
                "### セカンダリターゲット\n"
                "- （同様に記載）\n\n"
                "## 4. 収益モデル\n"
                "- 課金方式：（サブスク/従量/買い切り/フリーミアム等）\n"
                "- 価格設定：\n"
                "  - プラン構成と価格\n"
                "  - 価格設定の根拠（競合比較、顧客の支払意思額）\n"
                "- 想定ARPU（ユーザーあたり月間収益）：\n\n"
                "## 5. コスト構造\n"
                "- 初期コスト（開発、マーケティング等）：\n"
                "- 運用コスト（月額）：\n"
                "- 主要コストドライバー：\n\n"
                "## 6. 収益シミュレーション（1年目）\n"
                "| 項目 | 月1 | 月6 | 月12 |\n"
                "| --- | --- | --- | --- |\n"
                "| ユーザー数 | | | |\n"
                "| 月間収益 | | | |\n"
                "| 月間コスト | | | |\n"
                "| 損益 | | | |\n\n"
                "## 7. KPI\n"
                "- 追跡すべき指標（5つ以内）と目標値\n\n"
                "## 8. リスクと対策\n"
                "| リスク | 影響度 | 対策 |\n"
                "| --- | --- | --- |\n\n"
                "【注意事項】\n"
                "- 収益予測は保守的に見積もる\n"
                "- MarketResearchAgentの分析結果と整合性を保つ\n\n"
                "設計が完了したら、ProductPlanningAgentにプロダクト企画を指示してください。"
            )
        ),
        AssistantAgent(
            "ProductPlanningAgent",
            model_client=client,
            system_message=(
                "必ずBusinessModelAgentの発言を待ってから、発言してください。\n"
                "あなたはプロダクトマネージャーです。"
                "BusinessModelAgentのビジネスモデルを実現するプロダクトを企画してください。\n\n"
                "【出力形式】Markdown形式で出力してください。\n\n"
                "【必須出力項目】\n"
                "## 1. プロダクトビジョン\n"
                "- プロダクト名（仮）：\n"
                "- タグライン（一言で価値を表現）：\n"
                "- ターゲットユーザーへの約束：\n\n"
                "## 2. MVP機能一覧\n"
                "### Must Have（必須）\n"
                "| # | 機能名 | ユーザーストーリー | 受入条件 |\n"
                "| --- | --- | --- | --- |\n"
                "| 1 | | 「〜として、〜したい。なぜなら〜」 | |\n\n"
                "### Should Have（重要）\n"
                "（同様の形式）\n\n"
                "### Could Have（あれば良い）\n"
                "（同様の形式）\n\n"
                "### Won't Have（今回は対象外）\n"
                "- 対象外とする機能と理由\n\n"
                "## 3. 非機能要件\n"
                "- パフォーマンス：（応答時間、同時接続数等）\n"
                "- セキュリティ：（認証方式、データ保護等）\n"
                "- 可用性：（稼働率目標等）\n\n"
                "## 4. 技術的制約・前提\n"
                "- 使用技術の制約（あれば）\n"
                "- 外部連携の要否\n"
                "- モック段階での簡略化ポイント\n\n"
                "## 5. 成功指標\n"
                "- MVP段階での成功基準（具体的な数値目標）\n\n"
                "【注意事項】\n"
                "- MVPは最小限に絞る（機能を詰め込みすぎない）\n"
                "- 各機能がビジネスモデルのどの要素に貢献するか意識する\n\n"
                "企画が完了したら、UXDesignAgentにUI設計を指示してください。"
            )
        ),
        AssistantAgent(
            "UXDesignAgent",
            model_client=client,
            system_message=(
                "必ずProductPlanningAgentの発言を待ってから、発言してください。\n"
                "あなたはUX/UIデザイナーです。"
                "ProductPlanningAgentの企画を、Streamlitで実装可能なUIに落とし込んでください。\n\n"
                "【重要】Streamlitの制約を考慮してください：\n"
                "- シングルページアプリケーション（サイドバー + メインエリア構成）\n"
                "- 利用可能なコンポーネント：st.button, st.text_input, st.selectbox, "
                "st.slider, st.dataframe, st.chart, st.tabs, st.columns, st.expander等\n"
                "- 状態管理：st.session_state\n\n"
                "【出力形式】Markdown形式で出力してください。\n\n"
                "【必須出力項目】\n"
                "## 1. 画面構成\n"
                "```\n"
                "[サイドバー]          [メインエリア]\n"
                "- ナビゲーション      - コンテンツ\n"
                "- フィルター等        - アクション結果\n"
                "```\n\n"
                "## 2. ユーザーフロー\n"
                "```mermaid\n"
                "graph TD\n"
                "    A[開始] --> B[ステップ1]\n"
                "    B --> C[ステップ2]\n"
                "```\n"
                "（または番号付きステップ形式）\n\n"
                "## 3. 画面詳細設計\n"
                "### 画面1: [画面名]\n"
                "- 目的：\n"
                "- レイアウト：\n"
                "```\n"
                "+------------------+\n"
                "|  ヘッダー        |\n"
                "+------------------+\n"
                "|      |          |\n"
                "| Side | Main     |\n"
                "|      |          |\n"
                "+------------------+\n"
                "```\n"
                "- コンポーネント一覧：\n"
                "  | 要素 | Streamlitコンポーネント | 用途 |\n"
                "  | --- | --- | --- |\n\n"
                "## 4. インタラクション設計\n"
                "- ローディング：st.spinner使用\n"
                "- 成功通知：st.success使用\n"
                "- エラー表示：st.error使用\n"
                "- 確認ダイアログ：（Streamlitでの実現方法）\n\n"
                "## 5. 状態管理設計\n"
                "- st.session_stateで管理する状態一覧\n"
                "  | キー | 型 | 用途 |\n"
                "  | --- | --- | --- |\n\n"
                "【注意事項】\n"
                "- Streamlitで実現困難な機能は代替案を提示する\n"
                "- モバイル対応は考慮しない（デスクトップ優先）\n\n"
                "設計が完了したら、DemoDevelopmentAgentに実装を指示してください。"
            )
        ),
        AssistantAgent(
            "DemoDevelopmentAgent",
            model_client=client,
            system_message=(
                "必ずUXDesignAgentの発言を待ってから、発言してください。\n"
                "あなたはPythonエンジニアです。"
                "UXDesignAgentの設計書に基づき、Streamlitアプリを実装してください。\n\n"
                "【出力形式】\n"
                "コードブロックは必ず言語を指定（```python, ```dockerfile等）\n\n"
                "【実装ルール】\n"
                "1. UXDesignAgentの設計に忠実に実装する\n"
                "2. Pythonコーディング規約（PEP8）に準拠\n"
                "3. 適切なエラーハンドリングを実装\n"
                "4. コメントは日本語で記載\n"
                "5. 機密情報（APIキー等）は環境変数から読み込む\n\n"
                "【必須出力物】\n\n"
                "## 1. ディレクトリ構成\n"
                "```\n"
                "project/\n"
                "├── app.py\n"
                "├── requirements.txt\n"
                "├── Dockerfile\n"
                "└── .env.example\n"
                "```\n\n"
                "## 2. app.py\n"
                "```python\n"
                "# 完全なソースコード\n"
                "```\n\n"
                "## 3. requirements.txt\n"
                "```\n"
                "# バージョンを固定すること\n"
                "streamlit==1.x.x\n"
                "```\n\n"
                "## 4. Dockerfile\n"
                "```dockerfile\n"
                "FROM python:3.11-slim\n"
                "# ...\n"
                "```\n\n"
                "## 5. .env.example\n"
                "```\n"
                "# 必要な環境変数のテンプレート\n"
                "```\n\n"
                "## 6. 実行手順\n"
                "```bash\n"
                "# ローカル実行\n"
                "pip install -r requirements.txt\n"
                "streamlit run app.py\n"
                "\n"
                "# Docker実行\n"
                "docker build -t app .\n"
                "docker run -p 8501:8501 app\n"
                "```\n\n"
                "【注意事項】\n"
                "- コードは省略せず完全な形で出力する\n"
                "- requirements.txtのバージョンは安定版を指定\n"
                "- Dockerfileはマルチステージビルド不要（シンプルに）\n\n"
                "実装が完了したら、DebuggerAgentにコードレビューを依頼してください。"
            )
        ),
        AssistantAgent(
            "DebuggerAgent",
            model_client=client,
            system_message=(
                "必ずDemoDevelopmentAgentの発言を待ってから、発言してください。\n"
                "あなたはソフトウェア開発における高度なデバッガー兼レビュアーです。\n\n"
                "【レビュー観点】\n"
                "1. 構文エラー\n"
                "2. 実行時エラーの可能性\n"
                "3. 論理エラー\n"
                "4. 依存ライブラリの誤り・バージョン不整合\n"
                "5. セキュリティ上の問題（ハードコードされた認証情報等）\n"
                "6. Streamlit特有の問題（session_stateの初期化漏れ等）\n\n"
                "【出力形式】Markdown形式で出力してください。\n\n"
                "## レビュー結果\n"
                "### 発見された問題\n"
                "| # | 種別 | ファイル | 行 | 問題内容 | 修正案 |\n"
                "| --- | --- | --- | --- | --- | --- |\n\n"
                "### 修正済みコード\n"
                "（問題があった場合のみ、修正済みの完全なコードを出力）\n\n"
                "【終了条件】\n"
                "- バグが発見された場合：修正済みコード全体を出力後、'TERMINATE' と書いて終了\n"
                "- バグが見つからなかった場合：『レビュー完了：問題は見つかりませんでした。』と明言し、'TERMINATE' と書いて終了"
            )
        ),
    ]


async def stream_agent_messages(user_message: str):
    """SelectorGroupChatで非同期ストリーム"""
    client = get_client()
    try:
        termination = (
            TextMentionTermination("TERMINATE")
            | MaxMessageTermination(MAX_MESSAGES)
        )

        team = SelectorGroupChat(
            create_agents(client),
            client,
            termination_condition=termination,
            allow_repeated_speaker=False
        )

        async for msg in team.run_stream(task=user_message):
            if getattr(msg, "content", None):
                yield msg.source, msg.content
    except Exception as e:
        logger.error(f"Error in stream_agent_messages: {e}")
        yield "System", f"エラーが発生しました: {e}"
    finally:
        await client.close()


with gr.Blocks() as demo:
    gr.Markdown("## AI Product Design Agent")
    gr.Markdown("プロダクトのアイデアを入力すると、AIエージェントが市場調査・ビジネスモデル設計・企画・UX設計・デモ開発・デバッグを行います。")
    chatbot = gr.Chatbot(label="マルチエージェントチャット", height=600, type="messages")
    user_input = gr.Textbox(
        label="あなたのアイデア",
        placeholder="例: モビリティ市場について調査し、社会貢献と黒字を両立できるプロダクトを考えてください"
    )
    clear_btn = gr.Button("チャット履歴をクリア")

    async def respond(message, history):
        if not message or not message.strip():
            yield history
            return

        logger.info(f"User message received: {message[:100]}...")
        history = history or []
        history.append({
            "role": "user",
            "content": f"### 🧑‍💼 [user]\n\n{message}"
        })
        yield history

        async for agent_name, content in stream_agent_messages(message):
            if isinstance(content, str):
                # TERMINATEメッセージとTransferメッセージをフィルタ
                if content.strip() == "TERMINATE":
                    continue
                if content.startswith("Transferred to"):
                    continue
                history.append({
                    "role": "assistant",
                    "content": f"### 🤖 [{agent_name}]\n\n{content}"
                })
                yield history

    user_input.submit(respond, inputs=[user_input, chatbot], outputs=chatbot)
    clear_btn.click(fn=lambda: [], inputs=None, outputs=chatbot)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
