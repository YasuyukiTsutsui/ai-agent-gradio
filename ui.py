import os, asyncio, gradio as gr, logging
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import (
    TextMentionTermination,
    MaxMessageTermination,
)
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

logger = logging.getLogger(__name__)
load_dotenv()

# ── OpenAI クライアント ─────────────────────────────────────
def get_client():
    return OpenAIChatCompletionClient(
        model="gpt-4o",                     # 利用可能な ID に置換可
        api_key=os.getenv("OPENAI_API_KEY")
    )

# ── エージェント定義（元コードと同じ） ─────────────────────
def create_agents(client):
    return [
        AssistantAgent(
            "MarketResearchAgent",
            model_client=client,
            system_message=(
                "必ず最初に発言してください。"
                "あなたは市場調査のエキスパートです。ユーザーのヒアリング結果を分析し、"
                "その後、ProductPlanningAgentに市場調査をもとにしたプロダクト企画を指示してください。"
                "プロダクト企画の内容について、ProductPlanningAgentへフィードバックしてください。"
                "プロダクト企画が良ければ企画の承認を出してください。"
            )
        ),
        AssistantAgent(
            "ProductPlanningAgent",
            model_client=client,
            system_message=(
                "必ずMarketResearchAgentの指示を待ってから、発言してください。"
                "プロダクト企画が終わったら、MarketResearchAgentからフィードバックを受けてください。"
                "MarketResearchAgentから企画の承認が得られたら、具体的なデモアプリ開発案を"
                "DemoDevelopmentAgentに指示してください。"
                "ただし、ユーザーフィードバックはデモアプリ作成後に行うので必要ありません。"
            )
        ),
        AssistantAgent(
            "DemoDevelopmentAgent",
            model_client=client,
            system_message=(
                "必ずProductPlanningAgentの指示を待ってから、発言してください。"
                "企画案に基づいて最小限の機能を実装したデモアプリを設計してください。"
                "使用するライブラリは新しいバージョンを使ってください。"
                "デモアプリの概要について、ProductPlanningAgentからフィードバックを受けてください。"
                "ProductPlanningAgentから開発の承認が得られたら、Dockerfileで実行できる"
                "streamlitアプリの実装を行なってください。"
                "実装が完了したら、ソースコードとDockerfileを共有してください。"
            )
        ),
        AssistantAgent(
            "DebuggerAgent",
            model_client=client,
            system_message=(
                "必ずDemoDevelopmentAgentの発言を待ってから、発言してください。"
                "あなたはソフトウェア開発における高度なデバッガー兼レビュアーです。"
                "DemoDevelopmentAgentから提供されるコードとDockerfileをレビューし、"
                "構文エラー、実行時エラー、論理エラー、依存ライブラリの誤りなどを確認してください。"
                "バグが発見された場合は、修正済みのコード全体を出力してください。"
                "その後、'TERMINATE' とだけ書いて応答を終了してください。"
                "バグが見つからなかった場合は『バグは見つかりませんでした。』と明言し、"
                "'TERMINATE' とだけ書いて応答を終了してください。"
            )
        ),
    ]

# ── SelectorGroupChat で非同期ストリーム ───────────────────
async def stream_agent_messages(user_message: str):
    client = get_client()
    termination = (
        TextMentionTermination("TERMINATE")
        | MaxMessageTermination(50)
    )

    team = SelectorGroupChat(
        create_agents(client),
        client,
        termination_condition=termination,
        allow_repeated_speaker=False            # ★ 直前スピーカーは除外
    )

    async for msg in team.run_stream(task=user_message):
        if getattr(msg, "content", None):
            yield msg.source, msg.content

    await client.close()

# --------------------------------------------------------------------
# Gradio UI (変更なし)
# --------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🤝 AIエージェントくん チャットデモ")
    chatbot = gr.Chatbot(label="マルチエージェントチャット", height=600, type="messages")
    user_input = gr.Textbox(label="あなたの質問",
        placeholder="例: モビリティ市場について調査し、社会貢献と黒字を両立できるプロダクトを考えて下さい")
    clear_btn = gr.Button("チャット履歴をクリア")

    async def respond(message, history):
        logger.info("💬 User message received: %s", message)
        history = history or []
        if len(history) == 0:
            history.append({"role": "user",
                            "content": f"### 🧑‍💼 [user]\n\n{message}"})
            yield history
        async for agent_name, content in stream_agent_messages(message):
            if isinstance(content, str) and not content.startswith("Transferred to"):
                history.append({"role": "assistant",
                                "content": f"### 🤖 [{agent_name}]\n\n{content}"})
                yield history

    user_input.submit(respond, inputs=[user_input, chatbot], outputs=chatbot)
    clear_btn.click(fn=lambda: [], inputs=None, outputs=chatbot)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
