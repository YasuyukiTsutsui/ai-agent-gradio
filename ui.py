import os, asyncio, gradio as gr
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

import logging
logger = logging.getLogger(__name__)


load_dotenv()

def get_client():
    return OpenAIChatCompletionClient(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

def create_agents(client):
    return [
        AssistantAgent(
            "MarketResearchAgent",
            model_client=client,
            system_message=(
                "あなたは市場調査のエキスパートです。ユーザーのヒアリング結果を分析し、"
                "その後、ProductPlanningAgentに企画案をつなげてください。"
            )
        ),
        AssistantAgent(
            "ProductPlanningAgent",
            model_client=client,
            system_message=(
                "市場調査を受けてプロダクト企画を立ててください。具体的なデモアプリ開発案を "
                "DemoDevelopmentAgentに引き継いでください。"
                "デモアプリの内容について、DemoDevelopmentAgentへフィードバックしてください。"
                "ただし、ユーザーフィードバックはデモアプリ作成後に行うので必要ありません。"
            )
        ),
        AssistantAgent(
            "DemoDevelopmentAgent",
            model_client=client,
            system_message=(
                "企画案に基づいて最小限の機能を実装したデモアプリを設計してください。また、使用するライブラリは新しいバージョンを使ってください。"
                "デモアプリの概要について、ProductPlanningAgentからフィードバックを受けてください。"
                " ProductPlanningAgentから良い評価が得られたら、Dockerfileで実行できるstreamlitアプリの実装を行なってください。"
                "デモアプリの実装が完了したら、ソースコードと実行のためのDockerfileを記載してください。"
            )
        ),
        AssistantAgent(
            "DebuggerAgent",
            model_client=client,
            system_message=(
                "あなたはソフトウェア開発における高度なデバッガー兼レビュアーです。"
                "DemoDevelopmentAgent から提供されるコードに対してコードレビューを行い、"
                "構文エラー、実行時エラー、論理エラー、依存ライブラリの誤りなどを確認してください。"

                "バグが発見された場合は、修正済みのコード全体を出力してください。その後、'TERMINATE' とだけ書いて応答を終了してください。"

                "バグが見つからなかった場合は「バグは見つかりませんでした。」と明言した上で、"
                "'TERMINATE' とだけ書いて応答を終了してください。"

                "コードの一部のみを出力したり、'TERMINATE' を書き忘れたりしないでください。"
            )
        )
    ]

# 非同期エージェント応答ストリーミング
async def stream_agent_messages(user_message):
    client = get_client()
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=50)
    team = RoundRobinGroupChat(create_agents(client), termination_condition=termination)

    async for msg in team.run_stream(task=user_message):
        if hasattr(msg, "content") and msg.content:
            yield msg.source, msg.content  # エージェント名, 発言内容

    await client.close()

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤝 Multi-Agent チャットデモ")

    with gr.Row():
        chatbot = gr.Chatbot(
            label="マルチエージェントチャット",
            height=600,
            type="messages"  # ← 推奨形式
        )

    user_input = gr.Textbox(
        label="あなたの質問",
        placeholder="例: モビリティ市場について調査してください",
    )

    clear_btn = gr.Button("チャット履歴をクリア")

    # 応答関数
    async def respond(message, history):
        logger.info("💬 User message received: %s", message)

        history = history or []

        # 最初のユーザー発言を1回だけ表示（Markdownで整形）
        if len(history) == 0:
            user_block = f"### 🧑‍💼 [user]\n\n{message}"
            history.append({"role": "user", "content": user_block})
            logger.info("🧑‍💼 User message added to history")
            yield history

        # エージェントごとの発言を Markdown 形式でブロック表示
        async for agent_name, content in stream_agent_messages(message):
            agent_block = f"### 🤖 [{agent_name}]\n\n{content.strip()}"
            history.append({"role": "assistant", "content": agent_block})
            logger.info("🤖 [%s] generated message.", agent_name)
            yield history


    user_input.submit(respond, inputs=[user_input, chatbot], outputs=chatbot)
    clear_btn.click(fn=lambda: [], inputs=None, outputs=chatbot)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
