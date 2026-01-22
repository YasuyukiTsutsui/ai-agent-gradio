"""
AI Product Design Agent - マルチエージェントによるプロダクト設計支援ツール
"""
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

from config import load_config, build_system_message, AppConfig

load_dotenv()

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 設定読み込み
config = load_config()


def get_client(config: AppConfig) -> OpenAIChatCompletionClient:
    """OpenAIクライアントを生成する"""
    if not config.model.api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAIChatCompletionClient(
        model=config.model.name,
        api_key=config.model.api_key
    )


def create_agents(client: OpenAIChatCompletionClient, config: AppConfig) -> list[AssistantAgent]:
    """設定に基づいてエージェントを生成する"""
    agents = []
    for agent_def in config.agent_definitions:
        system_message = build_system_message(config, agent_def)
        agent = AssistantAgent(
            agent_def.name,
            model_client=client,
            system_message=system_message
        )
        agents.append(agent)
        logger.info(f"Created agent: {agent_def.name}")
    return agents


async def stream_agent_messages(user_message: str, config: AppConfig):
    """SelectorGroupChatで非同期ストリーム"""
    client = get_client(config)
    try:
        termination = (
            TextMentionTermination("TERMINATE")
            | MaxMessageTermination(config.agent.max_messages)
        )

        team = SelectorGroupChat(
            create_agents(client, config),
            client,
            termination_condition=termination,
            allow_repeated_speaker=config.agent.allow_repeated_speaker
        )

        async for msg in team.run_stream(task=user_message):
            if getattr(msg, "content", None):
                yield msg.source, msg.content
    except Exception as e:
        logger.error(f"Error in stream_agent_messages: {e}")
        yield "System", f"エラーが発生しました: {e}"
    finally:
        await client.close()


def create_ui(config: AppConfig) -> gr.Blocks:
    """Gradio UIを生成する"""
    with gr.Blocks() as demo:
        gr.Markdown(f"## {config.ui.title}")
        gr.Markdown(config.ui.description)
        chatbot = gr.Chatbot(
            label="マルチエージェントチャット",
            height=config.ui.chatbot_height,
            type="messages"
        )
        user_input = gr.Textbox(
            label="あなたのアイデア",
            placeholder=config.ui.input_placeholder
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

            async for agent_name, content in stream_agent_messages(message, config):
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

    return demo


if __name__ == "__main__":
    demo = create_ui(config)
    demo.launch(
        share=config.server.share,
        server_name=config.server.host,
        server_port=config.server.port
    )
