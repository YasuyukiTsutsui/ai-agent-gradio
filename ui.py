"""
AI Product Design Agent - マルチエージェントによるプロダクト設計支援ツール
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import gradio as gr
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import (
    TextMentionTermination,
    MaxMessageTermination,
)
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import (
    load_config,
    build_system_message,
    AppConfig,
    ConfigurationError,
    save_mock_project,
    load_mock_projects,
)

load_dotenv()

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 設定読み込み
try:
    config = load_config()
except ConfigurationError as e:
    logger.error(f"設定エラー: {e}")
    raise


@asynccontextmanager
async def get_client(config: AppConfig) -> AsyncGenerator[OpenAIChatCompletionClient, None]:
    """OpenAIクライアントを生成し、使用後にクローズする"""
    if not config.model.api_key:
        raise ConfigurationError("OPENAI_API_KEY が設定されていません")

    client = OpenAIChatCompletionClient(
        model=config.model.name,
        api_key=config.model.api_key
    )
    try:
        yield client
    finally:
        await client.close()


def create_agents(
    client: OpenAIChatCompletionClient,
    config: AppConfig
) -> list[AssistantAgent]:
    """設定に基づいてエージェントを生成する"""
    agents: list[AssistantAgent] = []
    for agent_def in config.agent_definitions:
        try:
            system_message = build_system_message(config, agent_def)
            agent = AssistantAgent(
                agent_def.name,
                model_client=client,
                system_message=system_message
            )
            agents.append(agent)
            logger.info(f"エージェント作成: {agent_def.name}")
        except FileNotFoundError as e:
            logger.error(f"プロンプトファイルエラー: {e}")
            raise
    return agents


async def stream_agent_messages(
    user_message: str,
    config: AppConfig
) -> AsyncGenerator[tuple[str, str], None]:
    """SelectorGroupChatで非同期ストリーム"""
    async with get_client(config) as client:
        try:
            termination = (
                TextMentionTermination("TERMINATE")
                | MaxMessageTermination(config.agent.max_messages)
            )

            agents = create_agents(client, config)
            team = SelectorGroupChat(
                agents,
                client,
                termination_condition=termination,
                allow_repeated_speaker=config.agent.allow_repeated_speaker
            )

            async for msg in team.run_stream(task=user_message):
                if getattr(msg, "content", None):
                    yield msg.source, msg.content

        except ConfigurationError as e:
            logger.error(f"設定エラー: {e}")
            yield "System", f"設定エラー: {e}"
        except Exception as e:
            logger.error(f"エージェント実行エラー: {e}", exc_info=True)
            yield "System", f"エラーが発生しました: {e}"


def create_ui(config: AppConfig) -> gr.Blocks:
    """Gradio UIを生成する"""
    with gr.Blocks(title=config.ui.title) as demo:
        # ヘッダー
        gr.Markdown(f"## {config.ui.title}")
        gr.Markdown(config.ui.description)

        # 状態管理
        current_user_input = gr.State("")

        with gr.Tabs():
            # メインチャットタブ
            with gr.TabItem("チャット"):
                chatbot = gr.Chatbot(
                    label="マルチエージェントチャット",
                    height=config.ui.chatbot_height
                )
                user_input = gr.Textbox(
                    label="あなたのアイデア",
                    placeholder=config.ui.input_placeholder,
                    lines=3
                )

                with gr.Row():
                    submit_btn = gr.Button("送信", variant="primary")
                    clear_btn = gr.Button("チャット履歴をクリア")

                # 保存セクション
                with gr.Accordion("プロジェクトを保存", open=False):
                    project_name = gr.Textbox(
                        label="プロジェクト名",
                        placeholder="例: モビリティサービス企画"
                    )
                    save_btn = gr.Button("保存")
                    save_status = gr.Textbox(
                        label="保存結果",
                        interactive=False
                    )

            # 保存済みプロジェクトタブ
            with gr.TabItem("保存済みプロジェクト"):
                refresh_btn = gr.Button("一覧を更新")
                projects_list = gr.Dataframe(
                    headers=["プロジェクト名", "作成日時", "ユーザー入力"],
                    label="保存済みプロジェクト一覧",
                    interactive=False
                )
                load_project_name = gr.Textbox(
                    label="読み込むプロジェクト名"
                )
                load_btn = gr.Button("プロジェクトを読み込む")

        # イベントハンドラ
        async def respond(
            message: str,
            history: list[dict[str, str]] | None
        ) -> AsyncGenerator[tuple[list[dict[str, str]], str], None]:
            """チャット応答を生成"""
            if not message or not message.strip():
                yield history or [], message
                return

            logger.info(f"ユーザーメッセージ受信: {message[:100]}...")
            history = history or []
            history.append({
                "role": "user",
                "content": f"### [user]\n\n{message}"
            })
            yield history, message

            async for agent_name, content in stream_agent_messages(message, config):
                if isinstance(content, str):
                    # TERMINATEメッセージとTransferメッセージをフィルタ
                    if content.strip() == "TERMINATE":
                        continue
                    if content.startswith("Transferred to"):
                        continue
                    history.append({
                        "role": "assistant",
                        "content": f"### [{agent_name}]\n\n{content}"
                    })
                    yield history, message

        def save_project(
            name: str,
            history: list[dict[str, str]] | None,
            user_msg: str
        ) -> str:
            """プロジェクトを保存"""
            if not name or not name.strip():
                return "エラー: プロジェクト名を入力してください"
            if not history:
                return "エラー: 保存するチャット履歴がありません"

            try:
                output_path = save_mock_project(
                    config,
                    name.strip(),
                    user_msg,
                    history
                )
                return f"保存完了: {output_path.name}"
            except Exception as e:
                logger.error(f"保存エラー: {e}", exc_info=True)
                return f"保存エラー: {e}"

        def refresh_projects() -> list[list[str]]:
            """プロジェクト一覧を更新"""
            try:
                projects = load_mock_projects(config)
                return [
                    [p.name, p.created_at, p.user_input[:50] + "..." if len(p.user_input) > 50 else p.user_input]
                    for p in projects
                ]
            except Exception as e:
                logger.error(f"プロジェクト一覧取得エラー: {e}")
                return []

        def load_project(
            name: str,
            current_history: list[dict[str, str]] | None
        ) -> tuple[list[dict[str, str]], str]:
            """プロジェクトを読み込む"""
            if not name or not name.strip():
                return current_history or [], ""

            try:
                projects = load_mock_projects(config)
                for project in projects:
                    if project.name == name.strip():
                        return project.chat_history, project.user_input
                return current_history or [], f"プロジェクト '{name}' が見つかりません"
            except Exception as e:
                logger.error(f"プロジェクト読み込みエラー: {e}")
                return current_history or [], f"読み込みエラー: {e}"

        # イベントバインディング
        submit_btn.click(
            respond,
            inputs=[user_input, chatbot],
            outputs=[chatbot, current_user_input]
        )
        user_input.submit(
            respond,
            inputs=[user_input, chatbot],
            outputs=[chatbot, current_user_input]
        )
        clear_btn.click(
            fn=lambda: ([], ""),
            inputs=None,
            outputs=[chatbot, current_user_input]
        )

        save_btn.click(
            save_project,
            inputs=[project_name, chatbot, current_user_input],
            outputs=save_status
        )

        refresh_btn.click(
            refresh_projects,
            inputs=None,
            outputs=projects_list
        )
        load_btn.click(
            load_project,
            inputs=[load_project_name, chatbot],
            outputs=[chatbot, user_input]
        )

        # 初期読み込み
        demo.load(refresh_projects, outputs=projects_list)

    return demo


def main() -> None:
    """アプリケーションのエントリーポイント"""
    try:
        config.validate()
    except ConfigurationError as e:
        logger.error(f"設定検証エラー: {e}")
        print(f"エラー: {e}")
        return

    demo = create_ui(config)
    demo.launch(
        share=config.server.share,
        server_name=config.server.host,
        server_port=config.server.port
    )


if __name__ == "__main__":
    main()
