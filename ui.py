import os, asyncio, gradio as gr
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

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
                "市場調査を受けてプロダクト企画を立ててください。具体的なモック開発案を "
                "MockDevelopmentAgentに引き継いでください。"
                "モックの内容について、MockDevelopmentAgentへフィードバックしてください"
                "ただし、ユーザーフィードバックはモック作成後に行うので必要ありません"
            )
        ),
        AssistantAgent(
            "MockDevelopmentAgent",
            model_client=client,
            system_message=(
                "企画案に基づいてモックの概要を設計してください。また、使用するライブラリは新しいバージョンを使ってください。"
                "モックの概要について、ProductPlanningAgentからフィードバックを受けてください"
                " ProductPlanningAgentから良い評価が得られたら、Dockerfileで実行できるstreamlitアプリの実装を行なってください"
                "モックの実装が完了したら、ソースコードと実行のためのDockerfileを記載してください"
                "'TERMINATE' とだけ書いて応答を終了してください。"
            )
        )
    ]

async def run_team(task: str):
    client = get_client()
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=50)
    team = RoundRobinGroupChat(create_agents(client), termination_condition=termination)
    stream = team.run_stream(task=task)
    result = await Console(stream)
    await client.close()
    return result

def chat_function(message, history):
    result = asyncio.run(run_team(message))
    return [
        gr.ChatMessage(role="assistant",
                       content=msg.content,
                       metadata={"title": msg.source})
        for msg in result.messages if hasattr(msg, "content")
    ]

iface = gr.ChatInterface(fn=chat_function, type="messages")
if __name__ == "__main__":
    iface.launch(share=False, server_name="0.0.0.0", server_port=7860)
