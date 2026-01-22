"""
アプリケーション設定モジュール

環境変数から設定を読み込み、一元管理する。
"""
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """LLMモデル関連の設定"""
    name: str = "gpt-4o"
    api_key: str = ""


@dataclass
class AgentConfig:
    """エージェント関連の設定"""
    max_messages: int = 80
    allow_repeated_speaker: bool = False


@dataclass
class UIConfig:
    """UI関連の設定"""
    title: str = "AI Product Design Agent"
    description: str = "プロダクトのアイデアを入力すると、AIエージェントが市場調査・ビジネスモデル設計・企画・UX設計・デモ開発・デバッグを行います。"
    chatbot_height: int = 600
    input_placeholder: str = "例: モビリティ市場について調査し、社会貢献と黒字を両立できるプロダクトを考えてください"


@dataclass
class ServerConfig:
    """サーバー関連の設定"""
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False


@dataclass
class AgentDefinition:
    """エージェント定義"""
    name: str
    prompt_file: str
    wait_for: str | None = None  # このエージェントの発言前に待つエージェント名
    next_agent: str | None = None  # 次に指示を出すエージェント名
    is_first: bool = False  # 最初に発言するエージェントか
    is_last: bool = False  # 最後のエージェントか（TERMINATEを出力）


@dataclass
class AppConfig:
    """アプリケーション全体の設定"""
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    prompts_dir: Path = field(default_factory=lambda: Path(__file__).parent / "prompts")

    # エージェント定義（順序が重要）
    agent_definitions: list[AgentDefinition] = field(default_factory=lambda: [
        AgentDefinition(
            name="MarketResearchAgent",
            prompt_file="market_research.txt",
            wait_for=None,
            next_agent="BusinessModelAgent",
            is_first=True,
        ),
        AgentDefinition(
            name="BusinessModelAgent",
            prompt_file="business_model.txt",
            wait_for="MarketResearchAgent",
            next_agent="ProductPlanningAgent",
        ),
        AgentDefinition(
            name="ProductPlanningAgent",
            prompt_file="product_planning.txt",
            wait_for="BusinessModelAgent",
            next_agent="UXDesignAgent",
        ),
        AgentDefinition(
            name="UXDesignAgent",
            prompt_file="ux_design.txt",
            wait_for="ProductPlanningAgent",
            next_agent="DemoDevelopmentAgent",
        ),
        AgentDefinition(
            name="DemoDevelopmentAgent",
            prompt_file="demo_development.txt",
            wait_for="UXDesignAgent",
            next_agent="DebuggerAgent",
        ),
        AgentDefinition(
            name="DebuggerAgent",
            prompt_file="debugger.txt",
            wait_for="DemoDevelopmentAgent",
            next_agent=None,
            is_last=True,
        ),
    ])


def load_config() -> AppConfig:
    """環境変数から設定を読み込んでAppConfigを生成する"""
    config = AppConfig()

    # モデル設定
    config.model.name = os.getenv("MODEL_NAME", config.model.name)
    config.model.api_key = os.getenv("OPENAI_API_KEY", "")

    # エージェント設定
    config.agent.max_messages = int(os.getenv("MAX_MESSAGES", str(config.agent.max_messages)))

    # サーバー設定
    config.server.port = int(os.getenv("SERVER_PORT", str(config.server.port)))
    config.server.share = os.getenv("SHARE", "false").lower() == "true"

    return config


def load_prompt(config: AppConfig, prompt_file: str) -> str:
    """プロンプトファイルを読み込む"""
    prompt_path = config.prompts_dir / prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def build_system_message(config: AppConfig, agent_def: AgentDefinition) -> str:
    """エージェント定義からsystem_messageを構築する"""
    parts = []

    # 発言順序の指示
    if agent_def.is_first:
        parts.append("必ず最初に発言してください。")
    elif agent_def.wait_for:
        parts.append(f"必ず{agent_def.wait_for}の発言を待ってから、発言してください。")

    # プロンプト本文
    prompt = load_prompt(config, agent_def.prompt_file)
    parts.append(prompt)

    # 次エージェントへの引き継ぎ指示
    if agent_def.next_agent:
        parts.append(f"\n分析・設計が完了したら、{agent_def.next_agent}に引き継いでください。")

    return "\n".join(parts)
