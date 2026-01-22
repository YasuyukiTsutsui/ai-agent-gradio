"""
アプリケーション設定モジュール

環境変数から設定を読み込み、一元管理する。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

# 定数定義
DEFAULT_MODEL: Final[str] = "gpt-4o"
DEFAULT_MAX_MESSAGES: Final[int] = 80
DEFAULT_SERVER_PORT: Final[int] = 7860
DEFAULT_CHATBOT_HEIGHT: Final[int] = 600

# 環境変数名
ENV_OPENAI_API_KEY: Final[str] = "OPENAI_API_KEY"
ENV_MODEL_NAME: Final[str] = "MODEL_NAME"
ENV_MAX_MESSAGES: Final[str] = "MAX_MESSAGES"
ENV_SERVER_PORT: Final[str] = "SERVER_PORT"
ENV_SHARE: Final[str] = "SHARE"


class ConfigurationError(Exception):
    """設定エラー用の例外"""
    pass


@dataclass
class ModelConfig:
    """LLMモデル関連の設定"""
    name: str = DEFAULT_MODEL
    api_key: str = ""

    def validate(self) -> None:
        """設定を検証する"""
        if not self.api_key:
            raise ConfigurationError(
                f"{ENV_OPENAI_API_KEY} が設定されていません。"
                ".envファイルを確認してください。"
            )
        if not self.api_key.startswith("sk-"):
            raise ConfigurationError(
                f"{ENV_OPENAI_API_KEY} の形式が無効です。"
                "sk-で始まるAPIキーを設定してください。"
            )


@dataclass
class AgentConfig:
    """エージェント関連の設定"""
    max_messages: int = DEFAULT_MAX_MESSAGES
    allow_repeated_speaker: bool = False


@dataclass
class UIConfig:
    """UI関連の設定"""
    title: str = "AI Product Design Agent"
    description: str = (
        "プロダクトのアイデアを入力すると、AIエージェントが"
        "市場調査・ビジネスモデル設計・企画・UX設計・デモ開発・デバッグを行います。"
    )
    chatbot_height: int = DEFAULT_CHATBOT_HEIGHT
    input_placeholder: str = (
        "例: モビリティ市場について調査し、社会貢献と黒字を両立できるプロダクトを考えてください"
    )


@dataclass
class ServerConfig:
    """サーバー関連の設定"""
    host: str = "0.0.0.0"
    port: int = DEFAULT_SERVER_PORT
    share: bool = False


@dataclass
class StorageConfig:
    """保存機能関連の設定"""
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "outputs")

    def ensure_output_dir(self) -> None:
        """出力ディレクトリを作成する"""
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AgentDefinition:
    """エージェント定義"""
    name: str
    prompt_file: str
    wait_for: str | None = None
    next_agent: str | None = None
    is_first: bool = False
    is_last: bool = False


@dataclass
class AppConfig:
    """アプリケーション全体の設定"""
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    prompts_dir: Path = field(default_factory=lambda: Path(__file__).parent / "prompts")

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

    def validate(self) -> None:
        """全設定を検証する"""
        self.model.validate()


def load_config() -> AppConfig:
    """環境変数から設定を読み込んでAppConfigを生成する"""
    config = AppConfig()

    # モデル設定
    config.model.name = os.getenv(ENV_MODEL_NAME, config.model.name)
    config.model.api_key = os.getenv(ENV_OPENAI_API_KEY, "")

    # エージェント設定
    max_messages_str = os.getenv(ENV_MAX_MESSAGES, str(config.agent.max_messages))
    try:
        config.agent.max_messages = int(max_messages_str)
    except ValueError:
        raise ConfigurationError(
            f"{ENV_MAX_MESSAGES} は整数である必要があります: {max_messages_str}"
        )

    # サーバー設定
    port_str = os.getenv(ENV_SERVER_PORT, str(config.server.port))
    try:
        config.server.port = int(port_str)
    except ValueError:
        raise ConfigurationError(
            f"{ENV_SERVER_PORT} は整数である必要があります: {port_str}"
        )
    config.server.share = os.getenv(ENV_SHARE, "false").lower() == "true"

    return config


def load_prompt(config: AppConfig, prompt_file: str) -> str:
    """プロンプトファイルを読み込む"""
    prompt_path = config.prompts_dir / prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"プロンプトファイルが見つかりません: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def build_system_message(config: AppConfig, agent_def: AgentDefinition) -> str:
    """エージェント定義からsystem_messageを構築する"""
    parts: list[str] = []

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


@dataclass
class MockProject:
    """保存されたモックプロジェクト"""
    name: str
    created_at: str
    user_input: str
    chat_history: list[dict[str, str]]

    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return {
            "name": self.name,
            "created_at": self.created_at,
            "user_input": self.user_input,
            "chat_history": self.chat_history
        }

    @classmethod
    def from_dict(cls, data: dict) -> MockProject:
        """辞書から生成"""
        return cls(
            name=data["name"],
            created_at=data["created_at"],
            user_input=data["user_input"],
            chat_history=data["chat_history"]
        )


def save_mock_project(
    config: AppConfig,
    name: str,
    user_input: str,
    chat_history: list[dict[str, str]]
) -> Path:
    """モックプロジェクトを保存する"""
    config.storage.ensure_output_dir()

    # ファイル名をサニタイズ
    safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}.json"

    project = MockProject(
        name=name,
        created_at=datetime.now().isoformat(),
        user_input=user_input,
        chat_history=chat_history
    )

    output_path = config.storage.output_dir / filename
    output_path.write_text(
        json.dumps(project.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return output_path


def load_mock_projects(config: AppConfig) -> list[MockProject]:
    """保存されたモックプロジェクト一覧を取得する"""
    config.storage.ensure_output_dir()

    projects: list[MockProject] = []
    for json_file in config.storage.output_dir.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            projects.append(MockProject.from_dict(data))
        except (json.JSONDecodeError, KeyError):
            continue

    # 作成日時の降順でソート
    projects.sort(key=lambda p: p.created_at, reverse=True)
    return projects


def load_mock_project_by_name(config: AppConfig, name: str) -> MockProject | None:
    """名前でモックプロジェクトを検索して読み込む"""
    projects = load_mock_projects(config)
    for project in projects:
        if project.name == name:
            return project
    return None
