from __future__ import annotations

from . import models
from .agent.config import AgentConfig
from .agent.messages import AgentError

OPENAI_URL = "https://api.openai.com/v1"
ANTHROPIC_URL = "https://api.anthropic.com"

PRESETS: tuple[tuple[str, str, str], ...] = (
    (models.OPENAI, "OpenAI", "https://api.openai.com/v1"),
    (models.OPENAI, "OpenRouter (hundreds of models, one key)", "https://openrouter.ai/api/v1"),
    (models.OPENAI, "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai"),
    (models.OPENAI, "Groq", "https://api.groq.com/openai/v1"),
    (models.OPENAI, "DeepSeek", "https://api.deepseek.com/v1"),
    (models.OPENAI, "xAI (Grok)", "https://api.x.ai/v1"),
    (models.OPENAI, "Mistral", "https://api.mistral.ai/v1"),
    (models.OPENAI, "Together AI", "https://api.together.xyz/v1"),
    (models.OPENAI, "Ollama (on this computer)", "http://localhost:11434/v1"),
    (models.OPENAI, "LM Studio (on this computer)", "http://localhost:1234/v1"),
    (models.ANTHROPIC, "Anthropic (Claude)", "https://api.anthropic.com"),
)


def list_models(provider: str, base_url: str, api_key: str) -> list[models.ModelInfo]:
    if provider == models.ANTHROPIC:
        from .agent import anthropic_agent

        return anthropic_agent.list_models(AgentConfig(provider, base_url, api_key, ""))
    from .agent import openai_agent

    return openai_agent.list_models(base_url, api_key)


def test_connection(cfg: AgentConfig) -> tuple[str, list[models.ModelInfo]]:
    try:
        listed = list_models(cfg.provider, cfg.base_url, cfg.api_key)
    except AgentError as exc:
        if exc.kind != "not_found":
            raise
        listed = []
    model = cfg.model or (models.default_chat(listed) if listed else "")
    if not model:
        return f"Connected, {len(listed)} models listed. Pick a model.", listed
    probe = AgentConfig(cfg.provider, cfg.base_url, cfg.api_key, model, model_info=models.find(listed, model))
    if cfg.provider == models.ANTHROPIC:
        from .agent import anthropic_agent

        anthropic_agent.test_chat(probe)
    else:
        from .agent import openai_agent

        openai_agent.test_chat(probe)
    count = f"{len(models.chat_models(listed))} chat models" if listed else "no model list"
    return f"Connected · {count} · {models.short_label(model, listed)} replied", listed
