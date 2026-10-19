from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .. import models

if TYPE_CHECKING:
    from .loop import Agent


@dataclass(frozen=True)
class AgentConfig:
    provider: str
    base_url: str
    api_key: str
    model: str
    reasoning: str = "auto"
    max_steps: int = 40
    model_info: models.ModelInfo | None = None
    send_images: bool = True
    use_fallbacks: bool = True

    @property
    def endpoint(self) -> tuple[Any, ...]:
        return (self.provider, self.base_url, self.api_key)

    @property
    def effort_key(self) -> str:
        return models.effort_key(self.base_url, self.model)

    @property
    def efforts(self) -> tuple[str, ...]:
        return models.offered_efforts(self.provider, self.model, self.model_info, self.effort_key)

    def effort(self) -> str:
        return models.effective_effort(self.reasoning, self.efforts)

    def reject_effort(self, level: str) -> str:
        models.reject_effort(self.effort_key, level)
        now = self.effort()
        return f"This model doesn't take {models.LEVEL_LABELS[level]} reasoning here; " + (
            f"using {models.LEVEL_LABELS[now]}." if now != "auto" else "using its default."
        )

    @property
    def context_window(self) -> int:
        info = self.model_info
        return info.context if info is not None and info.context else models.DEFAULT_CONTEXT.get(self.provider, 128_000)

    @property
    def vision(self) -> bool:
        if not self.send_images:
            return False
        info = self.model_info
        return info is None or info.vision is not False


def make_agent(cfg: AgentConfig) -> Agent:
    if cfg.provider == models.ANTHROPIC:
        from .anthropic_agent import AnthropicAgent

        return AnthropicAgent(cfg)
    from .openai_agent import OpenAIAgent

    return OpenAIAgent(cfg)
