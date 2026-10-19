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
