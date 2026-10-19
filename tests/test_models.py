from __future__ import annotations

import pytest

from blender_ai import models

OPENAI_LIST = {
    "object": "list",
    "data": [
        {"id": "gpt-5.1", "object": "model"},
        {"id": "gpt-5-mini", "object": "model"},
        {"id": "gpt-image-1", "object": "model"},
        {"id": "dall-e-3", "object": "model"},
        {"id": "text-embedding-3-large", "object": "model"},
        {"id": "whisper-1", "object": "model"},
        {"id": "tts-1", "object": "model"},
        {"id": "gpt-4o", "object": "model"},
    ],
}

OPENROUTER_LIST = {
    "data": [
        {
            "id": "anthropic/claude-sonnet-5.5",
            "name": "Anthropic: Claude Sonnet 5.5",
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            "supported_parameters": ["tools", "reasoning", "max_tokens"],
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        },
        {
            "id": "google/gemma-4-31b-it:free",
            "architecture": {"input_modalities": ["image", "text"], "output_modalities": ["text"]},
            "supported_parameters": ["tools"],
            "pricing": {"prompt": "0", "completion": "0"},
        },
        {
            "id": "google/gemini-2.5-flash-image",
            "name": "Google: Nano Banana",
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["image", "text"]},
            "supported_parameters": ["max_tokens"],
        },
        {"id": "some/text-only", "name": "Text only", "architecture": {"modality": "text->text"}, "supported_parameters": ["max_tokens"]},
    ]
}


def test_openai_list_is_sorted_into_chat_and_image() -> None:
    listed = models.parse_models(OPENAI_LIST)
    chat = [m.id for m in models.chat_models(listed)]
    image = [m.id for m in models.image_models(listed)]
    assert chat == ["gpt-4o", "gpt-5-mini", "gpt-5.1"]
    assert image == ["dall-e-3", "gpt-image-1"]
    assert models.default_chat(listed) == "gpt-5.1"
    assert models.default_image(listed) == "gpt-image-1"


def test_openrouter_modalities_and_tool_support() -> None:
    listed = models.parse_models(OPENROUTER_LIST)
    sonnet = models.find(listed, "anthropic/claude-sonnet-5.5")
    assert sonnet is not None and sonnet.vision is True and sonnet.tools is True
    assert [m.id for m in models.image_models(listed)] == ["google/gemini-2.5-flash-image"]
    assert [m.id for m in models.chat_models(listed)] == ["anthropic/claude-sonnet-5.5", "google/gemma-4-31b-it:free"]
    assert sonnet.free is False and models.find(listed, "google/gemma-4-31b-it:free").free is True
    assert models.short_label("anthropic/claude-sonnet-5.5", listed) == "Sonnet 5.5"
    assert models.short_label("vendor/custom-model") == "custom-model"
    text_only = models.find(listed, "some/text-only")
    assert text_only is not None and text_only.vision is False


def test_anthropic_list_with_capabilities() -> None:
    payload = {
        "data": [
            {
                "type": "model",
                "id": "claude-opus-5-5",
                "display_name": "Claude Opus 5.5",
                "max_tokens": 128000,
                "capabilities": {
                    "thinking": {"types": {"adaptive": {"supported": True}, "disabled": {"supported": False}}},
                    "image_input": {"supported": True},
                },
            }
        ]
    }
    (info,) = models.parse_models(payload)
    assert info.label == "Claude Opus 5.5" and info.vision is True and info.max_output == 128000
    caps = models.anthropic_caps(info.id, info)
    assert caps.thinking == "always" and caps.max_output == 128000


def test_bare_lists_and_garbage() -> None:
    assert [m.id for m in models.parse_models(["b", "a", "a"])] == ["a", "b"]
    assert models.parse_models({"models": [{"name": "llama3"}]})[0].id == "llama3"
    assert models.parse_models("nope") == []
    assert models.default_chat([]) == ""


@pytest.mark.parametrize(
    ("model_id", "thinking"),
    [
        ("claude-opus-5-5", "always"),
        ("claude-sonnet-5-5-20260901", "adaptive"),
        ("anthropic/claude-opus-4.8", "explicit"),
        ("claude-haiku-4-5-20251001", "budget"),
        ("claude-some-future-model", "adaptive"),
        ("llama-3", "none"),
    ],
)
def test_family_rules(model_id: str, thinking: str) -> None:
    assert models.anthropic_caps(model_id).thinking == thinking


def test_anthropic_params() -> None:
    always = models.anthropic_caps("claude-opus-5-5")
    assert models.anthropic_params(always, "auto") == {"max_tokens": 32000}
    high = models.anthropic_params(always, "high")
    assert high["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert high["output_config"] == {"effort": "high"}
    budget = models.anthropic_params(models.anthropic_caps("claude-haiku-4-5"), "medium", 4000)
    assert budget["thinking"] == {"type": "enabled", "budget_tokens": 8000}
    assert budget["max_tokens"] >= 16000 and "output_config" not in budget
    assert models.anthropic_params(models.anthropic_caps("llama"), "high") == {"max_tokens": 32000}
