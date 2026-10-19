from __future__ import annotations

import base64
import urllib.request
from dataclasses import dataclass
from typing import Any

from .. import models

TEXTURE_PROMPT = (
    "Seamless tileable texture of {subject}. The surface fills the entire square frame, seen straight on "
    "from directly above (orthographic, no perspective, no horizon). Flat, even, shadowless lighting with no "
    "highlights, reflections or vignette, so it works as a PBR albedo map. No objects on top, no text, no "
    "borders. Realistic, high-frequency surface detail."
)

_ROUTES: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class ImageConfig:
    base_url: str
    api_key: str
    model: str
    size: str = "1024x1024"
    timeout: float = 300.0
    image_only: bool = False


class ImageAPIError(Exception): ...


def texture_prompt(subject: str) -> str:
    return TEXTURE_PROMPT.format(subject=subject.strip().rstrip("."))


def _client(cfg: ImageConfig) -> Any:
    from ..agent.openai_agent import sdk_client

    return sdk_client(cfg.base_url, cfg.api_key, timeout=cfg.timeout)


def _error(exc: BaseException, cfg: ImageConfig) -> ImageAPIError:
    from ..agent.openai_agent import api_root, map_error

    return ImageAPIError(str(map_error(exc, api_root(cfg.base_url))))


def _decode_item(item: Any) -> bytes:
    b64 = getattr(item, "b64_json", None)
    if b64:
        return base64.b64decode(b64)
    url = getattr(item, "url", None)
    if url:
        return _download(url)
    raise ImageAPIError("The image API response had no image data.")


def _download(url: str) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.read()
    except OSError as exc:
        raise ImageAPIError(f"Couldn't download the generated image: {exc}") from exc


def _via_images(cfg: ImageConfig, prompt: str, reference: bytes | None) -> bytes:
    import openai

    client = _client(cfg)
    size = cfg.size if cfg.size and cfg.size != "auto" else None
    for attempt in range(2):
        kwargs: dict[str, Any] = {"model": cfg.model, "prompt": prompt, "n": 1}
        if size:
            kwargs["size"] = size
        try:
            if reference is not None:
                result = client.images.edit(image=("reference.png", reference, "image/png"), **kwargs)
            else:
                result = client.images.generate(**kwargs)
            return _decode_item(result.data[0])
        except openai.BadRequestError as exc:
            if attempt == 0 and size and "size" in str(exc).lower():
                size = None
                continue
            if attempt == 0 and reference is not None:
                reference = None
                continue
            raise
    raise ImageAPIError("The image API kept rejecting the request.")


def _via_chat(cfg: ImageConfig, prompt: str, reference: bytes | None) -> bytes:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if reference is not None:
        url = "data:image/png;base64," + base64.standard_b64encode(reference).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": url}})
    import openai

    modalities = ["image"] if cfg.image_only else ["image", "text"]
    try:
        response = _client(cfg).chat.completions.create(model=cfg.model, messages=[{"role": "user", "content": content}], extra_body={"modalities": modalities})
    except openai.BadRequestError:
        other = ["image", "text"] if cfg.image_only else ["image"]
        response = _client(cfg).chat.completions.create(model=cfg.model, messages=[{"role": "user", "content": content}], extra_body={"modalities": other})
    message = response.choices[0].message
    extra = getattr(message, "model_extra", None) or {}
    for image in extra.get("images") or []:
        url = (image.get("image_url") or {}).get("url") if isinstance(image, dict) else None
        if url:
            return _download(url)
    for part in message.content if isinstance(message.content, list) else []:
        if isinstance(part, dict) and part.get("type") == "image_url":
            return _download(part["image_url"]["url"])
    raise ImageAPIError(f"{cfg.model} answered without an image. Pick an image model.")


def generate(cfg: ImageConfig, prompt: str, reference: bytes | None = None) -> bytes:
    import openai

    if not cfg.model:
        raise ImageAPIError("No image model is selected. Pick one on the prompt bar or in settings.")
    from ..agent.openai_agent import api_root

    route_key = (api_root(cfg.base_url), cfg.model)
    route = _ROUTES.get(route_key, "images")
    try:
        if route == "images":
            try:
                data = _via_images(cfg, prompt, reference)
            except (openai.NotFoundError, openai.UnprocessableEntityError) as exc:
                try:
                    data = _via_chat(cfg, prompt, reference)
                except openai.APIError:
                    raise exc from None
                route = "chat"
        else:
            data = _via_chat(cfg, prompt, reference)
    except ImageAPIError:
        raise
    except Exception as exc:
        raise _error(exc, cfg) from exc
    _ROUTES[route_key] = route
    return data


def test_connection(cfg: ImageConfig) -> tuple[str, list[models.ModelInfo]]:
    from ..agent.openai_agent import list_models

    try:
        listed = list_models(cfg.base_url, cfg.api_key)
    except Exception as exc:
        raise ImageAPIError(str(exc)) from exc
    found = models.image_models(listed)
    if not found:
        return "Connected, but this endpoint lists no image models.", listed
    note = "" if not cfg.model or models.find(listed, cfg.model) else f"; '{cfg.model}' isn't listed"
    return f"Connected · {len(found)} image models{note}", listed
