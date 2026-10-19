from __future__ import annotations

import base64

from mcp.server.mcpserver import MCPServer
from mcp.types import ImageContent

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

server = MCPServer("demo")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@server.tool()
def preview(name: str) -> list[ImageContent]:
    """Return a preview image."""
    return [ImageContent(type="image", data=base64.b64encode(PNG).decode(), mime_type="image/png")]


@server.tool()
def broken() -> str:
    """Always fails."""
    raise ValueError("the service is down")


if __name__ == "__main__":
    server.run()
