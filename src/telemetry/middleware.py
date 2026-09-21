"""Small ASGI middleware for correlation; it does not inspect request bodies."""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from .logging import bind, clear_context
from .request_id import request_context


class RequestIdMiddleware:
    def __init__(self, app: Callable[..., Awaitable[Any]]) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Awaitable[Any]], send: Callable[..., Awaitable[Any]]) -> Any:
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        headers = {key.decode("latin1").lower(): value.decode("latin1") for key, value in scope.get("headers", [])}
        supplied = headers.get("x-request-id")
        with request_context(supplied) as request_id:
            bind(request_id=request_id)
            async def send_with_id(message: dict[str, Any]) -> Any:
                if message.get("type") == "http.response.start":
                    raw_headers = list(message.get("headers", []))
                    raw_headers.append((b"x-request-id", request_id.encode("ascii")))
                    message = {**message, "headers": raw_headers}
                return await send(message)
            try:
                return await self.app(scope, receive, send_with_id)
            finally:
                clear_context()
