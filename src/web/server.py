"""Web server — FastAPI application for the Sleeping LLM web UI."""

import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from pathlib import Path

from src.config import Config
from src.orchestrator import Orchestrator

# Shared state
_orchestrator = None
_model_lock = asyncio.Lock()


class ChatRequest(BaseModel):
    message: str


def create_app(config: Config, disable_memit: bool = False) -> FastAPI:
    """Create and configure the FastAPI application."""
    global _orchestrator

    app = FastAPI(title="Sleeping LLM")

    # Initialize orchestrator
    _orchestrator = Orchestrator(config, disable_memit=disable_memit)

    static_dir = Path(__file__).parent / "static"

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/api/chat/stream")
    async def chat_stream(message: str):
        """Stream chat response tokens via SSE.

        Event sequence:
          token* → done → [sleep_start → sleep_progress* → sleep_done] → complete
          OR: token* → done → [nap_start → nap_progress* → nap_done] → complete
        """
        async def event_generator():
            auto_sleep = False
            auto_nap = False
            async with _model_lock:
                gen = _orchestrator.process_message_stream(message)
                try:
                    while True:
                        token = await asyncio.to_thread(next, gen, None)
                        if token is None:
                            break
                        if isinstance(token, dict):
                            if token.get("__auto_sleep__"):
                                auto_sleep = True
                                break
                            if token.get("__auto_nap__"):
                                auto_nap = True
                                break
                        yield {"event": "token", "data": json.dumps({"token": token})}
                except StopIteration:
                    pass

                yield {"event": "done", "data": ""}

                if auto_sleep:
                    yield {"event": "sleep_start", "data": ""}
                    sleep_gen = _orchestrator.trigger_sleep_web()
                    try:
                        while True:
                            event = await asyncio.to_thread(next, sleep_gen, None)
                            if event is None:
                                break
                            yield {"event": "sleep_progress", "data": json.dumps(event)}
                    except StopIteration:
                        pass
                    yield {"event": "sleep_done", "data": ""}

                elif auto_nap:
                    yield {"event": "nap_start", "data": ""}
                    nap_gen = _orchestrator.trigger_nap_web()
                    try:
                        while True:
                            event = await asyncio.to_thread(next, nap_gen, None)
                            if event is None:
                                break
                            yield {"event": "nap_progress", "data": json.dumps(event)}
                    except StopIteration:
                        pass
                    yield {"event": "nap_done", "data": ""}

            yield {"event": "complete", "data": ""}

        return EventSourceResponse(event_generator())

    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        """Non-streaming chat endpoint."""
        async with _model_lock:
            response = await asyncio.to_thread(
                _orchestrator.process_message, request.message
            )
        return {"response": response}

    @app.get("/api/sleep/stream")
    async def sleep_stream():
        """Trigger sleep and stream progress via SSE."""
        async def event_generator():
            async with _model_lock:
                gen = _orchestrator.trigger_sleep_web()
                try:
                    while True:
                        event = await asyncio.to_thread(next, gen, None)
                        if event is None:
                            break
                        yield {"event": "progress", "data": json.dumps(event)}
                except StopIteration:
                    pass
            yield {"event": "done", "data": ""}

        return EventSourceResponse(event_generator())

    @app.get("/api/nap/stream")
    async def nap_stream():
        """Trigger nap and stream progress via SSE."""
        async def event_generator():
            async with _model_lock:
                gen = _orchestrator.trigger_nap_web()
                try:
                    while True:
                        event = await asyncio.to_thread(next, gen, None)
                        if event is None:
                            break
                        yield {"event": "progress", "data": json.dumps(event)}
                except StopIteration:
                    pass
            yield {"event": "done", "data": ""}

        return EventSourceResponse(event_generator())

    @app.get("/api/health")
    async def health():
        """Return health monitor data."""
        return _orchestrator.health_monitor.to_dict()

    @app.get("/api/status")
    async def status():
        """Return current system status."""
        return _orchestrator.get_status()

    @app.get("/api/history")
    async def history():
        """Return current session messages."""
        return {"messages": _orchestrator.get_current_messages()}

    @app.post("/api/reset/weights")
    async def reset_weights():
        """Reset model to base weights."""
        async with _model_lock:
            result = await asyncio.to_thread(_orchestrator.reset_weights)
        return result

    @app.post("/api/reset/factory")
    async def factory_reset():
        """Full factory reset — clears everything."""
        async with _model_lock:
            result = await asyncio.to_thread(_orchestrator.factory_reset)
        return result

    return app
