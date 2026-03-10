"""Web server — FastAPI application for the Sleeping LLM web UI."""

import asyncio
import json
import time

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
        """Stream chat response tokens via SSE."""
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

    @app.get("/api/buffer")
    async def buffer_status():
        """Return current fact buffer status."""
        if not _orchestrator.fact_buffer:
            return {"enabled": False}
        return _orchestrator.fact_buffer.to_dict()

    @app.post("/api/consolidate")
    async def consolidate():
        """Manually trigger consolidation of the fact buffer."""
        if not _orchestrator.fact_buffer:
            return {"status": "disabled", "message": "Consolidation moments not enabled"}
        if _orchestrator.fact_buffer.is_empty:
            return {"status": "empty", "message": "Buffer is empty"}
        async with _model_lock:
            count = await asyncio.to_thread(
                _orchestrator.fact_buffer.consolidate, "manual"
            )
        return {"status": "ok", "facts_consolidated": count}

    @app.get("/api/facts")
    async def facts():
        """Return all facts split into buffered, consolidated, and graduated."""
        result = {"buffered": [], "consolidated": [], "graduated": []}
        if _orchestrator.fact_buffer:
            for bf in _orchestrator.fact_buffer._buffer:
                result["buffered"].append({
                    "question": bf.qa.question,
                    "answer": bf.qa.answer,
                    "value": bf.qa.value,
                    "age_seconds": round(time.time() - bf.buffered_at, 1),
                    "source_turn": bf.source_turn,
                })
        for entry in _orchestrator.fact_ledger.get_active_facts():
            qa = entry.get("qa", {})
            result["consolidated"].append({
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "value": qa.get("value", ""),
                "fact_id": entry["fact_id"],
                "stage": entry.get("stage", 0),
                "recall_rate": round(entry.get("recall_rate", 0.0), 2),
                "train_count": entry.get("train_count", 0),
            })
        for entry in _orchestrator.fact_ledger.get_graduated_facts():
            qa = entry.get("qa", {})
            result["graduated"].append({
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "value": qa.get("value", ""),
                "fact_id": entry["fact_id"],
                "train_count": entry.get("train_count", 0),
            })
        return result

    @app.post("/api/microsleep")
    async def microsleep():
        """Manually trigger a micro-sleep pass on highest-priority facts."""
        if not _orchestrator.micro_sleep:
            return {"status": "disabled", "message": "Micro-sleep not enabled. Set micro_sleep.enabled: true in config."}
        if _orchestrator.micro_sleep.is_running:
            return {"status": "running", "message": "Micro-sleep already in progress."}
        if _orchestrator.background_sleep.is_sleeping:
            return {"status": "blocked", "message": "Can't micro-sleep during full sleep/nap."}

        started = _orchestrator.micro_sleep.maybe_trigger(
            1.0, background_sleep_manager=_orchestrator.background_sleep,
        )
        if started:
            return {"status": "ok", "message": "Micro-sleep started in background."}
        return {"status": "skipped", "message": "No eligible facts (all recently trained or cooldown active)."}

    @app.post("/api/reset/ledger")
    async def reset_ledger():
        """Clear all active (non-graduated) facts from the ledger.

        Graduated facts are preserved — this tests if LoRA recall works
        without the system prompt safety net.
        """
        try:
            ledger = _orchestrator.fact_ledger
            active = ledger.get_active_facts()
            for entry in active:
                ledger.mark_pruned(entry["fact_id"])
            return {"status": "ok", "message": f"Cleared {len(active)} consolidated facts. Graduated facts preserved."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.post("/api/reset/weights")
    async def reset_weights():
        """Reset model to base weights."""
        try:
            async with _model_lock:
                result = await asyncio.to_thread(_orchestrator.reset_weights)
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": f"Weight reset failed: {e}"}

    @app.post("/api/reset/factory")
    async def factory_reset():
        """Full factory reset — clears everything."""
        try:
            async with _model_lock:
                result = await asyncio.to_thread(_orchestrator.factory_reset)
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": f"Factory reset failed: {e}"}

    return app
