"""Background research thread with cooperative cancellation."""
from __future__ import annotations

import asyncio
from typing import Any


def research_thread(app_ref: Any, cfg: dict, resume_value: list[str], result_holder: dict) -> None:
    """
    Run the graph to completion inside a dedicated asyncio event loop.

    Stores the outcome into *result_holder*:
      success    -> {"result": <state dict>, "done": True}
      failure    -> {"error": <str>,         "done": True}
      cancelled  -> {"cancelled": True,      "done": True}      (see ``request_cancel``)
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result_holder["loop"] = loop
    try:
        from langgraph.types import Command  # noqa: PLC0415

        from research_analyst.config import settings  # noqa: PLC0415
        from research_analyst.observability import tracing  # noqa: PLC0415

        async def _run() -> Any:
            sid = cfg.get("configurable", {}).get("session_id", "")
            with tracing.run_span("research", sid, tags=cfg.get("tags")):
                result = await asyncio.wait_for(
                    app_ref.ainvoke(Command(resume=resume_value), config=cfg),
                    timeout=settings.run_timeout_seconds,
                )
                tracing.record_scores(result)
                return result

        task = loop.create_task(_run())
        result_holder["task"] = task
        if result_holder.get("cancel_requested"):      # the user clicked Stop before the task existed
            task.cancel()
        result_holder["result"] = loop.run_until_complete(task)
    except asyncio.CancelledError:
        result_holder["cancelled"] = True
    except asyncio.TimeoutError:
        result_holder["error"] = "Run exceeded the time limit and was cancelled."
    except Exception as exc:  # noqa: BLE001
        result_holder["error"] = str(exc)
    finally:
        result_holder["done"] = True
        loop.close()


def request_cancel(result_holder: dict | None) -> bool:
    """Ask a running research thread to stop.  Safe to call from the Streamlit thread at any time.

    Cancels the awaiting coroutines; an LLM request already in flight in a worker thread cannot be
    interrupted and finishes in the background (its result is discarded and its tokens are still spent).
    Returns True if a cancel was issued."""
    if not result_holder or result_holder.get("done"):
        return False
    result_holder["cancel_requested"] = True
    loop, task = result_holder.get("loop"), result_holder.get("task")
    if loop is not None and task is not None and not loop.is_closed():
        loop.call_soon_threadsafe(task.cancel)
    return True
