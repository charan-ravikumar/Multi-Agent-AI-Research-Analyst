"""Background research thread."""
from __future__ import annotations

from typing import Any

import asyncio


def research_thread(
    app_ref: Any,
    cfg: dict,
    resume_value: list[str],
    result_holder: dict,
) -> None:
    """
    Run the graph to completion inside a dedicated asyncio event loop.

    Stores outcome into *result_holder*:
      • success  → {"result": <state dict>, "done": True}
      • failure  → {"error": <str>,         "done": True}
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        from langgraph.types import Command  # noqa: PLC0415
        from research_analyst.config import settings  # noqa: PLC0415
        from research_analyst.observability import tracing  # noqa: PLC0415

        async def _run() -> Any:
            sid = cfg.get("configurable", {}).get("session_id", "")
            with tracing.run_span("research", sid):
                return await asyncio.wait_for(
                    app_ref.ainvoke(Command(resume=resume_value), config=cfg),
                    timeout=settings.run_timeout_seconds,
                )

        result = loop.run_until_complete(_run())
        result_holder["result"] = result
    except asyncio.TimeoutError:
        result_holder["error"] = "Run exceeded the time limit and was cancelled."
    except Exception as exc:
        result_holder["error"] = str(exc)
    finally:
        result_holder["done"] = True
        loop.close()
