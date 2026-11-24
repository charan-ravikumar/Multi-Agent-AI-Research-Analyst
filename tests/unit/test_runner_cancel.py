import asyncio
import threading
import time

from research_analyst.ui.runner import request_cancel, research_thread


class SlowApp:
    """Stands in for the compiled graph: ainvoke runs 'forever' unless cancelled."""

    def __init__(self):
        self.started = threading.Event()
        self.cancelled = False

    async def ainvoke(self, *_a, **_k):
        self.started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return {"final_report": "never"}


class QuickApp:
    async def ainvoke(self, *_a, **_k):
        return {"final_report": "ok"}


def _start(app, holder):
    t = threading.Thread(target=research_thread, args=(app, {"configurable": {"session_id": "s"}}, ["q?"], holder), daemon=True)
    t.start()
    return t


def test_cancel_stops_a_running_graph_quickly():
    app, holder = SlowApp(), {"done": False}
    t = _start(app, holder)
    assert app.started.wait(5)
    t0 = time.monotonic()
    assert request_cancel(holder) is True
    t.join(5)
    assert not t.is_alive() and time.monotonic() - t0 < 3
    assert holder["cancelled"] is True and holder["done"] is True and "result" not in holder and app.cancelled


def test_cancel_before_the_task_exists_is_not_lost():
    holder = {"done": False, "cancel_requested": True}                  # user clicked Stop immediately
    t = _start(SlowApp(), holder)
    t.join(5)
    assert not t.is_alive() and holder["cancelled"] is True


def test_normal_completion_is_untouched_and_late_cancel_is_a_noop():
    holder = {"done": False}
    t = _start(QuickApp(), holder)
    t.join(5)
    assert holder["result"] == {"final_report": "ok"} and "cancelled" not in holder
    assert request_cancel(holder) is False                              # already finished
    assert request_cancel(None) is False


def test_timeout_is_reported_as_an_error_not_a_cancel(monkeypatch):
    from research_analyst.config import settings
    monkeypatch.setattr(settings, "run_timeout_seconds", 0.2)
    holder = {"done": False}
    _start(SlowApp(), holder).join(5)
    assert "time limit" in holder["error"] and "cancelled" not in holder
