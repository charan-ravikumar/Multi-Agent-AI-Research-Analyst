"""
A real worker process with a scripted LLM and search, for the kill test:  python -m tests.fake_worker

Environment: REDIS_URL, QUEUE_NAMESPACE, FAKE_CALL_LOG (one line per LLM call: "<pid> <role> <aspect>"),
FAKE_HANG_ASPECT (fact extraction for that sub-question never returns).
"""
import os
import re
import time

import research_analyst.agents.searcher as searcher
from research_analyst import llm
from research_analyst.jobs.worker import build_worker, main  # noqa: F401
from tests.helpers import FakeLLM, make_search

_LOG = os.environ["FAKE_CALL_LOG"]
_HANG = os.environ.get("FAKE_HANG_ASPECT", "")


class LoggingLLM(FakeLLM):
    def __call__(self, prompt, **kwargs):
        role = self.role_of(prompt)
        found = re.search(r"aspect \d", prompt[-1]["content"])
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()}|{role}|{found.group(0) if found else ''}\n")
        if role == "reader" and _HANG and _HANG in prompt[-1]["content"]:
            time.sleep(3600)
        return super().__call__(prompt, **kwargs)


if __name__ == "__main__":
    llm.generate = LoggingLLM()
    searcher.search_web = make_search()

    async def _nothing(*args, **kwargs):
        return {}

    async def _no_results(*args, **kwargs):
        return []

    searcher.fetch_many, searcher.search_arxiv = _nothing, _no_results
    main()
