"""
Durable execution: research runs as jobs on Redis, executed by worker processes.

  checkpoint.py   LangGraph checkpointer on core Redis (what has been computed)
  store.py        run records + the job stream (who should compute what, and its state)
  execution.py    start or resume one stage of a run from its checkpoint
  worker.py       the worker process: claim, heartbeat, cancel, hand over, dead-letter

Used when ``EXECUTION_MODE=queue``.  The default, ``inline``, keeps everything inside the UI process.
"""
