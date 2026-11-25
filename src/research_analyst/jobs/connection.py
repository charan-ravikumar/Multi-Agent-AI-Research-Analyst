"""Redis clients for the job system, built from ``settings.redis_url``."""
from __future__ import annotations

import redis

from research_analyst.config import settings

# Longer than the worker's blocking read, short enough that a hung server is noticed within one lease.
_SOCKET_TIMEOUT_S = 10.0


def new_client(*, decode: bool) -> redis.Redis:
    """``decode=True`` for run records and the stream (text); False for checkpoints (binary)."""
    return redis.Redis.from_url(
        settings.redis_url, decode_responses=decode, socket_connect_timeout=5,
        socket_timeout=_SOCKET_TIMEOUT_S, health_check_interval=30,
    )
