"""
Redis for the job-system tests: always fakeredis; additionally a REAL server when one is available
(``TEST_REDIS_URL``, or a ``redis-server`` binary, which is then started on a free port for the session).
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import fakeredis
import redis


@dataclass
class Backend:
    name: str
    text: redis.Redis          # decodes responses (run records, stream)
    binary: redis.Redis        # raw bytes (checkpoints)
    prefix: str                # unique per test, so tests on a shared server cannot see each other
    url: str = ""


def fake_backend() -> Backend:
    server = fakeredis.FakeServer()
    return Backend("fake", fakeredis.FakeRedis(server=server, decode_responses=True),
                   fakeredis.FakeRedis(server=server), f"t{uuid.uuid4().hex[:8]}")


def real_backend(url: str) -> Backend:
    return Backend("real", redis.Redis.from_url(url, decode_responses=True), redis.Redis.from_url(url),
                   f"t{uuid.uuid4().hex[:8]}", url)


def cleanup(backend: Backend) -> None:
    keys = list(backend.binary.scan_iter(match=f"{backend.prefix}:*", count=500))
    if keys:
        backend.binary.delete(*keys)


def server_url() -> Iterator[str | None]:
    """Yields a URL of a real Redis, or None if there is no way to get one."""
    if url := os.environ.get("TEST_REDIS_URL"):
        yield url
        return
    binary = shutil.which("redis-server")
    if not binary:
        yield None
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([binary, "--port", str(port), "--bind", "127.0.0.1", "--save", "", "--appendonly", "no"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"redis://127.0.0.1:{port}/0"
    try:
        client = redis.Redis.from_url(url)
        for _ in range(100):
            try:
                client.ping()
                break
            except redis.ConnectionError:
                time.sleep(0.05)
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)
