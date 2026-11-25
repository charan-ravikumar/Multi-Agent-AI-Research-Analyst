"""
jobs/checkpoint.py — a LangGraph checkpointer on plain Redis.

Why not ``langgraph-checkpoint-redis``: it requires the RedisJSON and query-engine modules, which a plain
``redis`` server (including Homebrew's build and the ``redis:7`` image) does not have.  This saver uses only core
commands, so it runs on any Redis and on fakeredis in offline tests.  It is checked against LangGraph's official
conformance suite (tests/integration/test_redis_checkpoint.py).

Layout, per thread (``{thread}`` is a hash tag so one thread's keys share a cluster slot):
  <p>:ckpt:{thread}:idx:<ns>        sorted set of checkpoint ids (all score 0 -> ordered by id)
  <p>:ckpt:{thread}:cp:<ns>:<id>    hash: checkpoint, metadata, parent
  <p>:ckpt:{thread}:w:<ns>:<id>     hash: pending writes of that checkpoint, one field per (task, index)
  <p>:ckpt:{thread}:blobs           hash: channel values, one field per (ns, channel, version)
  <p>:ckpt:{thread}:ns / :keys      bookkeeping (namespaces; every key of the thread, for deletion)

Every write is one MULTI/EXEC transaction and refreshes a TTL, so abandoned runs expire on their own.
Values are msgpack via LangGraph's serializer with an explicit allow-list of this project's models; pickle is off.
"""
from __future__ import annotations

import asyncio
import builtins
import json
import random
from collections.abc import AsyncIterator, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, cast

import redis
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from research_analyst.models.citation import Citation
from research_analyst.models.fact import Fact
from research_analyst.models.research_plan import ResearchPlan
from research_analyst.models.search_result import SearchResult

_STATE_TYPES = (Citation, Fact, ResearchPlan, SearchResult)


def _serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=[(t.__module__, t.__name__) for t in _STATE_TYPES])


def _pack(header: list[Any], data: bytes) -> bytes:
    """JSON header + newline + payload (JSON never contains a raw newline, so the first one splits them)."""
    return json.dumps(header).encode() + b"\n" + data


def _unpack(raw: bytes) -> tuple[list[Any], bytes]:
    header, data = raw.split(b"\n", 1)
    return json.loads(header), data


class RedisCheckpointSaver(BaseCheckpointSaver[str]):
    """Durable checkpoints on core Redis.  ``client`` must return bytes (``decode_responses=False``)."""

    def __init__(self, client: redis.Redis, *, prefix: str = "ra", ttl_seconds: int = 7 * 24 * 3600) -> None:
        super().__init__(serde=_serializer())
        self.client = client
        self.prefix = prefix
        self.ttl = ttl_seconds
        # Own threads: the event loop's default executor can be full of LLM calls sleeping on rate limits, and a
        # checkpoint write must not queue behind them.
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="checkpoint")

    # ── keys ─────────────────────────────────────────────────────────────────

    def _base(self, thread_id: str) -> str:
        return f"{self.prefix}:ckpt:{{{thread_id}}}"

    def _threads_key(self) -> str:
        return f"{self.prefix}:ckpt:threads"

    def _touch(self, pipe: Any, thread_id: str, *keys: str) -> None:
        """Register keys for later deletion and (re)set their TTL, inside the caller's transaction."""
        registry = f"{self._base(thread_id)}:keys"
        pipe.sadd(registry, *keys)
        for key in (*keys, registry):
            pipe.expire(key, self.ttl)

    # ── reads ────────────────────────────────────────────────────────────────

    def _members(self, key: str) -> builtins.list[str]:
        return sorted(m.decode() for m in cast("set[bytes]", self.client.smembers(key)))

    def _ids(self, index_key: str, start: int, end: int) -> builtins.list[str]:
        """Checkpoint ids of an index, newest first."""
        return [m.decode() for m in cast("builtins.list[bytes]", self.client.zrevrange(index_key, start, end))]

    def _tuple(self, thread_id: str, ns: str, checkpoint_id: str) -> CheckpointTuple | None:
        base = self._base(thread_id)
        pipe = self.client.pipeline()
        pipe.hgetall(f"{base}:cp:{ns}:{checkpoint_id}")
        pipe.hgetall(f"{base}:w:{ns}:{checkpoint_id}")
        saved, raw_writes = pipe.execute()
        if not saved:
            return None
        checkpoint: Checkpoint = self._load(saved[b"checkpoint"])
        versions = checkpoint["channel_versions"]
        values: dict[str, Any] = {}
        if versions:
            channels = list(versions)
            blobs = self.client.hmget(f"{base}:blobs", [f"{ns}\x00{c}\x00{versions[c]}" for c in channels])
            for channel, raw in zip(channels, blobs, strict=True):
                if raw is None:
                    continue
                (kind,), data = _unpack(cast(bytes, raw))
                if kind != "empty":
                    values[channel] = self.serde.loads_typed((kind, data))
        writes = []
        for raw in raw_writes.values():
            (task_id, channel, kind, task_path, idx), data = _unpack(raw)
            writes.append(((task_path, task_id, idx), (task_id, channel, self.serde.loads_typed((kind, data)))))
        writes.sort(key=lambda w: w[0])
        parent = saved.get(b"parent", b"").decode()
        return CheckpointTuple(
            config=self._config(thread_id, ns, checkpoint_id),
            checkpoint={**checkpoint, "channel_values": values},
            metadata=self._load(saved[b"metadata"]),
            pending_writes=[w for _, w in writes],
            parent_config=self._config(thread_id, ns, parent) if parent else None,
        )

    def _dump(self, value: Any) -> bytes:
        kind, data = self.serde.dumps_typed(value)
        return _pack([kind], data)

    def _load(self, raw: bytes) -> Any:
        (kind,), data = _unpack(raw)
        return self.serde.loads_typed((kind, data))

    @staticmethod
    def _config(thread_id: str, ns: str, checkpoint_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": checkpoint_id}}

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        if not checkpoint_id:
            latest = self._ids(f"{self._base(thread_id)}:idx:{ns}", 0, 0)
            if not latest:
                return None
            checkpoint_id = latest[0]
        return self._tuple(thread_id, ns, checkpoint_id)

    def list(
        self, config: RunnableConfig | None, *, filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None, limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        threads = [config["configurable"]["thread_id"]] if config else self._members(self._threads_key())
        wanted_ns = config["configurable"].get("checkpoint_ns") if config else None
        wanted_id = get_checkpoint_id(config) if config else None
        before_id = get_checkpoint_id(before) if before else None
        remaining = limit
        for thread_id in threads:
            base = self._base(thread_id)
            namespaces = [wanted_ns] if wanted_ns is not None else self._members(f"{base}:ns")
            for ns in namespaces:
                for checkpoint_id in self._ids(f"{base}:idx:{ns}", 0, -1):
                    if (wanted_id and checkpoint_id != wanted_id) or (before_id and checkpoint_id >= before_id):
                        continue
                    if remaining is not None and remaining <= 0:
                        return
                    found = self._tuple(thread_id, ns, checkpoint_id)
                    if found is None or (filter and any(found.metadata.get(k) != v for k, v in filter.items())):
                        continue
                    if remaining is not None:
                        remaining -= 1
                    yield found

    # ── writes ───────────────────────────────────────────────────────────────

    def put(
        self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        body: dict[str, Any] = dict(checkpoint)
        values: dict[str, Any] = body.pop("channel_values")
        base = self._base(thread_id)
        cp_key, idx_key, blob_key, ns_key = (
            f"{base}:cp:{ns}:{checkpoint['id']}", f"{base}:idx:{ns}", f"{base}:blobs", f"{base}:ns")
        blobs = {
            f"{ns}\x00{channel}\x00{version}": self._dump(values[channel]) if channel in values else _pack(["empty"], b"")
            for channel, version in new_versions.items()
        }
        pipe = self.client.pipeline(transaction=True)
        if blobs:
            pipe.hset(blob_key, mapping=cast("dict[Any, Any]", blobs))
        pipe.hset(cp_key, mapping={
            "checkpoint": self._dump(body),
            "metadata": self._dump(get_checkpoint_metadata(config, metadata)),
            "parent": config["configurable"].get("checkpoint_id") or "",
        })
        pipe.zadd(idx_key, {checkpoint["id"]: 0})
        pipe.sadd(ns_key, ns)
        pipe.sadd(self._threads_key(), thread_id)
        pipe.expire(self._threads_key(), self.ttl)
        self._touch(pipe, thread_id, cp_key, idx_key, blob_key, ns_key)
        pipe.execute()
        return self._config(thread_id, ns, checkpoint["id"])

    def put_writes(
        self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        key = f"{self._base(thread_id)}:w:{ns}:{config['configurable']['checkpoint_id']}"
        if not writes:
            return
        pipe = self.client.pipeline(transaction=True)
        for position, (channel, value) in enumerate(writes):
            idx = WRITES_IDX_MAP.get(channel, position)
            kind, data = self.serde.dumps_typed(value)
            field = f"{task_id}\x00{idx}"
            packed = _pack([task_id, channel, kind, task_path, idx], data)
            # A repeated regular write is ignored (first one wins); special writes (negative index) are replaced.
            if idx >= 0:
                pipe.hsetnx(key, field, packed)
            else:
                pipe.hset(key, field, packed)
        self._touch(pipe, thread_id, key)
        pipe.execute()

    def delete_thread(self, thread_id: str) -> None:
        registry = f"{self._base(thread_id)}:keys"
        keys = self.client.smembers(registry)
        pipe = self.client.pipeline(transaction=True)
        pipe.delete(registry, *keys)
        pipe.srem(self._threads_key(), thread_id)
        pipe.execute()

    def get_next_version(self, current: str | None, channel: None) -> str:
        previous = 0 if current is None else int(str(current).split(".")[0])
        return f"{previous + 1:032}.{random.random():016}"  # noqa: S311 — uniqueness, not security

    # ── async API ────────────────────────────────────────────────────────────

    async def _off_loop(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._pool, partial(fn, *args, **kwargs))

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await self._off_loop(self.get_tuple, config)

    async def alist(
        self, config: RunnableConfig | None, *, filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None, limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        found = await self._off_loop(lambda: [*self.list(config, filter=filter, before=before, limit=limit)])
        for item in found:
            yield item

    async def aput(
        self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await self._off_loop(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = "",
    ) -> None:
        await self._off_loop(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await self._off_loop(self.delete_thread, thread_id)
