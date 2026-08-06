"""Local, spawn-safe task execution primitives.

Worker callables must be defined at module scope.  This constraint keeps the same
task interface usable by a future distributed executor.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Executor(Protocol):
    """Minimal backend contract shared by local and future distributed executors."""

    def map(
        self, function: Callable[[InputT], OutputT], items: Iterable[InputT]
    ) -> Iterable[OutputT]:
        """Apply ``function`` to every item, preserving input order."""


@dataclass(frozen=True)
class RetryPolicy:
    """Retry configuration; delays grow exponentially and are capped."""

    attempts: int = 3
    initial_delay: float = 0.1
    multiplier: float = 2.0
    max_delay: float = 30.0
    exceptions: tuple[type[Exception], ...] = (Exception,)

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be at least one")
        if self.initial_delay < 0 or self.multiplier < 0 or self.max_delay < 0:
            raise ValueError("retry delays cannot be negative")


@dataclass(frozen=True)
class RetryableTask(Generic[InputT, OutputT]):
    """Pickle-friendly retry wrapper for a module-level callable."""

    function: Callable[[InputT], OutputT]
    policy: RetryPolicy = RetryPolicy()

    def __call__(self, item: InputT) -> OutputT:
        delay = self.policy.initial_delay
        for attempt in range(1, self.policy.attempts + 1):
            try:
                return self.function(item)
            except self.policy.exceptions:
                if attempt == self.policy.attempts:
                    raise
                time.sleep(min(delay, self.policy.max_delay))
                delay *= self.policy.multiplier
        raise RuntimeError("unreachable")


class LocalExecutor:
    """Sequential executor implementing the backend-neutral contract."""

    def map(
        self, function: Callable[[InputT], OutputT], items: Iterable[InputT]
    ) -> Iterable[OutputT]:
        return map(function, items)


@dataclass(frozen=True)
class SpawnProcessExecutor:
    """Process executor that always uses the Windows-compatible ``spawn`` context."""

    max_workers: int | None = None
    chunksize: int = 1

    def map(
        self, function: Callable[[InputT], OutputT], items: Iterable[InputT]
    ) -> Iterable[OutputT]:
        # Materialize before leaving the context: ProcessPoolExecutor.map is lazy.
        with ProcessPoolExecutor(
            max_workers=self.max_workers, mp_context=mp.get_context("spawn")
        ) as pool:
            return list(pool.map(function, items, chunksize=self.chunksize))


def run_batch(
    function: Callable[[InputT], OutputT],
    items: Iterable[InputT],
    *,
    executor: Executor | None = None,
) -> list[OutputT]:
    """Execute a batch through an interchangeable execution backend."""
    return list((executor or LocalExecutor()).map(function, items))


def submit_batch(
    pool: ProcessPoolExecutor,
    function: Callable[[InputT], OutputT],
    items: Sequence[InputT],
) -> list[Future[OutputT]]:
    """Submit a batch without closures, suitable for spawn-based pools."""
    return [pool.submit(function, item) for item in items]


def chunked(items: Iterable[InputT], size: int) -> Iterator[list[InputT]]:
    """Lazily split an iterable into bounded lists without loading it all at once."""
    if size < 1:
        raise ValueError("size must be at least one")
    batch: list[InputT] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


__all__ = [
    "Executor",
    "LocalExecutor",
    "RetryPolicy",
    "RetryableTask",
    "SpawnProcessExecutor",
    "chunked",
    "run_batch",
    "submit_batch",
]
