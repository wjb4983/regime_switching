"""Backend-neutral utilities for reliable local and parallel execution."""

from .checkpoint import CheckpointManager
from .data import MemmapArray
from .execution import (
    Executor,
    LocalExecutor,
    RetryableTask,
    RetryPolicy,
    SpawnProcessExecutor,
    chunked,
    run_batch,
    submit_batch,
)
from .observability import (
    Progress,
    ResourceUsage,
    log_resource_usage,
    report_progress,
    resource_usage,
)
from .system import (
    GPUInfo,
    InterruptionError,
    InterruptionHandler,
    detect_gpu,
    gpu_available,
    graceful_interrupts,
    initialize_seed,
)

__all__ = [
    "CheckpointManager",
    "Executor",
    "GPUInfo",
    "InterruptionError",
    "InterruptionHandler",
    "LocalExecutor",
    "MemmapArray",
    "Progress",
    "ResourceUsage",
    "RetryPolicy",
    "RetryableTask",
    "SpawnProcessExecutor",
    "chunked",
    "detect_gpu",
    "gpu_available",
    "graceful_interrupts",
    "initialize_seed",
    "log_resource_usage",
    "report_progress",
    "resource_usage",
    "run_batch",
    "submit_batch",
]
