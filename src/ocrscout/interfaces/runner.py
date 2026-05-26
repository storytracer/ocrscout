"""Runner ABC: orchestrates the compute backend for OCR workloads.

A Runner provisions the vLLM + LiteLLM stack (locally, in a remote pool, or
on HF-sponsored compute), dispatches OCR work to it, reports status, streams
logs, and tears it down. The same Runner interface scales from a single
workstation (``LocalRunner``) to a Kubernetes pool (``SkyPilotRunner``) to
HuggingFace Jobs (``HuggingFaceRunner``).

Concrete Runners are registered under the ``runners`` entry-point group.
Third-party packages add new runners (Slurm, AI Factory, …) by declaring
``ocrscout.runners`` entry points; no source-tree changes required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ocrscout.types import JobHandle, PipelineConfig, RunnerHandle, RunnerStatus


class Runner(ABC):
    """Manages compute for OCR workloads.

    Lifecycle: ``launch()`` → ``submit()`` → ``status()``/``logs()`` →
    ``down()``. Each method is independently idempotent: calling
    ``launch()`` when the stack matches the requested config is a no-op
    that returns the existing handle; calling ``down()`` when already-down
    is a no-op.

    The CLI is stateless across invocations — nothing about an active run
    is held in memory between commands. ``status()`` re-queries the
    runner's external state (PID files for local, scheduler API for
    remote) on every call, so the orchestrating process can crash and
    restart without losing track of in-flight work.
    """

    name: ClassVar[str]
    requires_local_gpu: ClassVar[bool] = False
    """Whether the orchestrating machine must itself have a GPU.

    ``True`` for ``LocalRunner`` (the daemon stack runs locally);
    ``False`` for ``SkyPilotRunner`` and ``HuggingFaceRunner`` (work
    ships to a remote worker; the orchestrator is GPU-free).
    """

    @abstractmethod
    def launch(
        self,
        *,
        models: list[str],
        gpu_type: str | None = None,
        workers: int = 1,
        **kwargs: object,
    ) -> RunnerHandle:
        """Provision infrastructure and start the core stack.

        Returns once everything is healthy: the LiteLLM proxy's
        ``/health/liveliness`` passes and every vLLM ``/v1/models`` is
        answering. If the runner is already up with a compatible config,
        returns the existing handle without spawning anything new.
        """

    @abstractmethod
    def submit(
        self,
        *,
        config: PipelineConfig,
        resume: bool = False,
        **kwargs: object,
    ) -> JobHandle:
        """Submit a pipeline run. Returns a ``JobHandle``. Non-blocking.

        ``config`` is the full ``PipelineConfig`` (source / reference /
        comparisons / models / export) — the same shape that
        ``ocrscout run`` produces and writes to ``pipeline.yaml`` for
        reproducibility. Runners are free to interpret the source
        adapter's own kwargs (``start_idx``, ``end_idx``, …) for work
        partitioning; everything else round-trips into the worker
        unchanged.

        For ``LocalRunner`` the work runs in a daemonised worker process;
        for ``SkyPilotRunner`` the SkyPilot controller takes over; for
        ``HuggingFaceRunner`` the HF Jobs API takes over. In every case
        ``submit`` returns immediately and the orchestrating CLI can
        exit without affecting in-flight work.

        Output is written incrementally to
        ``<output_dir>/data/train-NNNNN.parquet`` with a sibling
        ``progress.json`` checkpoint. Pass ``resume=True`` to skip
        already-completed page ids recorded there.
        """

    @abstractmethod
    def status(self) -> RunnerStatus:
        """Query current state of infrastructure and jobs.

        Cheap to call (suitable for polling): reads PID files, scheduler
        APIs, and the output Parquet glob; does not consult in-memory
        caches.
        """

    @abstractmethod
    def logs(self, job_id: str | None = None, *, follow: bool = True) -> None:
        """Stream logs.

        When ``job_id`` is provided, stream that submission's worker log.
        When ``None``, stream all daemon/worker logs the runner manages,
        prefixed with ``[<name>]`` so concurrent output stays readable.
        """

    @abstractmethod
    def down(self) -> None:
        """Tear down infrastructure.

        Idempotent: a no-op if already down. Does not delete output
        Parquet/progress files — only the compute is torn down.
        """
