"""One-shot UV-script execution, local or via HF Jobs.

This is the thin "where does compute run?" abstraction for PEP-723 inline
scripts. It's deliberately *not* a :class:`~ocrscout.interfaces.runner.Runner`
— :class:`Runner` orchestrates long-lived daemonised stacks (LiteLLM
proxy + N vLLM serves + KV preflight + state.yaml). ``run_uv_script`` is
stateless: hand it a URL + args, choose ``local`` or ``hf``, get a result
back. No launch/down lifecycle, no proxy state.

Used by source admin actions (e.g. :class:`BhlSourceAdapter.Refresh
<ocrscout.sources.bhl.BhlSourceAdapter.Refresh>`) to dispatch classifier
scripts to either the user's own machine (``uv run --with X --with Y
<url> -- ...``) or HF-sponsored compute (``HfApi().run_job(command=...)``).
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ocrscout.errors import ScoutError

log = logging.getLogger(__name__)

JobRunner = Literal["local", "hf"]


@dataclass(frozen=True)
class JobResult:
    """One-shot UV-script execution outcome.

    For ``local``: ``exit_code`` and ``duration_seconds`` are always set;
    ``job_id`` is ``None``. For ``hf``: ``job_id`` is the HF Jobs job
    identifier; ``exit_code`` and ``duration_seconds`` reflect the
    log-streaming wait if ``stream_logs=True``, or are ``None`` if the
    caller chose to detach.
    """

    runner: JobRunner
    exit_code: int | None
    duration_seconds: float | None
    job_id: str | None


def run_uv_script(
    script_url: str,
    args: Sequence[str],
    *,
    runner: JobRunner = "local",
    with_packages: Sequence[str] = (),
    secrets: dict[str, str] | None = None,
    flavor: str = "l4x1",
    timeout: str = "1h",
    stream_logs: bool = True,
) -> JobResult:
    """Execute a PEP-723 inline UV script.

    Args:
        script_url: URL to a UV inline script (or local path).
        args: Arguments to pass after ``--`` to the script.
        runner: ``"local"`` runs via ``subprocess`` here; ``"hf"`` submits
            to the HuggingFace Jobs API.
        with_packages: Extra ``--with`` flags for ``uv run``. The classifier
            UV script needs e.g. ``vllm<0.12.0``.
        secrets: For ``hf``, passed as the job's env block (canonically
            ``{"HF_TOKEN": "..."}``); for ``local``, merged into the
            subprocess env.
        flavor: HF Jobs GPU flavor identifier (``hf`` only). Ignored for
            ``local`` — the user's own GPU is used.
        timeout: HF Jobs timeout, human-readable (``"1h"`` etc.). Ignored
            for ``local``.
        stream_logs: Block on log output. ``True`` for ``local`` (always
            blocks). For ``hf``, ``True`` tails the job's logs; ``False``
            returns immediately after submission.

    Raises:
        ScoutError: subprocess failure (local), missing huggingface_hub
            (hf), or HF Jobs API error (hf).
    """
    if runner == "local":
        return _run_local(script_url, args, with_packages, secrets)
    if runner == "hf":
        return _run_hf(
            script_url, args, with_packages, secrets, flavor, timeout, stream_logs
        )
    raise ScoutError(f"unknown runner {runner!r}; expected 'local' or 'hf'.")


def _run_local(
    script_url: str,
    args: Sequence[str],
    with_packages: Sequence[str],
    secrets: dict[str, str] | None,
) -> JobResult:
    cmd: list[str] = ["uv", "run"]
    for pkg in with_packages:
        cmd.extend(["--with", pkg])
    cmd.append(script_url)
    cmd.extend(args)

    env = os.environ.copy()
    if secrets:
        env.update(secrets)

    log.info(
        "[uv-script] local: %s",
        " ".join(shlex.quote(c) for c in cmd),
    )
    start = time.monotonic()
    result = subprocess.run(cmd, env=env, check=False)
    duration = time.monotonic() - start

    if result.returncode != 0:
        raise ScoutError(
            f"local uv script failed (exit {result.returncode}): "
            f"{shlex.join(cmd)}"
        )

    return JobResult(
        runner="local",
        exit_code=result.returncode,
        duration_seconds=duration,
        job_id=None,
    )


def _run_hf(
    script_url: str,
    args: Sequence[str],
    with_packages: Sequence[str],
    secrets: dict[str, str] | None,
    flavor: str,
    timeout: str,
    stream_logs: bool,
) -> JobResult:
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ScoutError(
            "huggingface_hub is required for `--runner hf`; install via "
            "`pip install huggingface_hub` (or `pip install ocrscout[hf]`)."
        ) from e

    api = HfApi()
    cmd_parts = ["uv", "run"]
    for pkg in with_packages:
        cmd_parts.extend(["--with", shlex.quote(pkg)])
    cmd_parts.append(shlex.quote(script_url))
    cmd_parts.extend(shlex.quote(a) for a in args)
    command = " ".join(cmd_parts)

    log.info("[uv-script] hf (flavor=%s, timeout=%s): %s", flavor, timeout, command)

    # HfApi.run_job is in active development; check for it defensively
    # and surface a clear upgrade path if the SDK is too old.
    run_job = getattr(api, "run_job", None)
    if run_job is None:
        raise ScoutError(
            "huggingface_hub.HfApi has no `run_job`; upgrade with "
            "`uv pip install -U huggingface_hub`."
        )

    start = time.monotonic()
    try:
        job = run_job(
            command=command,
            flavor=flavor,
            env=secrets or {},
            timeout=timeout,
        )
    except Exception as e:  # noqa: BLE001
        raise ScoutError(f"HF Jobs API run_job failed: {e}") from e

    job_id = (
        getattr(job, "id", None)
        or getattr(job, "job_id", None)
        or str(job)
    )
    log.info("[uv-script] HF Job submitted: id=%s", job_id)

    if not stream_logs:
        return JobResult(
            runner="hf",
            exit_code=None,
            duration_seconds=None,
            job_id=str(job_id),
        )

    stream = getattr(api, "stream_job_logs", None) or getattr(
        api, "get_job_logs", None
    )
    if stream is None:
        log.warning(
            "huggingface_hub too old to stream job logs; submitted job %s "
            "but cannot follow.", job_id,
        )
        return JobResult(
            runner="hf",
            exit_code=None,
            duration_seconds=time.monotonic() - start,
            job_id=str(job_id),
        )

    try:
        for chunk in stream(job_id):
            print(chunk, end="" if isinstance(chunk, str) else "\n")
    except Exception as e:  # noqa: BLE001
        log.warning("HF Job log streaming failed (%s); job %s still running", e, job_id)

    return JobResult(
        runner="hf",
        exit_code=None,
        duration_seconds=time.monotonic() - start,
        job_id=str(job_id),
    )
