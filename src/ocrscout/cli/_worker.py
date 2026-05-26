"""Hidden ``ocrscout _worker`` subcommand: drive a previously-submitted job.

Spawned by ``LocalRunner.submit`` via :func:`daemonize_subprocess`; never
invoked directly by users. Reads ``~/.ocrscout/jobs/<job_id>/config.yaml``,
resolves the active runner's proxy URL into ``OCRSCOUT_VLLM_URL``, and
runs the same dispatch loop ``ocrscout run`` uses.

Failures are logged to the job's log file (set up by the daemonising
parent) and the worker exits with a non-zero status. Status surfaces
through ``ocrscout status`` reading the job's ``state`` field.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import typer
import yaml

from ocrscout import state as state_mod
from ocrscout.cli import app
from ocrscout.log import setup_logging
from ocrscout.types import PipelineConfig

log = logging.getLogger(__name__)


@app.command("_worker", hidden=True)
def _worker(
    job_id: str = typer.Option(..., "--job", help="Job id under ~/.ocrscout/jobs/"),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Internal: execute a previously-submitted pipeline."""
    # Daemonised; setup_logging still configures the root logger which
    # then writes to the redirected stdout (the job log file).
    setup_logging(verbosity=verbose, quiet=quiet)

    job_dir = state_mod.jobs_dir() / job_id
    config_path = job_dir / "config.yaml"
    if not config_path.is_file():
        log.error("Worker: config file missing at %s", config_path)
        raise typer.Exit(code=1)

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        log.error("Worker: cannot read config %s: %s", config_path, e)
        raise typer.Exit(code=1) from e

    try:
        cfg = PipelineConfig.model_validate(raw)
    except Exception as e:  # noqa: BLE001
        log.error("Worker: invalid pipeline config in %s: %s", config_path, e)
        raise typer.Exit(code=1) from e

    # Inherit the active runner's proxy URL so backends can find it.
    state = state_mod.read_state()
    if state is not None and state.proxy_url:
        os.environ["OCRSCOUT_VLLM_URL"] = state.proxy_url

    _write_job_state(job_dir, status="running")

    # Local import avoids a circular dependency at module import time:
    # cli.__init__ pulls every cli.* subcommand including this one, and
    # cli.run pulls LocalRunner which transitively pulls this worker.
    from ocrscout.cli.run import _execute

    parallel_models = 1
    try:
        _execute(cfg, parallel_models=parallel_models)
    except SystemExit:
        _write_job_state(job_dir, status="failed")
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("Worker: pipeline execution failed: %s", e)
        _write_job_state(job_dir, status="failed")
        sys.exit(1)
    _write_job_state(job_dir, status="done")


def _write_job_state(job_dir: Path, *, status: str) -> None:
    """Persist the worker's progress to ``<job_dir>/state.yaml``."""
    from datetime import datetime, timezone

    job_state_path = job_dir / "state.yaml"
    payload = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        job_state_path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
    except OSError as e:
        log.warning("Worker: could not write job state: %s", e)
