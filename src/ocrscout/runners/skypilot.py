"""SkyPilotRunner: remote GPU orchestration via SkyPilot on Kubernetes.

For benchmarking at scale and production re-OCR on OVHcloud Managed K8s.
Each ``launch`` provisions a SkyPilot pool (``sky jobs pool apply``);
each ``submit`` enqueues one managed job against that pool
(``sky jobs launch --pool``) so SkyPilot's controller handles failure
recovery, auto-restart, and resource cleanup. The orchestrating
ocrscout CLI is fully stateless across invocations — it can crash and
restart without losing track of in-flight work because state lives in
the SkyPilot controller's own database.

SkyPilot is lazy-imported inside method bodies so ``import
ocrscout.runners.skypilot`` succeeds on a workstation that hasn't
installed the ``[skypilot]`` extra; only ``launch``/``submit`` etc.
need the SDK or CLI to be present.

Pool name is derived from the model set + a short suffix so subsequent
``submit`` calls (and ``ocrscout status`` / ``down``) find the right
pool through ``~/.ocrscout/state.yaml`` rather than needing the user
to keep track of names.

The shipped pool/job YAML setup script installs ``ocrscout[vllm]`` via
``uv`` from PyPI by default. Override the install line via
``ocrscout_install`` kwarg on ``launch`` for development from a git
ref (``ocrscout[vllm] @ git+https://github.com/storytracer/ocrscout.git@dev``).
"""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any, ClassVar

import yaml

from ocrscout import state as state_mod
from ocrscout.errors import RunnerError
from ocrscout.interfaces.runner import Runner
from ocrscout.profile import resolve
from ocrscout.runners._gpu_pricing import lookup as _lookup_pricing
from ocrscout.state import GpuConfig, ManagedProcess, RunnerStateFile
from ocrscout.types import (
    JobHandle,
    PipelineConfig,
    RunnerEndpoint,
    RunnerHandle,
    RunnerStatus,
)

log = logging.getLogger(__name__)

_DEFAULT_OCRSCOUT_INSTALL = "ocrscout[vllm]"
_DEFAULT_INFRA = "k8s"
_DEFAULT_CLOUD: str | None = None


class SkyPilotRunner(Runner):
    """Provisions and drives a SkyPilot pool on the active K8s context."""

    name = "skypilot"
    requires_local_gpu: ClassVar[bool] = False

    def launch(
        self,
        *,
        models: list[str],
        gpu_type: str | None = None,
        workers: int = 1,
        infra: str = _DEFAULT_INFRA,
        cloud: str | None = _DEFAULT_CLOUD,
        ocrscout_install: str = _DEFAULT_OCRSCOUT_INSTALL,
        pool_name: str | None = None,
        **_: Any,
    ) -> RunnerHandle:
        if not models:
            raise RunnerError("SkyPilotRunner.launch: at least one model required")
        if gpu_type is None:
            raise RunnerError(
                "SkyPilotRunner.launch: --gpu is required (e.g. L4, H100). "
                "SkyPilot needs the accelerator label to provision a pod."
            )

        # Resolve profiles up front so we fail fast on unknown names; the
        # actual model fetches happen on the worker side.
        for name in models:
            resolve(name)

        pool_name = pool_name or _derive_pool_name(models)
        log.info(
            "SkyPilotRunner: applying pool %s (gpu=%s, workers=%d, infra=%s)",
            pool_name, gpu_type, workers, infra or "default",
        )

        pool_yaml = _pool_yaml_str(
            workers=workers,
            gpu_type=gpu_type,
            infra=infra,
            cloud=cloud,
            ocrscout_install=ocrscout_install,
        )
        log.debug("Pool YAML:\n%s", pool_yaml)

        # SkyPilot's YAML must be written to disk for ``sky jobs pool apply``.
        # We keep the file under ~/.ocrscout/skypilot/ so subsequent ``down``
        # and reruns can find it.
        pool_dir = state_mod.state_dir() / "skypilot" / pool_name
        pool_dir.mkdir(parents=True, exist_ok=True)
        pool_path = pool_dir / "pool.yaml"
        pool_path.write_text(pool_yaml, encoding="utf-8")

        try:
            subprocess.run(
                ["sky", "jobs", "pool", "apply", str(pool_path), "--name", pool_name, "-y"],
                check=True,
            )
        except FileNotFoundError as e:
            raise RunnerError(
                "`sky` CLI not found on PATH; install the [skypilot] extra: "
                "`uv pip install 'ocrscout[skypilot]'` (or "
                "`uv pip install 'skypilot-nightly[kubernetes]'`)."
            ) from e
        except subprocess.CalledProcessError as e:
            raise RunnerError(
                f"`sky jobs pool apply` failed (exit {e.returncode}). "
                f"Run `sky check` to verify your K8s context is set up."
            ) from e

        gpu_price = _lookup_pricing(gpu_type, provider=cloud)
        state = RunnerStateFile(
            runner=self.name,
            models=models,
            proxy_url=None,
            processes=[
                ManagedProcess(name="skypilot-pool", handle=pool_name)
            ],
            gpu=GpuConfig(
                type=gpu_type,
                cost_per_hour=gpu_price.cost_per_hour,
                provider=gpu_price.provider,
            ),
            launched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            args={
                "pool_name": pool_name,
                "gpu_type": gpu_type,
                "workers": workers,
                "infra": infra,
                "cloud": cloud,
            },
        )
        state_mod.write_state(state)

        return RunnerHandle(
            runner=self.name,
            proxy_url="",  # remote; ocrscout doesn't talk to it directly.
            endpoints=[
                RunnerEndpoint(model=m, url=f"sky://pool/{pool_name}", pid=None)
                for m in models
            ],
            extra={"pool_name": pool_name},
        )

    def submit(
        self,
        *,
        config: PipelineConfig,
        resume: bool = False,
        num_jobs: int = 1,
        **_: Any,
    ) -> JobHandle:
        state = state_mod.read_state()
        if state is None or state.runner != self.name:
            raise RunnerError(
                "SkyPilotRunner.submit: no active pool. Call "
                "`ocrscout launch --runner skypilot --gpu <type> --models ...` first."
            )
        pool_name = next(
            (p.handle for p in state.processes if p.name == "skypilot-pool"), None
        )
        if pool_name is None:
            raise RunnerError(
                "SkyPilotRunner.submit: state.yaml missing pool handle"
            )

        job_id = uuid.uuid4().hex[:12]
        job_dir = state_mod.jobs_dir() / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        config_path = job_dir / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )

        job_yaml = _job_yaml_str(
            gpu=state.gpu,
            ocrscout_install=_DEFAULT_OCRSCOUT_INSTALL,
            num_jobs=num_jobs,
            resume=resume,
            config_basename=config_path.name,
        )
        job_yaml_path = job_dir / "job.yaml"
        job_yaml_path.write_text(job_yaml, encoding="utf-8")

        argv = [
            "sky", "jobs", "launch",
            "--pool", pool_name,
            "--num-jobs", str(num_jobs),
            "-y",
            "--workdir", str(job_dir),
            str(job_yaml_path),
        ]
        try:
            subprocess.run(argv, check=True)
        except subprocess.CalledProcessError as e:
            raise RunnerError(
                f"`sky jobs launch` failed (exit {e.returncode})"
            ) from e

        return JobHandle(
            job_id=job_id,
            runner=self.name,
            output_dir=str(config.output_dir) if config.output_dir else "",
        )

    def status(self) -> RunnerStatus:
        state = state_mod.read_state()
        if state is None or state.runner != self.name:
            return RunnerStatus(runner=self.name, state="down")
        pool_name = next(
            (p.handle for p in state.processes if p.name == "skypilot-pool"), None
        )

        details: dict[str, Any] = {"pool": pool_name}
        sky_state: str = "ready"
        try:
            pool_status = subprocess.check_output(
                ["sky", "jobs", "pool", "status", "--json"],
                text=True,
            )
            details["pool_status"] = json.loads(pool_status)
        except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as e:
            sky_state = "error"
            details["pool_status_error"] = str(e)

        try:
            queue = subprocess.check_output(
                ["sky", "jobs", "queue", "--json"], text=True
            )
            details["queue"] = json.loads(queue)
        except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as e:
            details["queue_error"] = str(e)

        return RunnerStatus(
            runner=self.name,
            state=sky_state,
            models=state.models,
            proxy_url=None,
            details=details,
        )

    def logs(self, job_id: str | None = None, *, follow: bool = True) -> None:
        if job_id is None:
            raise RunnerError(
                "SkyPilotRunner.logs: --job-id is required (remote logs "
                "don't share a single tail stream)."
            )
        argv = ["sky", "jobs", "logs", job_id]
        if not follow:
            argv.append("--no-follow")
        try:
            subprocess.run(argv, check=True)
        except subprocess.CalledProcessError as e:
            raise RunnerError(
                f"`sky jobs logs {job_id}` failed (exit {e.returncode})"
            ) from e

    def down(self) -> None:
        state = state_mod.read_state()
        if state is None or state.runner != self.name:
            log.info("SkyPilotRunner: nothing to tear down")
            return
        pool_name = next(
            (p.handle for p in state.processes if p.name == "skypilot-pool"), None
        )
        if pool_name is not None:
            try:
                subprocess.run(
                    ["sky", "jobs", "pool", "down", pool_name, "-y"],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                log.warning(
                    "`sky jobs pool down %s` failed (%s); state cleared anyway.",
                    pool_name, e,
                )
        state_mod.clear_state()
        log.info("SkyPilotRunner: torn down")


def _derive_pool_name(models: list[str]) -> str:
    """A short, deterministic-ish pool name for a model set."""
    prefix = "-".join(m.replace("/", "_") for m in sorted(set(models)))[:30]
    return f"ocrscout-{prefix}-{uuid.uuid4().hex[:6]}"


def _pool_yaml_str(
    *,
    workers: int,
    gpu_type: str,
    infra: str,
    cloud: str | None,
    ocrscout_install: str,
) -> str:
    """Generate a SkyPilot pool YAML.

    Pool YAMLs declare the resource shape + setup; per-job YAMLs (see
    ``_job_yaml_str``) target the pool by name and only specify the run
    command.
    """
    body: dict[str, Any] = {
        "pool": {"workers": workers},
        "resources": {
            "accelerators": f"{gpu_type}:1",
            "infra": infra,
        },
        "setup": _setup_script(ocrscout_install),
    }
    if cloud:
        body["resources"]["cloud"] = cloud
    return yaml.safe_dump(body, sort_keys=False)


def _job_yaml_str(
    *,
    gpu: GpuConfig,
    ocrscout_install: str,
    num_jobs: int,
    resume: bool,
    config_basename: str,
) -> str:
    """Generate a SkyPilot job YAML targeting an existing pool.

    The run block on each worker:
    * sources the local PipelineConfig (uploaded via ``--workdir`` from
      the orchestrator),
    * stamps GPU pricing via ``envs:`` so the worker's Parquet rows are
      self-describing,
    * derives ``--start-idx`` / ``--end-idx`` from ``$SKYPILOT_JOB_RANK``
      and ``$SKYPILOT_NUM_JOBS`` so each worker owns a non-overlapping
      slice of the deterministic sample.
    """
    envs: dict[str, str] = {
        "OCRSCOUT_GPU_TYPE": gpu.type,
        "OCRSCOUT_COST_PER_HOUR": f"{gpu.cost_per_hour:.6f}",
        "OCRSCOUT_PROVIDER": gpu.provider,
        "OCRSCOUT_NUM_JOBS": str(num_jobs),
    }

    resume_flag = "--resume" if resume else ""

    run_script = f"""
        export PATH="$HOME/.local/bin:$PATH"
        # Worker partitions the deterministic sample into N equal windows
        # based on its rank. Sources without native start_idx/end_idx
        # support fall through to islice (see SourceAdapter ABC).
        PAGES=$(yq '.sample // 0' < {config_basename})
        if [ "$PAGES" -gt 0 ] && [ "$OCRSCOUT_NUM_JOBS" -gt 1 ]; then
          STEP=$((PAGES / OCRSCOUT_NUM_JOBS))
          START_IDX=$((SKYPILOT_JOB_RANK * STEP))
          if [ "$SKYPILOT_JOB_RANK" -eq $((OCRSCOUT_NUM_JOBS - 1)) ]; then
            END_IDX=$PAGES
          else
            END_IDX=$((START_IDX + STEP))
          fi
          RANGE_FLAGS="--start-idx $START_IDX --end-idx $END_IDX"
        else
          RANGE_FLAGS=""
        fi
        # ``apply`` reads the PipelineConfig YAML, applies the rank-derived
        # range slice, and runs the pipeline through the same code path as
        # ``ocrscout run`` — including auto-launch of a worker-local
        # vLLM + LiteLLM stack (since runtime=vllm profiles need one).
        # Resume is a no-op on the very first run; on re-runs of the same
        # job (SkyPilot ``--max-restarts-on-errors``) the progress.json
        # in the output dir lets the second attempt skip done pages.
        ocrscout apply {config_basename} \\
          $RANGE_FLAGS \\
          {resume_flag}
    """.strip()

    body: dict[str, Any] = {
        "envs": envs,
        "setup": _setup_script(ocrscout_install),
        "run": run_script,
    }
    return yaml.safe_dump(body, sort_keys=False)


def _setup_script(ocrscout_install: str) -> str:
    return f"""
        set -ex
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        # yq is needed in the run block to read PipelineConfig YAML; ship it
        # via the same install. uv pip --system installs into the worker
        # pod's system Python so subsequent invocations don't pay env setup.
        uv pip install --system "{ocrscout_install}" yq
    """.strip()
