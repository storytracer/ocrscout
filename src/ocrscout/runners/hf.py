"""HuggingFaceRunner: remote OCR via the HuggingFace Jobs API.

For the BHL / FineBooks pilot with Daniel van Strien — HF-sponsored
compute provisions a GPU pod, installs ``ocrscout[vllm]`` via ``uv``,
and runs ``ocrscout apply`` against a PipelineConfig uploaded to the
job's workspace. Source / output paths use the ``hf://`` scheme so
datasets stay on HF Hub without ever touching S3.

The HF Jobs API is intentionally narrower than SkyPilot's pool/job
abstraction: each ``launch`` provisions a single GPU pod (no multi-worker
pool), each ``submit`` is one HF Job, and ``--workers`` is only used by
the source partitioning math. If you need parallel workers, submit
multiple jobs with non-overlapping ``--start-idx``/``--end-idx`` windows.

``huggingface_hub`` is lazy-imported so ``import
ocrscout.runners.hf`` succeeds without the ``[hf]`` extra. Actual API
calls require ``HF_TOKEN`` in the environment.
"""

from __future__ import annotations

import logging
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
# HF's "flavor" identifiers for GPU pods. The user-facing GPU labels
# (L4, A10G, H100) are mapped onto these here. Keep in sync with HF's
# documented set; unknown values bubble up to HF as-is.
_HF_FLAVOR: dict[str, str] = {
    "T4": "t4-small",
    "L4": "l4x1",
    "A10G": "a10g-large",
    "A100": "a100-large",
    "H100": "h100x1",
}


class HuggingFaceRunner(Runner):
    """Single-pod HF Jobs API runner."""

    name = "hf"
    requires_local_gpu: ClassVar[bool] = False

    def launch(
        self,
        *,
        models: list[str],
        gpu_type: str | None = None,
        workers: int = 1,
        flavor: str | None = None,
        space_id: str | None = None,
        ocrscout_install: str = _DEFAULT_OCRSCOUT_INSTALL,
        **_: Any,
    ) -> RunnerHandle:
        if not models:
            raise RunnerError("HuggingFaceRunner.launch: at least one model required")
        if gpu_type is None and flavor is None:
            raise RunnerError(
                "HuggingFaceRunner.launch: pass --gpu (L4/A10G/A100/H100/T4) "
                "or --flavor (HF's pod flavor identifier)."
            )

        # Resolve profiles up front so unknown names fail fast.
        for name in models:
            resolve(name)

        hf_flavor = flavor or _HF_FLAVOR.get(gpu_type or "", "")
        if not hf_flavor:
            raise RunnerError(
                f"HuggingFaceRunner: unknown gpu_type {gpu_type!r}; pass "
                f"--flavor explicitly or use one of: "
                f"{sorted(_HF_FLAVOR.keys())}."
            )

        gpu_price = _lookup_pricing(gpu_type or "", provider="hf")
        # HF doesn't publish a unified pricing table here, so cost_per_hour
        # may be 0.0 — token-based cost via litellm.completion_cost still
        # populates litellm_cost on every row.

        state = RunnerStateFile(
            runner=self.name,
            models=models,
            proxy_url=None,
            processes=[
                ManagedProcess(name="hf-launch", handle=f"flavor:{hf_flavor}")
            ],
            gpu=GpuConfig(
                type=gpu_type or hf_flavor,
                cost_per_hour=gpu_price.cost_per_hour,
                provider="huggingface",
            ),
            launched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            args={
                "flavor": hf_flavor,
                "space_id": space_id,
                "ocrscout_install": ocrscout_install,
                "workers": workers,
            },
        )
        state_mod.write_state(state)

        return RunnerHandle(
            runner=self.name,
            proxy_url="",  # remote; each HF Job runs its own LocalRunner.
            endpoints=[
                RunnerEndpoint(model=m, url=f"hf://flavor/{hf_flavor}", pid=None)
                for m in models
            ],
            extra={"flavor": hf_flavor, "space_id": space_id},
        )

    def submit(
        self,
        *,
        config: PipelineConfig,
        resume: bool = False,
        **_: Any,
    ) -> JobHandle:
        state = state_mod.read_state()
        if state is None or state.runner != self.name:
            raise RunnerError(
                "HuggingFaceRunner.submit: no active launch. Call "
                "`ocrscout launch --runner hf --gpu <type> --models ...` first."
            )

        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise RunnerError(
                "huggingface_hub is required; install with "
                "`uv pip install 'ocrscout[hf]'`."
            ) from e

        job_id = uuid.uuid4().hex[:12]
        job_dir = state_mod.jobs_dir() / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        config_path = job_dir / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )

        # Build the HF Job's command and environment.
        flavor = state.args.get("flavor")
        ocrscout_install = state.args.get(
            "ocrscout_install", _DEFAULT_OCRSCOUT_INSTALL
        )
        resume_flag = "--resume" if resume else ""
        # HF Jobs receive the workdir contents at /workspace by default; the
        # exact path depends on the API. We use a known path and copy the
        # PipelineConfig from the orchestrator-side workdir snapshot.
        command = " && ".join(
            [
                'curl -LsSf https://astral.sh/uv/install.sh | sh',
                'export PATH="$HOME/.local/bin:$PATH"',
                f'uv pip install --system "{ocrscout_install}"',
                f"ocrscout apply config.yaml {resume_flag}",
            ]
        ).strip()

        envs = {
            "OCRSCOUT_GPU_TYPE": state.gpu.type,
            "OCRSCOUT_COST_PER_HOUR": f"{state.gpu.cost_per_hour:.6f}",
            "OCRSCOUT_PROVIDER": state.gpu.provider,
        }

        api = HfApi()
        try:
            # The Jobs API in huggingface_hub is still maturing; we use
            # ``run_job`` if available, else fall back to whatever the
            # installed version supports. This keeps us forward-compatible
            # as the API stabilises.
            run_job = getattr(api, "run_job", None)
            if run_job is None:
                raise RunnerError(
                    "huggingface_hub.HfApi has no `run_job`; upgrade with "
                    "`uv pip install -U huggingface_hub`."
                )
            hf_job = run_job(
                command=command,
                flavor=flavor,
                env=envs,
                # The PipelineConfig YAML is uploaded as a file on the job;
                # the exact kwarg name has been ``inputs`` / ``files`` /
                # ``workdir`` across HF SDK versions. Pass via inputs and
                # let HF's own deprecation message guide future updates.
                inputs={"config.yaml": str(config_path)},
            )
        except RunnerError:
            raise
        except Exception as e:  # noqa: BLE001
            raise RunnerError(
                f"HuggingFaceRunner.submit: HF Jobs API call failed: {e}"
            ) from e

        hf_job_id = getattr(hf_job, "id", None) or getattr(hf_job, "job_id", None) or str(hf_job)
        (job_dir / "hf_job_id").write_text(str(hf_job_id), encoding="utf-8")
        log.info("HF Job submitted: hf_id=%s ocrscout_id=%s", hf_job_id, job_id)

        return JobHandle(
            job_id=job_id,
            runner=self.name,
            output_dir=str(config.output_dir) if config.output_dir else "",
        )

    def status(self) -> RunnerStatus:
        state = state_mod.read_state()
        if state is None or state.runner != self.name:
            return RunnerStatus(runner=self.name, state="down")

        try:
            from huggingface_hub import HfApi
        except ImportError:
            return RunnerStatus(
                runner=self.name,
                state="ready",  # we recorded launch state, just can't query
                models=state.models,
                details={"warning": "huggingface_hub not installed; can't query"},
            )

        api = HfApi()
        details: dict[str, Any] = {"flavor": state.args.get("flavor")}
        # List active jobs for the user. HF's job-listing API has shifted
        # across versions; degrade to "we recorded launch" if it errors.
        try:
            list_jobs = getattr(api, "list_jobs", None)
            if list_jobs is not None:
                details["jobs"] = [_jobinfo(j) for j in list_jobs()]
        except Exception as e:  # noqa: BLE001
            details["list_jobs_error"] = str(e)

        return RunnerStatus(
            runner=self.name,
            state="ready",
            models=state.models,
            details=details,
        )

    def logs(self, job_id: str | None = None, *, follow: bool = True) -> None:
        if job_id is None:
            raise RunnerError(
                "HuggingFaceRunner.logs: --job-id is required (HF Jobs API "
                "is per-job, not per-runner)."
            )
        # Map ocrscout job_id → HF job id via the file the runner wrote
        # at submit time.
        job_dir = state_mod.jobs_dir() / job_id
        hf_id_path = job_dir / "hf_job_id"
        if not hf_id_path.is_file():
            raise RunnerError(
                f"HuggingFaceRunner.logs: no hf_job_id recorded for {job_id}"
            )
        hf_id = hf_id_path.read_text(encoding="utf-8").strip()

        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise RunnerError(
                "huggingface_hub is required to tail HF Job logs"
            ) from e

        api = HfApi()
        stream = getattr(api, "stream_job_logs", None) or getattr(
            api, "get_job_logs", None
        )
        if stream is None:
            raise RunnerError(
                "huggingface_hub version too old to stream job logs; "
                "upgrade with `uv pip install -U huggingface_hub`."
            )
        for chunk in stream(hf_id):
            print(chunk, end="" if isinstance(chunk, str) else "\n")

    def down(self) -> None:
        state = state_mod.read_state()
        if state is None or state.runner != self.name:
            log.info("HuggingFaceRunner: nothing to tear down")
            return
        # No persistent pool to tear down; HF jobs auto-terminate when their
        # command exits. Cancelling in-flight jobs is a per-job operation
        # via ``ocrscout logs <job> --cancel`` (future) — for now ``down``
        # just clears the state.
        state_mod.clear_state()
        log.info("HuggingFaceRunner: state cleared")


def _jobinfo(j: Any) -> dict[str, Any]:
    """Coerce a huggingface_hub job-object to a small dict for status payload."""
    if isinstance(j, dict):
        return j
    out: dict[str, Any] = {}
    for key in ("id", "job_id", "status", "flavor", "command", "created_at"):
        val = getattr(j, key, None)
        if val is not None:
            out[key] = val
    return out
