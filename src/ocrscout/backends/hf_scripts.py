"""HfScriptsBackend: run a uv-scripts/ocr Python file in a `uv run` subprocess.

All IO is local — the input dataset is saved to a temp directory and the
upstream script's ``load_dataset`` and ``Dataset.push_to_hub`` calls are
redirected to local ``load_from_disk`` / ``save_to_disk`` via a shim
prepended into a temp copy of the script. PEP 723 metadata stays at the
top of the file so ``uv run`` still resolves the script's GPU dependencies.

No HF token is required for any of this. The script source itself is
downloaded from the public ``uv-scripts/ocr`` repo via
``huggingface_hub.hf_hub_download`` (token-optional, public dataset).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import deque
from collections.abc import Iterator, Sequence
from pathlib import Path

from datasets import Dataset, Features
from datasets import Image as ImageFeat
from datasets import Value, load_from_disk

from ocrscout.errors import BackendError
from ocrscout.interfaces.backend import ModelBackend
from ocrscout.profile import ModelProfile
from ocrscout.sync.cache import scripts_cache_dir
from ocrscout.types import BackendInvocation, PageImage, RawOutput

log = logging.getLogger(__name__)

# Sentinel string we pass as the upstream script's positional `input_dataset`
# argument; the shim recognizes it and routes to load_from_disk(OCRSCOUT_IN).
_INPUT_SENTINEL_PREFIX = "ocrscout-local://"

# Match the PEP 723 inline-metadata block. Used to find where to insert the shim.
_PEP723_RE = re.compile(
    r"^# /// script\s*$.*?^# ///\s*$",
    re.MULTILINE | re.DOTALL,
)

# Shim prepended into a temp copy of the upstream script. Runs after PEP 723
# is consumed by uv but before the script's own imports — so when the script
# does `from datasets import load_dataset` it gets our patched function.
_SHIM = '''\
# === ocrscout local-IO shim (auto-generated) ===
import os as _ocrscout_os

_OCRSCOUT_IN = _ocrscout_os.environ.get("OCRSCOUT_IN")
_OCRSCOUT_OUT = _ocrscout_os.environ.get("OCRSCOUT_OUT")
_OCRSCOUT_IN_REF = _ocrscout_os.environ.get("OCRSCOUT_IN_REF", "")

if _OCRSCOUT_IN and _OCRSCOUT_OUT:
    import datasets as _ocrscout_datasets
    from datasets import (
        Dataset as _OcrscoutDataset,
        DatasetDict as _OcrscoutDatasetDict,
        load_from_disk as _ocrscout_load_from_disk,
    )

    _ocrscout_orig_load = _ocrscout_datasets.load_dataset

    def _ocrscout_load_dataset(path, *args, **kwargs):
        # When the script asks for our sentinel, hand back the local dataset.
        if str(path) == _OCRSCOUT_IN_REF:
            kwargs.pop("split", None)
            return _ocrscout_load_from_disk(_OCRSCOUT_IN)
        return _ocrscout_orig_load(path, *args, **kwargs)

    _ocrscout_datasets.load_dataset = _ocrscout_load_dataset

    def _ocrscout_push(self, repo_id=None, *args, **kwargs):
        # Persist locally; the orchestrator will load_from_disk() on this path.
        self.save_to_disk(_OCRSCOUT_OUT)
        return None

    _OcrscoutDataset.push_to_hub = _ocrscout_push
    _OcrscoutDatasetDict.push_to_hub = _ocrscout_push

    try:
        from huggingface_hub import DatasetCard as _OcrscoutDatasetCard

        _OcrscoutDatasetCard.push_to_hub = lambda self, *a, **kw: None
    except Exception:
        pass

    try:
        import huggingface_hub as _ocrscout_hf

        _ocrscout_hf.login = lambda *a, **kw: None
    except Exception:
        pass
# === end ocrscout shim ===

'''


class HfScriptsBackend(ModelBackend):
    name = "hf_scripts"

    def prepare(
        self, profile: ModelProfile, pages: Sequence[PageImage]
    ) -> BackendInvocation:
        workdir = Path(tempfile.mkdtemp(prefix=f"ocrscout-{profile.name}-"))
        in_dir = workdir / "in"
        out_dir = workdir / "out"
        wrapped_script = workdir / f"{profile.name}-wrapped.py"

        try:
            _build_input_dataset(pages, in_dir)
            script_source = _resolve_script_source(profile)
            _write_wrapped_script(script_source, wrapped_script)
        except Exception:
            shutil.rmtree(workdir, ignore_errors=True)
            raise

        in_ref = f"{_INPUT_SENTINEL_PREFIX}{profile.name}-{uuid.uuid4().hex[:8]}"
        # Output ref is unused by the shim (push_to_hub routes to OCRSCOUT_OUT
        # regardless), but the script still requires it as a positional arg.
        out_ref = f"{_INPUT_SENTINEL_PREFIX}{profile.name}-out"
        output_col = _output_col(profile)

        argv: list[str] = ["uv", "run", *profile.uv_args, str(wrapped_script),
            in_ref,
            out_ref,
            "--image-column",
            "image",
            "--output-column",
            output_col,
        ]
        if profile.preferred_prompt_mode:
            argv += ["--prompt-mode", profile.preferred_prompt_mode]
        for k, v in profile.backend_args.items():
            argv += [f"--{k.replace('_', '-')}", str(v)]

        return BackendInvocation(
            kind="subprocess",
            argv=argv,
            profile=profile,
            pages=[p.page_id for p in pages],
            extra={
                "workdir": str(workdir),
                "in_dir": str(in_dir),
                "out_dir": str(out_dir),
                "in_ref": in_ref,
                "output_col": output_col,
            },
        )

    def run(self, invocation: BackendInvocation) -> Iterator[RawOutput]:
        if invocation.argv is None:
            raise BackendError("HfScriptsBackend.run: invocation has no argv")

        workdir = invocation.extra.get("workdir")
        in_dir = invocation.extra.get("in_dir")
        out_dir = invocation.extra.get("out_dir")
        in_ref = invocation.extra.get("in_ref")
        output_col = invocation.extra.get("output_col", "markdown")

        env = {
            **os.environ,
            **invocation.profile.env,
            "OCRSCOUT_IN": str(in_dir),
            "OCRSCOUT_OUT": str(out_dir),
            "OCRSCOUT_IN_REF": str(in_ref),
        }

        try:
            print(f"  $ {' '.join(invocation.argv)}", flush=True)
            tail_buffer: deque[str] = deque(maxlen=400)
            proc = subprocess.Popen(
                invocation.argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                tail_buffer.append(line)
            returncode = proc.wait()
            if returncode != 0:
                raise BackendError(
                    f"HfScriptsBackend: script exited {returncode}; last lines:\n"
                    + "".join(list(tail_buffer)[-100:])
                )

            try:
                ds = load_from_disk(out_dir)
            except Exception as e:
                raise BackendError(
                    f"HfScriptsBackend: failed to load local output dataset at {out_dir!r}: {e}"
                ) from e

            if "page_id" not in ds.column_names:
                raise BackendError(
                    f"HfScriptsBackend: output dataset is missing 'page_id' "
                    f"column (have {ds.column_names!r})"
                )

            index = {row["page_id"]: row for row in ds}
            for page_id in invocation.pages:
                row = index.get(page_id)
                if row is None:
                    yield RawOutput(
                        page_id=page_id,
                        output_format=invocation.profile.output_format,
                        payload="",
                        error="page missing in script output",
                    )
                    continue
                payload = row.get(output_col)
                if payload is None:
                    yield RawOutput(
                        page_id=page_id,
                        output_format=invocation.profile.output_format,
                        payload="",
                        error=f"output column {output_col!r} is null for this row",
                    )
                    continue
                yield RawOutput(
                    page_id=page_id,
                    output_format=invocation.profile.output_format,
                    payload=str(payload),
                )
        finally:
            if workdir:
                shutil.rmtree(workdir, ignore_errors=True)


def _build_input_dataset(pages: Sequence[PageImage], in_dir: Path) -> None:
    """Materialize the page list as an HF dataset on local disk."""
    in_dir.parent.mkdir(parents=True, exist_ok=True)
    features = Features({"page_id": Value("string"), "image": ImageFeat()})
    ds = Dataset.from_dict(
        {"page_id": [p.page_id for p in pages], "image": [p.image for p in pages]},
        features=features,
    )
    ds.save_to_disk(str(in_dir))


def _resolve_script_source(profile: ModelProfile) -> Path:
    """Locate a local copy of the upstream script's source.

    Order: the ``ocrscout sync`` cache → ``hf_hub_download`` from the public
    upstream dataset (no token required).
    """
    if not profile.script:
        raise BackendError(
            f"HfScriptsBackend: profile {profile.name!r} has no `script` field"
        )
    filename = Path(profile.script).name
    repo = profile.repo or "uv-scripts/ocr"

    cached = scripts_cache_dir() / filename
    if cached.is_file():
        return cached

    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
        )
        return Path(local)
    except Exception as e:
        raise BackendError(
            f"HfScriptsBackend: cannot locate script {filename!r} for profile "
            f"{profile.name!r} (cache miss and download failed): {e}"
        ) from e


def _write_wrapped_script(original: Path, wrapped: Path) -> None:
    """Write a temp copy of ``original`` with the local-IO shim inserted.

    The shim goes immediately after the PEP 723 metadata block (so ``uv run``
    still finds the same dependency declaration) but before the script's own
    imports (so monkey-patches take effect at first use).
    """
    text = original.read_text(encoding="utf-8")
    m = _PEP723_RE.search(text)
    if m:
        idx = m.end()
        body = text[:idx] + "\n\n" + _SHIM + text[idx:]
    else:
        body = _SHIM + text
    wrapped.write_text(body, encoding="utf-8")


def _output_col(profile: ModelProfile) -> str:
    return str(profile.metadata.get("output_column", "markdown"))
