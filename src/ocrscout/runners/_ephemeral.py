"""Foreground-bound spawn primitive for ``LocalRunner``'s ephemeral path.

A separate module from :mod:`ocrscout.runners._daemon` on purpose. The
double-fork primitive over there is intentionally incompatible with
``PR_SET_PDEATHSIG`` — the whole point of daemonisation is that the
child outlives the launching process. The ephemeral path here wants the
opposite: every child should die when this Python process dies.

``EphemeralStack`` is the public surface. ``LocalRunner._launch_ephemeral``
instantiates one, spawns the LiteLLM proxy + N vLLM serves into it via
``spawn(cmd=...)``, and either lets it die with the orchestrator (Ctrl-C
on ``ocrscout run``) or calls ``terminate_all()`` explicitly during a
graceful shutdown.

Defense in depth on Linux:

1. **``PR_SET_PDEATHSIG=SIGTERM``** via a ctypes-bound ``prctl()`` set in
   each child's ``preexec_fn``. The kernel delivers SIGTERM to the child
   the moment its parent dies — survives SIGKILL of the orchestrator,
   the orchestrator panicking before atexit, OOM kills, anything.
2. **``start_new_session=True``** on every Popen so SIGINT to the
   orchestrator's controlling terminal doesn't double-deliver to the
   children's process groups. The orchestrator handles the signal and
   propagates explicit termination instead.
3. **atexit registration** that calls ``terminate_all()``. Catches the
   common case where the orchestrator exits cleanly from a non-signal
   path (an exception, ``sys.exit``).
4. **Signal handlers for SIGINT/SIGTERM/SIGHUP** that call
   ``terminate_all()`` and re-raise the default behavior, so even Python
   threads holding the GIL don't delay teardown.

On macOS, layers 1 is unavailable (no ``PR_SET_PDEATHSIG`` syscall).
Layers 2-4 still apply, but SIGKILL of the orchestrator leaks children
— ``ocrscout down --force`` is the documented recovery path.

This module never touches ``~/.ocrscout/state.yaml`` or
``~/.ocrscout/pids/``: by design the ephemeral path leaves no on-disk
traces. Log files still land in ``log_dir()`` so ``ocrscout logs`` can
tail them while the run is live.
"""

from __future__ import annotations

import atexit
import ctypes
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ocrscout.runners._daemon import pid_alive, terminate

log = logging.getLogger(__name__)

_PR_SET_PDEATHSIG = 1


def _is_ephemeral_handler(handler: Any) -> bool:
    """True if ``handler`` is a bound ``_signal_handler`` on some ``EphemeralStack``.

    Used by handler installation to avoid capturing a prior stack's handler
    as our "fallback" — chaining through a torn-down stack adds no value
    and obscures stack traces.
    """
    if not callable(handler):
        return False
    self_ref = getattr(handler, "__self__", None)
    return isinstance(self_ref, EphemeralStack)


def _set_pdeathsig() -> None:
    """preexec_fn: ``prctl(PR_SET_PDEATHSIG, SIGTERM)``.

    Linux only. No-op on macOS. Errors are swallowed — PDEATHSIG is a
    best-effort safety net, not load-bearing for correctness (layers
    2-4 handle teardown when this is unavailable).
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # int prctl(int option, unsigned long arg2, ...);
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:  # noqa: BLE001
        # Don't break the spawn just because PDEATHSIG couldn't be set —
        # the parent's atexit + signal handlers carry the responsibility.
        pass


@dataclass
class EphemeralProcess:
    """One subprocess.Popen managed by an EphemeralStack.

    Carried for log-prefixing and teardown ordering only — no PID file
    is written. The Popen object holds the process group leader (since
    we spawn with ``start_new_session=True``) so ``os.killpg`` on
    ``popen.pid`` reaches grandchildren too.
    """

    name: str
    popen: subprocess.Popen
    port: int | None
    log_path: Path
    log_fd: int  # file descriptor we opened for the child's stdout/stderr
    started_at: float


class EphemeralStack:
    """LiteLLM proxy + N vLLM serves whose lifetime is yoked to this process.

    Use as a context manager (``with EphemeralStack() as stack: ...``)
    when ergonomic, or hold a reference and call ``terminate_all()``
    explicitly when ownership is shared with a long-lived object (as
    ``LocalRunner`` does).
    """

    _registered_signals: ClassVar[tuple[int, ...]] = (
        signal.SIGINT, signal.SIGTERM, signal.SIGHUP,
    )

    def __init__(self) -> None:
        self.processes: list[EphemeralProcess] = []
        self._prior_handlers: dict[int, Any] = {}
        self._atexit_registered = False
        self._closed = False

    # --- public API --------------------------------------------------

    def spawn(
        self,
        cmd: list[str],
        *,
        name: str,
        log_path: Path,
        port: int | None = None,
        env: dict[str, str] | None = None,
    ) -> EphemeralProcess:
        """Popen ``cmd`` with PDEATHSIG + new session; redirect stdio
        to ``log_path``; register the child with this stack.

        Installs the parent-side signal/atexit handlers on first spawn
        (idempotent). The child's stdout and stderr go to ``log_path``
        (append mode) and stdin is ``/dev/null``.
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fd = os.open(
            str(log_path),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            popen = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_fd,
                stderr=log_fd,
                start_new_session=True,
                preexec_fn=(
                    _set_pdeathsig
                    if sys.platform.startswith("linux") else None
                ),
                env={**os.environ, **(env or {})},
            )
        except BaseException:
            try:
                os.close(log_fd)
            except OSError:
                pass
            raise

        proc = EphemeralProcess(
            name=name,
            popen=popen,
            port=port,
            log_path=log_path,
            log_fd=log_fd,
            started_at=time.monotonic(),
        )
        self.processes.append(proc)
        log.debug("ephemeral spawn: %s pid=%d port=%s log=%s",
                  name, popen.pid, port, log_path)

        self._install_handlers_if_needed()
        return proc

    def terminate_all(self, *, grace: float = 10.0) -> None:
        """Reverse-order teardown: SIGTERM each child's process group,
        wait up to ``grace`` seconds, then SIGKILL the group.

        Idempotent: safe to call from atexit, signal handlers, and an
        explicit shutdown path. After the first call subsequent calls
        are no-ops.
        """
        if self._closed:
            return
        self._closed = True

        if not self.processes:
            self._restore_handlers()
            return

        log.debug("EphemeralStack: terminating %d child(ren)",
                  len(self.processes))
        # Reverse order matches daemonised teardown — proxy dies before
        # the upstream serves so transient "model X gone" errors don't
        # surface during the shutdown window.
        for proc in reversed(self.processes):
            if proc.popen.poll() is not None:
                continue  # already exited
            if not pid_alive(proc.popen.pid):
                continue
            try:
                terminate(proc.popen.pid, grace=grace)
            except Exception as e:  # noqa: BLE001
                log.warning("ephemeral terminate(%s) raised: %s",
                            proc.name, e)

        for proc in self.processes:
            try:
                os.close(proc.log_fd)
            except OSError:
                pass
        self.processes.clear()
        self._restore_handlers()

    def __enter__(self) -> EphemeralStack:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.terminate_all()

    # --- handler installation ---------------------------------------

    def _install_handlers_if_needed(self) -> None:
        if self._atexit_registered:
            return
        self._atexit_registered = True
        atexit.register(self.terminate_all)
        for sig in self._registered_signals:
            try:
                prior = signal.getsignal(sig)
            except (ValueError, OSError):
                continue
            # Don't chain through another EphemeralStack's handler. When
            # model-major chunking creates a fresh stack per chunk, the
            # previous stack may still be in mid-teardown (or have not
            # restored its handlers yet for some reason); capturing its
            # bound method as `prior` would ricochet a signal through a
            # closed stack instead of going to the OS default.
            if _is_ephemeral_handler(prior):
                prior = signal.SIG_DFL
            self._prior_handlers[sig] = prior
            try:
                signal.signal(sig, self._signal_handler)
            except (ValueError, OSError) as e:
                # Some environments (non-main thread, restricted
                # contexts) reject signal() — fine; atexit still covers
                # us on normal exit.
                log.debug("could not install handler for signal %d: %s",
                          sig, e)

    def _restore_handlers(self) -> None:
        for sig, prior in self._prior_handlers.items():
            try:
                signal.signal(sig, prior)
            except (ValueError, OSError, TypeError):
                pass
        self._prior_handlers.clear()

    def _signal_handler(self, signum: int, frame) -> None:
        # Run teardown synchronously so children die before this
        # process does. Then re-raise the original handler if it was
        # a callable (Python's default SIGINT raises KeyboardInterrupt;
        # SIGTERM/SIGHUP default exits the process).
        #
        # Critically, do NOT use the ``logging`` module here. Python
        # delivers signals on the main thread, possibly mid-``emit()``
        # of another log record; a nested ``log.info(...)`` would
        # re-enter the same non-reentrant ``BufferedWriter`` and raise
        # ``RuntimeError: reentrant call inside <_io.BufferedWriter
        # name='<stderr>'>``. Worse: ``Handler.handleError`` only
        # catches ``OSError``, so the ``RuntimeError`` escapes the
        # signal handler before ``terminate_all()`` ever runs, leaking
        # the entire daemon stack until the user Ctrl-C's again
        # (observed: 9-hour stall on a benchmark run; vLLM idle the
        # whole time). Two defenses:
        #   1. ``os.write(2, ...)`` bypasses the logging module *and*
        #      the BufferedWriter — it's the lowest-level signal-safe
        #      stderr we have.
        #   2. ``terminate_all()`` runs inside ``try/except
        #      BaseException`` so even a freak failure can't skip the
        #      child cleanup.
        try:
            self.terminate_all()
        except BaseException:  # noqa: BLE001
            # Last-resort: never let cleanup errors prevent the
            # process from honoring the original signal disposition.
            pass
        try:
            os.write(
                2,
                f"EphemeralStack: signal {signum} received; "
                f"stack terminated\n".encode(),
            )
        except OSError:
            pass

        prior = self._prior_handlers.get(signum)
        if signum == signal.SIGINT:
            # Surface Ctrl-C to the orchestrator as KeyboardInterrupt
            # so its existing try/except/finally chain runs.
            raise KeyboardInterrupt
        if callable(prior) and prior not in (
            signal.SIG_DFL, signal.SIG_IGN,
        ):
            try:
                prior(signum, frame)
                return
            except Exception:  # noqa: BLE001
                pass
        # Default for SIGTERM / SIGHUP: exit promptly.
        sys.exit(128 + signum)
