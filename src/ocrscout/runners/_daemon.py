"""Daemonisation primitives for ``LocalRunner``.

Spawns long-lived child processes (vLLM serves, LiteLLM proxy, page-dispatch
workers) that survive the launching CLI's exit. ``start_new_session=True``
makes each child its own session leader so the controlling terminal's
SIGHUP/Ctrl-C doesn't cascade; ``stdin`` is closed and ``stdout``/``stderr``
are routed to a log file. The PID is written atomically to ``pid_path``
so subsequent CLI invocations can find and signal the daemon without any
in-memory state.

On Linux we additionally ask the kernel to SIGTERM the child if the
launching process dies abnormally (``PR_SET_PDEATHSIG``) — useful when the
launching CLI is killed by OOM, ``kill -9``, or a screen-cascade. On
macOS we degrade silently; the process group + state-file PID tracking
still gives a working shutdown path.

PID files contain a single integer (no JSON envelope) so they can be
inspected and signalled with shell tools. Stale files (PID file present
but the PID is gone) are detectable via :func:`pid_alive` and silently
cleared by :func:`terminate`.
"""

from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_DEFAULT_GRACE_SECONDS = 10.0


def daemonize_subprocess(
    cmd: list[str],
    *,
    log_path: Path,
    pid_path: Path,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> int:
    """Spawn ``cmd`` as a fully-detached UNIX daemon. Returns the daemon's PID.

    Uses the classic double-fork idiom so the resulting daemon is
    reparented to ``init`` (PID 1, or ``launchd`` on macOS):

    1. Fork. The parent waits for the first child to exit and returns
       the grandchild's PID (communicated over a pipe).
    2. The first child calls ``setsid`` (detaching from the controlling
       terminal), forks the grandchild, writes its PID to the pipe, and
       exits.
    3. The grandchild (now reparented to init) redirects its stdio to
       ``log_path``, writes its own PID to ``pid_path``, and ``execvp``\\ s
       the requested command.

    Because the daemon is no longer a child of the launching process,
    the kernel auto-reaps it on exit — so ``pid_alive(pid)`` correctly
    returns ``False`` after the daemon dies. (A single-fork variant
    leaves the daemon as a zombie until the launching process reaps it,
    which made ``pid_alive`` give wrong answers from outside the
    spawning shell.)
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    pipe_r, pipe_w = os.pipe()
    parent_pid = os.fork()
    if parent_pid > 0:
        # Original process: read grandchild PID from pipe, reap the
        # first child, and return.
        os.close(pipe_w)
        try:
            pid_bytes = b""
            while True:
                chunk = os.read(pipe_r, 64)
                if not chunk:
                    break
                pid_bytes += chunk
        finally:
            os.close(pipe_r)
        os.waitpid(parent_pid, 0)
        text = pid_bytes.decode().strip()
        if not text:
            raise OSError("daemon failed to start (no PID reported)")
        if text.startswith("ERR "):
            raise OSError(f"daemon failed to start: {text[4:]}")
        return int(text)

    # First child — detach from controlling terminal and fork again.
    try:
        os.close(pipe_r)
        os.setsid()
        grandchild_pid = os.fork()
        if grandchild_pid > 0:
            # First child: write the PID file (so it's visible BEFORE the
            # parent's `daemonize_subprocess` returns), report grandchild
            # PID to the parent, and exit so the grandchild is orphaned
            # to init.
            try:
                _atomic_write_text(pid_path, str(grandchild_pid))
                os.write(pipe_w, str(grandchild_pid).encode())
            finally:
                os.close(pipe_w)
            os._exit(0)
    except Exception as e:  # noqa: BLE001
        try:
            os.write(pipe_w, f"ERR {type(e).__name__}: {e}".encode())
        except OSError:
            pass
        os._exit(1)

    # Grandchild — this becomes the daemon.
    try:
        os.close(pipe_w)
    except OSError:
        pass

    try:
        if cwd:
            os.chdir(cwd)
        os.umask(0o022)

        # Redirect stdio: stdin from /dev/null; stdout/stderr appending to log.
        with open(os.devnull, "rb") as devnull:
            os.dup2(devnull.fileno(), 0)
        log_fd = os.open(
            str(log_path),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        os.close(log_fd)

        # PID file was written by the first child before exit, so it
        # already exists on disk by the time the launching process
        # returns from daemonize_subprocess.

        # PR_SET_PDEATHSIG is meaningless after double-fork — the daemon's
        # parent is now init/launchd, and the whole point of daemonisation
        # is that we outlive the launching ocrscout process. Subsequent
        # ``ocrscout status``/``down`` invocations read state.yaml to find
        # the daemon, so the launching shell can close freely.

        new_env = {**os.environ, **(env or {})}
        os.execvpe(cmd[0], cmd, new_env)
    except Exception as e:  # noqa: BLE001
        # Last-ditch: write the error to the log so the parent's
        # readiness probe can surface it.
        try:
            with open(str(log_path), "a", encoding="utf-8") as f:
                f.write(f"daemonize exec failed: {type(e).__name__}: {e}\n")
        except OSError:
            pass
        os._exit(1)
    # Unreachable — execvpe replaces this process.
    return -1  # pragma: no cover


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` atomically (tmp file in the same dir + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_pid(pid_path: Path) -> int | None:
    """Read a PID file. Returns ``None`` if absent or malformed."""
    try:
        text = pid_path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def pid_alive(pid: int) -> bool:
    """Is the given PID alive? Uses ``os.kill(pid, 0)``."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        if e.errno == errno.ESRCH:
            return False
        if e.errno == errno.EPERM:
            # Process exists but we lack permission to signal it.
            return True
        return False
    return True


def terminate(
    pid_or_path: int | Path,
    *,
    grace: float = _DEFAULT_GRACE_SECONDS,
) -> bool:
    """SIGTERM → wait up to ``grace`` seconds → SIGKILL.

    Returns ``True`` when the process is gone after the sequence,
    ``False`` if we lost track of it entirely (PID file missing, never
    spawned, etc.). Cleans up the PID file on success when one was
    provided.

    Accepts either a raw PID (int) or a PID-file path so callers can hand
    in ``~/.ocrscout/pids/vllm-dots-mocr.pid`` directly.
    """
    pid_path: Path | None = None
    if isinstance(pid_or_path, Path):
        pid_path = pid_or_path
        pid = read_pid(pid_path)
        if pid is None:
            # Already gone — treat as success and tidy the file just in case.
            try:
                pid_path.unlink()
            except FileNotFoundError:
                pass
            return False
    else:
        pid = pid_or_path

    if not pid_alive(pid):
        if pid_path is not None:
            try:
                pid_path.unlink()
            except FileNotFoundError:
                pass
        return False

    # Try to signal the whole process group so children of the daemon
    # (e.g. uv's grandchild vllm/litellm processes) also receive the
    # signal. Fall back to signalling the PID directly if the group lookup
    # fails (process already exited between checks).
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            if pid_path is not None:
                try:
                    pid_path.unlink()
                except FileNotFoundError:
                    pass
            return True
        time.sleep(0.2)

    # Still alive after grace — escalate to SIGKILL on the group.
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    # Wait a short window for the kernel to clean up the process table
    # entry; SIGKILL is essentially instant but we don't rely on it.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            if pid_path is not None:
                try:
                    pid_path.unlink()
                except FileNotFoundError:
                    pass
            return True
        time.sleep(0.1)
    return False


def uptime_seconds(pid: int) -> float | None:
    """Best-effort wall-clock uptime of a PID, in seconds.

    Returns ``None`` if the process is gone or the platform doesn't expose
    creation time cheaply. Used by ``RunnerStatus`` to surface "vllm
    serve has been up for X minutes."
    """
    if not pid_alive(pid):
        return None
    try:
        if sys.platform == "darwin":
            # ``ps -o lstart=`` is parseable but slow; use ``etime`` for a
            # direct duration string.
            out = subprocess.check_output(
                ["ps", "-o", "etime=", "-p", str(pid)],
                text=True,
            ).strip()
            return _parse_etime(out)
        if sys.platform.startswith("linux"):
            with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
                fields = f.read().split()
            # Field 22 (0-indexed 21) is starttime in clock ticks since boot.
            starttime_ticks = int(fields[21])
            hz = os.sysconf("SC_CLK_TCK") or 100
            boot_time = _linux_boot_time()
            if boot_time is None:
                return None
            return max(0.0, time.time() - (boot_time + starttime_ticks / hz))
    except (subprocess.CalledProcessError, OSError, ValueError):
        return None
    return None


def _parse_etime(s: str) -> float | None:
    """Parse ``ps -o etime`` output: ``[[dd-]hh:]mm:ss``."""
    if not s:
        return None
    days = 0
    if "-" in s:
        day_part, _, rest = s.partition("-")
        days = int(day_part)
        s = rest
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = (int(x) for x in parts)
        elif len(parts) == 2:
            h = 0
            m, sec = (int(x) for x in parts)
        else:
            return None
    except ValueError:
        return None
    return days * 86400 + h * 3600 + m * 60 + sec


def _linux_boot_time() -> float | None:
    try:
        with open("/proc/stat", encoding="utf-8") as f:
            for line in f:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except OSError:
        return None
    return None
