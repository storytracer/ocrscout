"""Port-by-PID resolution for ``LocalRunner``.

Two real callers:

1. **Persistent path post-readiness PID correction.** After ``vllm serve``
   reports ``/v1/models`` healthy, we want the *actual* server PID, not
   the transient ``uv run --with vllm -- vllm serve`` wrapper PID that
   ``daemonize_subprocess`` recorded. ``uv run`` resolves dependencies in
   a subprocess that exits as soon as deps are synced; the real server is
   a grandchild. We query the kernel for "who's listening on port N?"
   and rewrite the state.

2. **``ocrscout down --force`` belt-and-suspenders.** When a previous
   ``ocrscout run`` was Ctrl-C'd mid-launch and the orchestrator dies
   before writing ``state.yaml``, orphan vLLM servers can hold ports
   4000 / 8000-8000+N. ``down --force`` enumerates listeners on the
   default port range and SIGTERMs each one's process group.

Probe chain prefers ``ss -ltnp`` on Linux (iproute2, fast, single
syscall via netlink) and falls back to parsing ``/proc/net/tcp`` +
walking ``/proc/<pid>/fd/`` for stripped containers where ``ss`` is
absent. macOS / BSD uses ``lsof -iTCP -sTCP:LISTEN``.

No new dependencies — ``psutil`` is intentionally avoided. The shell-out
tools are universal: ``ss`` ships with iproute2 on every modern Linux
distribution, ``lsof`` is preinstalled on macOS.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from ocrscout.runners._daemon import terminate

log = logging.getLogger(__name__)


def resolve_listener_pid(port: int) -> int | None:
    """Return the PID listening on ``port`` (any local bind), or None.

    Tries ``ss`` first on Linux, falls back to ``/proc/net/tcp`` parsing,
    falls back to ``lsof`` on macOS. Every failure is logged at DEBUG
    and the next probe is attempted.
    """
    if sys.platform.startswith("linux"):
        pid = _probe_ss(port)
        if pid is not None:
            return pid
        return _probe_proc_net_tcp(port)
    if sys.platform == "darwin":
        return _probe_lsof(port)
    return _probe_lsof(port)


def listeners_on_ports(ports: Iterable[int]) -> dict[int, int]:
    """Batch resolution. Returns ``{port: pid}`` for ports with a listener.

    On Linux a single ``ss`` invocation covers all ports — faster than
    N individual probes when scanning a range during ``down --force``.
    On other platforms we degrade to per-port ``resolve_listener_pid``.
    """
    port_set = set(ports)
    if not port_set:
        return {}
    if sys.platform.startswith("linux"):
        batched = _probe_ss_batch(port_set)
        if batched is not None:
            return batched
        # ss failed entirely (not installed?); fall back to /proc/net/tcp
        # per-port. Less efficient but works everywhere.
        return {
            p: pid
            for p in port_set
            if (pid := _probe_proc_net_tcp(p)) is not None
        }
    return {
        p: pid
        for p in port_set
        if (pid := _probe_lsof(p)) is not None
    }


def kill_listener_on_port(port: int, *, grace: float = 10.0) -> bool:
    """Find the PID listening on ``port`` and SIGTERM its process group.

    Returns True if a listener existed and was signalled; False if the
    port was already free. SIGTERM-→-grace-→-SIGKILL semantics come from
    :func:`ocrscout.runners._daemon.terminate`.
    """
    pid = resolve_listener_pid(port)
    if pid is None:
        return False
    log.info("port %d: terminating listener pid %d", port, pid)
    terminate(pid, grace=grace)
    return True


# --- Linux: ss --------------------------------------------------------------


def _probe_ss(port: int) -> int | None:
    """Single-port ``ss`` probe. Returns the listener PID or None."""
    try:
        result = subprocess.run(
            ["ss", "-Hltnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=5.0, check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        log.debug("ss probe failed for port %d: %s", port, e)
        return None
    if result.returncode != 0:
        log.debug("ss returned %d for port %d: %s",
                  result.returncode, port, result.stderr.strip())
        return None
    return _parse_ss_first_pid(result.stdout)


def _probe_ss_batch(ports: set[int]) -> dict[int, int] | None:
    """Batch ``ss`` probe over a port set. Returns None if ss is unusable
    (not installed), {} when none of the ports have listeners.
    """
    try:
        # ``ss -Hltnp`` with no filter dumps every listening TCP socket.
        # Cheap (sub-millisecond) and means we don't need to express a
        # multi-port filter that ss's syntax doesn't cleanly support.
        result = subprocess.run(
            ["ss", "-Hltnp"],
            capture_output=True, text=True, timeout=5.0, check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        log.debug("ss batch probe failed: %s", e)
        return None
    if result.returncode != 0:
        log.debug("ss batch returned %d: %s",
                  result.returncode, result.stderr.strip())
        return None
    return _parse_ss_filter_ports(result.stdout, ports)


_SS_PID_RE = re.compile(r"pid=(\d+)")
# ``ss -Hltnp`` columns: State Recv-Q Send-Q LocalAddress:Port PeerAddress:Port
#   users:(("name",pid=NNN,fd=N),...)
# Local address is column 4; we extract the port suffix.
_SS_LOCAL_PORT_RE = re.compile(r"[:\[]([0-9]+)\s")


def _parse_ss_first_pid(stdout: str) -> int | None:
    """Pull the first pid=NNN from ``ss -Hltnp`` output."""
    m = _SS_PID_RE.search(stdout)
    return int(m.group(1)) if m else None


def _parse_ss_filter_ports(stdout: str, ports: set[int]) -> dict[int, int]:
    """Walk ``ss -Hltnp`` lines, return ``{port: pid}`` for matched ports."""
    out: dict[int, int] = {}
    for line in stdout.splitlines():
        if not line.strip():
            continue
        # Column-split: state, recv-q, send-q, local, peer, users:(...)
        parts = line.split(None, 5)
        if len(parts) < 5:
            continue
        local = parts[3]
        # Local is like "0.0.0.0:8000" or "[::]:8000" or "127.0.0.1:4000".
        # Take the segment after the LAST ":" since IPv6 addresses contain
        # colons.
        port_part = local.rsplit(":", 1)
        if len(port_part) != 2:
            continue
        try:
            port = int(port_part[1])
        except ValueError:
            continue
        if port not in ports:
            continue
        pid_match = _SS_PID_RE.search(line)
        if pid_match:
            out[port] = int(pid_match.group(1))
    return out


# --- Linux: /proc/net/tcp fallback -----------------------------------------


def _probe_proc_net_tcp(port: int) -> int | None:
    """Find the listener on ``port`` by walking ``/proc/net/tcp[6]``.

    Listening sockets have remote address ``0.0.0.0:0`` (or all-zero in
    IPv6) and state ``0A``. We extract the inode of the matching socket
    then walk ``/proc/<pid>/fd/`` for a symlink target ``socket:[<inode>]``.
    """
    inodes = _proc_net_tcp_listener_inodes(port)
    if not inodes:
        return None
    return _find_pid_for_any_inode(inodes)


def _proc_net_tcp_listener_inodes(port: int) -> set[int]:
    """Inodes of listening TCP sockets bound to ``port`` (v4 and v6).

    The hex column in ``/proc/net/tcp`` for the local address is
    "<be32 ipv4>:<be16 port>"; v6 is "<32 hex>:<4 hex>".
    """
    inodes: set[int] = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path, encoding="ascii") as f:
                next(f, None)  # header
                for line in f:
                    cols = line.split()
                    if len(cols) < 10:
                        continue
                    local = cols[1]
                    state = cols[3]
                    if state != "0A":  # 0A = TCP_LISTEN
                        continue
                    port_hex = local.rsplit(":", 1)[-1]
                    try:
                        if int(port_hex, 16) != port:
                            continue
                    except ValueError:
                        continue
                    try:
                        inodes.add(int(cols[9]))
                    except ValueError:
                        continue
        except OSError as e:
            log.debug("could not read %s: %s", path, e)
    return inodes


def _find_pid_for_any_inode(inodes: set[int]) -> int | None:
    """Walk ``/proc/*/fd/*`` looking for ``socket:[<inode>]``.

    Best-effort across PID directories that disappear mid-walk.
    """
    needles = {f"socket:[{i}]" for i in inodes}
    proc = Path("/proc")
    try:
        entries = list(proc.iterdir())
    except OSError as e:
        log.debug("could not list /proc: %s", e)
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        fd_dir = entry / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except (FileNotFoundError, PermissionError, OSError):
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target in needles:
                try:
                    return int(entry.name)
                except ValueError:
                    return None
    return None


# --- macOS / BSD: lsof ------------------------------------------------------


def _probe_lsof(port: int) -> int | None:
    """``lsof -iTCP:<port> -sTCP:LISTEN -P -n -F p`` parse.

    The ``-F p`` machine-readable output is one line per record, with
    ``p`` prefix for PIDs.
    """
    try:
        result = subprocess.run(
            ["lsof", "-iTCP:" + str(port), "-sTCP:LISTEN", "-P", "-n",
             "-F", "p"],
            capture_output=True, text=True, timeout=5.0, check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        log.debug("lsof probe failed for port %d: %s", port, e)
        return None
    # lsof exits 1 when no matches; that's "port free", not an error.
    if result.returncode not in (0, 1):
        log.debug("lsof returned %d for port %d: %s",
                  result.returncode, port, result.stderr.strip())
        return None
    for line in result.stdout.splitlines():
        if line.startswith("p"):
            try:
                return int(line[1:])
            except ValueError:
                continue
    return None
