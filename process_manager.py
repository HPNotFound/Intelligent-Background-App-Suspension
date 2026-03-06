import os
import signal
import psutil

# Shared state: pid → process name
suspended: dict[int, str] = {}


def suspend_process(pid: int, name: str = "") -> bool:
    """
    Send SIGSTOP to `pid`.  Returns True on success, False otherwise.
    Skips the process if it is already stopped.
    """
    if pid in suspended:
        return False   # already tracked as suspended
    try:
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_STOPPED:
            return False
        # Only suspend processes owned by the current user
        import os as _os
        if proc.uids().real != _os.getuid():
            return False
        os.kill(pid, signal.SIGSTOP)
        suspended[pid] = name or proc.name()
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError, PermissionError):
        return False


def resume_process(pid: int) -> bool:
    """
    Send SIGCONT to `pid`.  Returns True on success, False otherwise.
    """
    try:
        os.kill(pid, signal.SIGCONT)
        suspended.pop(pid, None)
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError, PermissionError):
        suspended.pop(pid, None)   # clean up stale entry
        return False


def resume_all() -> list[int]:
    """
    Resume every currently suspended process.
    Returns list of PIDs that were successfully resumed.
    """
    resumed = []
    for pid in list(suspended.keys()):
        if resume_process(pid):
            resumed.append(pid)
    return resumed


def get_suspended() -> dict[int, str]:
    """Return a snapshot copy of the suspended pid→name map."""
    return dict(suspended)
