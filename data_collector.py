import psutil
import time
import subprocess

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_focused_pid() -> int | None:
    """
    Return the PID of the process that currently owns the focused window.
    Uses xdotool (X11) with a fallback to wmctrl.  Returns None if the
    active window cannot be determined (e.g. on a headless session).
    """
    try:
        win_id = subprocess.check_output(
            ["xdotool", "getactivewindow"], stderr=subprocess.DEVNULL
        ).decode().strip()
        pid = subprocess.check_output(
            ["xdotool", "getwindowpid", win_id], stderr=subprocess.DEVNULL
        ).decode().strip()
        return int(pid)
    except Exception:
        pass

    try:
        # wmctrl fallback: active window line starts with "0x...  -1"
        lines = subprocess.check_output(
            ["wmctrl", "-lp"], stderr=subprocess.DEVNULL
        ).decode().splitlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "-1":
                return int(parts[2])
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Core collection
# ---------------------------------------------------------------------------

def collect_process_data(duration: int = 60, interval: float = 1.0) -> list[list]:
    """
    Observe every running process for `duration` seconds, sampling every
    `interval` seconds.

    Per-process features tracked
    ─────────────────────────────
    idle_duration   – continuous seconds since the process last held focus
                      (proxy: time since last time it was the focused PID)
    focus_time      – total cumulative seconds the process was the focused
                      window during this observation window
    session_age     – seconds since the process was created (context feature)

    Label
    ─────
    1 = should be suspended (idle_duration > 30 s and focus_time == 0)
    0 = active / keep running
    """

    # pid → last-seen-as-focused timestamp
    last_focused: dict[int, float] = {}
    # pid → cumulative seconds in focus
    focus_accumulator: dict[int, float] = {}

    data: list[list] = []
    start = time.time()

    while time.time() - start < duration:
        tick_start = time.time()
        focused_pid = get_focused_pid()

        # Snapshot all live processes
        snapshot: dict[int, psutil.Process] = {}
        for p in psutil.process_iter(["pid", "create_time"]):
            try:
                snapshot[p.pid] = p
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        now = time.time()

        for pid, proc in snapshot.items():
            try:
                create_time = proc.info["create_time"]
            except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
                continue

            # ── Focus tracking ──────────────────────────────────────────
            if pid == focused_pid:
                last_focused[pid] = now
                focus_accumulator[pid] = focus_accumulator.get(pid, 0.0) + interval

            # Initialise last_focused for new processes
            if pid not in last_focused:
                last_focused[pid] = now          # treat as "just seen" on first encounter
                focus_accumulator[pid] = 0.0

            # ── Feature calculation ─────────────────────────────────────
            idle_duration = now - last_focused[pid]            # seconds idle
            focus_time    = focus_accumulator.get(pid, 0.0)    # seconds in focus
            session_age   = now - create_time                  # seconds alive

            # ── Labelling heuristic (ground-truth for training) ─────────
            # A process is a suspension candidate when it has been idle for
            # at least 30 s AND has received zero focus time this window.
            label = 1 if (idle_duration >= 30 and focus_time == 0) else 0

            data.append([idle_duration, focus_time, session_age, label])

        # Clean up PIDs that have died so memory doesn't grow unbounded
        dead = set(last_focused) - set(snapshot)
        for pid in dead:
            last_focused.pop(pid, None)
            focus_accumulator.pop(pid, None)

        # Honour the requested interval (account for processing time)
        elapsed = time.time() - tick_start
        sleep_for = max(0.0, interval - elapsed)
        time.sleep(sleep_for)

    return data
