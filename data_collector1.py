"""
data_collector.py
=================
Collects per-process idle-behaviour data for model training.

Safe processes (bash, sh, zsh, system daemons) are excluded entirely —
they never appear in the training dataset and cannot be labelled for suspension.

Focus detection supports both Wayland (GNOME gdbus) and X11 (xdotool/wmctrl).
"""

import os
import psutil
import subprocess
import time

# ── Processes excluded from data collection (and therefore from suspension) ──
SAFE_NAMES = {
    "bash", "sh", "zsh", "fish", "dash", "ksh", "tcsh", "csh",
    "systemd", "init", "login", "sshd", "dbus-daemon",
    "gnome-session", "gnome-shell", "Xorg", "X", "Xwayland",
    "python3", "python",
}

# ── Focus detection ──────────────────────────────────────────────────────────

def _session_type() -> str:
    return os.environ.get("XDG_SESSION_TYPE", "unknown").lower()


def get_focused_pid() -> int | None:
    """
    Return the PID of the currently focused window.
    Tries Wayland (GNOME gdbus) first, then X11 (xdotool / wmctrl).
    Returns None if focus cannot be determined.
    """
    if _session_type() == "wayland":
        # GNOME Wayland
        try:
            out = subprocess.check_output([
                "gdbus", "call", "--session",
                "--dest", "org.gnome.Shell",
                "--object-path", "/org/gnome/Shell",
                "--method", "org.gnome.Shell.Eval",
                "global.display.focus_window?.get_pid() ?? -1"
            ], stderr=subprocess.DEVNULL).decode().strip()
            pid = int(out.split(",")[-1].strip().strip(")'"))
            return pid if pid > 0 else None
        except Exception:
            pass

        # KDE Wayland
        try:
            wid = subprocess.check_output(
                ["qdbus", "org.kde.KWin", "/KWin", "activeWindow"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            out = subprocess.check_output(
                ["xprop", "-id", wid, "_NET_WM_PID"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            return int(out.split("=")[-1].strip())
        except Exception:
            pass

    # X11 — xdotool
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

    # X11 fallback — wmctrl
    try:
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


# ── Core collection ──────────────────────────────────────────────────────────

def collect_process_data(duration: int = 60, interval: float = 1.0) -> list[list]:
    """
    Observe every eligible running process for `duration` seconds, sampling
    every `interval` seconds.

    Features per sample
    ───────────────────
    idle_duration  – seconds since the process last held window focus
    focus_time     – cumulative seconds the process was the focused window
    session_age    – seconds since the process was created

    Label (heuristic — overridden by learned threshold in train_model.py)
    ─────
    1 = suspend candidate  (idle >= 30 s AND zero focus time)
    0 = active

    Excluded entirely
    ─────────────────
    bash, sh, zsh, system daemons (see SAFE_NAMES), PIDs < 100,
    and processes owned by other users.
    """
    last_focused:      dict[int, float] = {}
    focus_accumulator: dict[int, float] = {}
    current_uid = os.getuid()

    data: list[list] = []
    start = time.time()

    while time.time() - start < duration:
        tick_start  = time.time()
        focused_pid = get_focused_pid()
        now         = time.time()

        # Snapshot eligible processes
        snapshot: dict[int, psutil.Process] = {}
        for p in psutil.process_iter(["pid", "name", "create_time", "uids"]):
            try:
                name = (p.info["name"] or "").lower()
                uid  = p.info["uids"].real
                pid  = p.info["pid"]
            except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
                continue

            # Enforce safe-name blocklist and ownership check
            if name in SAFE_NAMES or pid < 100 or uid != current_uid:
                continue

            snapshot[pid] = p

        for pid, proc in snapshot.items():
            try:
                create_time = proc.info["create_time"]
            except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
                continue

            # Focus tracking
            if pid == focused_pid:
                last_focused[pid]      = now
                focus_accumulator[pid] = focus_accumulator.get(pid, 0.0) + interval

            if pid not in last_focused:
                last_focused[pid]      = now   # first encounter → treat as just seen
                focus_accumulator[pid] = 0.0

            idle_duration = now - last_focused[pid]
            focus_time    = focus_accumulator.get(pid, 0.0)
            session_age   = now - create_time

            # Heuristic label (train_model.py re-labels with learned threshold)
            label = 1 if (idle_duration >= 30 and focus_time == 0) else 0

            data.append([idle_duration, focus_time, session_age, label])

        # Clean up dead PIDs
        live = set(snapshot.keys())
        for pid in list(last_focused):
            if pid not in live:
                last_focused.pop(pid, None)
                focus_accumulator.pop(pid, None)

        elapsed   = time.time() - tick_start
        time.sleep(max(0.0, interval - elapsed))

    return data
