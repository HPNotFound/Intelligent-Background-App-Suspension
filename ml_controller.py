import psutil
import time
import joblib
import subprocess
from process_manager import suspend_process, resume_process, suspended

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
try:
    model = joblib.load("process_classifier.pkl")
except FileNotFoundError:
    raise RuntimeError(
        "process_classifier.pkl not found. Run train_model.py first."
    )

# ---------------------------------------------------------------------------
# Focus tracking state  (mirrors data_collector.py exactly)
# ---------------------------------------------------------------------------
_last_focused: dict[int, float] = {}    # pid → timestamp last seen focused
_focus_accum:  dict[int, float] = {}    # pid → cumulative focus seconds

_running = False   # flag so the GUI can stop the loop


def get_focused_pid() -> int | None:
    """Return PID of the currently focused window (X11 via xdotool/wmctrl)."""
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


def stop_ml_controller():
    global _running
    _running = False


def run_ml_controller(callback):
    """
    Main loop.  Called in a background thread by the GUI.
    Features fed to the model must exactly match what was used in training:
        [idle_duration, focus_time, session_age]
    """
    global _running
    _running = True
    INTERVAL  = 5.0    # seconds between suspension checks
    SAFE_PIDS = {1}    # never touch init/systemd

    while _running:
        tick = time.time()
        focused_pid = get_focused_pid()

        for p in psutil.process_iter(["pid", "create_time", "name", "status"]):
            try:
                pid         = p.info["pid"]
                create_time = p.info["create_time"]
                name        = p.info["name"] or "unknown"
                status      = p.info["status"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

            if pid in SAFE_PIDS or pid < 100:
                continue

            now = tick

            # ── Update focus tracking ──────────────────────────────────
            if pid == focused_pid:
                _last_focused[pid] = now
                _focus_accum[pid]  = _focus_accum.get(pid, 0.0) + INTERVAL

            if pid not in _last_focused:
                _last_focused[pid] = now
                _focus_accum[pid]  = 0.0

            # ── Build feature vector (must match train_model.py) ───────
            idle_duration = now - _last_focused[pid]
            focus_time    = _focus_accum.get(pid, 0.0)
            session_age   = now - create_time

            features = [[idle_duration, focus_time, session_age]]

            # ── Predict ────────────────────────────────────────────────
            try:
                prediction = model.predict(features)[0]
            except Exception as e:
                callback(f"[ERROR] Prediction failed for {name}: {e}")
                continue

            # ── Act ────────────────────────────────────────────────────
            if prediction == 1 and status != psutil.STATUS_STOPPED:
                suspend_process(pid, name)
                callback(
                    f"[SUSPEND] {name} (PID {pid}) "
                    f"— idle {idle_duration:.0f}s, focus {focus_time:.0f}s"
                )

            elif prediction == 0 and status == psutil.STATUS_STOPPED:
                # Model now thinks it should be active → resume it
                resume_process(pid)
                callback(f"[RESUME]  {name} (PID {pid}) — back to active")

        # ── Clean up dead PIDs ─────────────────────────────────────────
        live = {p.pid for p in psutil.process_iter()}
        for pid in list(_last_focused):
            if pid not in live:
                _last_focused.pop(pid, None)
                _focus_accum.pop(pid, None)

        time.sleep(INTERVAL)

    callback("[SYSTEM] ML controller stopped.")
