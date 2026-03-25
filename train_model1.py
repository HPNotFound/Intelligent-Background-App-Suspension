"""
train_model.py
==============
Intelligent Background Application Suspension — Model Trainer

Design goals
------------
- Lightweight / non-intrusive  : runs at IDLE OS scheduler priority (nice=19)
- Learned idle threshold        : per-app median idle period computed from
                                  collected data; no hardcoded magic number
- Safe process exclusion        : bash, sh, zsh and system daemons are never
                                  labelled as suspend candidates
- Transparent output            : prints which apps the trained model would
                                  suspend right now so you can verify before
                                  deploying
"""

import os
import sys
import time
import numpy as np
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from data_collector import collect_process_data

# ── Step 0 : drop to idle OS priority immediately ───────────────────────────
# nice=19 → kernel schedules us only when nothing else wants CPU.
try:
    os.nice(19)
except PermissionError:
    pass

# ── Configuration ────────────────────────────────────────────────────────────
COLLECTION_DURATION = 120    # seconds of observation (raise to 300+ for richer data)
COLLECTION_INTERVAL = 1.0    # sampling cadence in seconds
MODEL_PATH          = "process_classifier.pkl"
THRESHOLD_PATH      = "idle_thresholds.pkl"   # learned threshold persisted here

FEATURE_NAMES = ["idle_duration", "focus_time", "session_age"]

# Processes that must NEVER be suspended
SAFE_NAMES = {
    "bash", "sh", "zsh", "fish", "dash", "ksh", "tcsh", "csh",   # shells
    "systemd", "init", "login", "sshd", "dbus-daemon",            # session/init
    "gnome-session", "gnome-shell", "Xorg", "X", "Xwayland",      # desktop
    "python3", "python",                                           # this trainer
}

# ── Step 1 : collect live behaviour data ────────────────────────────────────
print(f"[trainer] OS priority  : nice=19 (idle — will not burden foreground work)")
print(f"[trainer] Collecting behaviour data for {COLLECTION_DURATION}s …")
print(f"[trainer] Use your computer normally during this window.\n")

raw = collect_process_data(
    duration=COLLECTION_DURATION,
    interval=COLLECTION_INTERVAL,
)

if not raw:
    sys.exit("[trainer] ERROR: No data collected. Check psutil permissions.")

data = np.array(raw, dtype=float)
X_raw = data[:, :3]   # idle_duration | focus_time | session_age
y_raw = data[:,  3]   # heuristic label from data_collector

print(f"[trainer] Raw samples collected : {len(X_raw)}")
print(f"[trainer]   heuristic suspend   : {int(y_raw.sum())}")
print(f"[trainer]   heuristic active    : {int((y_raw == 0).sum())}\n")

if y_raw.sum() == 0:
    print("[trainer] WARNING: zero suspend-candidate samples.")
    print("[trainer] Observation window may be too short. Try COLLECTION_DURATION=300.\n")

# ── Step 2 : learn idle threshold from data (replaces hardcoded 30 s) ───────
# Compute median idle_duration of processes that were clearly active (y=0).
# Anything idle longer than 2× that median with zero focus time is a candidate.
active_idle = X_raw[y_raw == 0, 0]
if len(active_idle) > 0:
    learned_threshold = float(np.median(active_idle)) * 2.0
    learned_threshold = max(15.0,  learned_threshold)   # floor  : 15 s
    learned_threshold = min(300.0, learned_threshold)   # ceiling: 5 min
else:
    learned_threshold = 30.0   # safe fallback

print(f"[trainer] Learned idle threshold : {learned_threshold:.1f}s")
print(f"[trainer] Re-labelling with learned threshold …\n")

# Re-label: idle >= threshold AND zero focus time → suspend candidate
y = ((X_raw[:, 0] >= learned_threshold) & (X_raw[:, 1] == 0)).astype(float)
X = X_raw

print(f"[trainer] Final label distribution:")
print(f"[trainer]   suspend : {int(y.sum())}")
print(f"[trainer]   active  : {int((y == 0).sum())}\n")

# Persist learned threshold so ml_controller.py can load it at runtime
joblib.dump({"idle_threshold": learned_threshold}, THRESHOLD_PATH)
print(f"[trainer] Learned threshold saved → '{THRESHOLD_PATH}'\n")

# ── Step 3 : train / validation split ───────────────────────────────────────
can_stratify = int(y.sum()) >= 2 and int((y == 0).sum()) >= 2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if can_stratify else None,
)

# ── Step 4 : train RandomForest (single core to stay lightweight) ────────────
print("[trainer] Training model …")
t0 = time.time()

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_leaf=3,       # avoids over-fitting on small windows
    class_weight="balanced",  # handles active >> suspend imbalance
    n_jobs=1,                 # single core — non-intrusive
    random_state=42,
)
model.fit(X_train, y_train)
print(f"[trainer] Training complete in {time.time() - t0:.1f}s\n")

# ── Step 5 : evaluate ────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print("Validation report:")
print(classification_report(
    y_test, y_pred,
    target_names=["active", "suspend"],
    zero_division=0,
))

print("Feature importances:")
for name, imp in zip(FEATURE_NAMES, model.feature_importances_):
    bar = "█" * int(imp * 40)
    print(f"  {name:<16} {imp:.4f}  {bar}")
print()

# ── Step 6 : live snapshot — show what WOULD be suspended right now ──────────
print("=" * 62)
print("  Apps the model would suspend RIGHT NOW (live snapshot)")
print("=" * 62)

now = time.time()
suspend_list = []

for p in psutil.process_iter(["pid", "name", "create_time", "status", "uids"]):
    try:
        name        = (p.info["name"] or "unknown").lower()
        pid         = p.info["pid"]
        create_time = p.info["create_time"]
        status      = p.info["status"]
        uid         = p.info["uids"].real
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

    # Skip safe names, system PIDs, and other users' processes
    if name in SAFE_NAMES or pid < 100 or uid != os.getuid():
        continue

    session_age   = now - create_time
    idle_duration = session_age   # conservative: treat as never focused
    focus_time    = 0.0

    pred = model.predict([[idle_duration, focus_time, session_age]])[0]
    if pred == 1 and status != psutil.STATUS_STOPPED:
        suspend_list.append((p.info["name"] or name, pid, idle_duration))

if suspend_list:
    print(f"\n  {'Process':<26} {'PID':<8} {'Idle (s)'}")
    print(f"  {'-------':<26} {'---':<8} {'--------'}")
    for sname, spid, idle in sorted(suspend_list, key=lambda x: -x[2]):
        print(f"  {sname:<26} {spid:<8} {idle:.0f}s")
    print(f"\n  Total : {len(suspend_list)} process(es) flagged for suspension")
else:
    print("\n  (none — all user processes appear active in this snapshot)")

print(f"\n  Excluded from suspension: bash, sh, zsh + system daemons")
print(f"  Learned idle threshold  : {learned_threshold:.1f}s")
print("=" * 62)

# ── Step 7 : save model ──────────────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)
print(f"\n[trainer] Model saved → '{MODEL_PATH}'")
print(f"[trainer] Done.\n")
