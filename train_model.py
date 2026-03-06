import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from data_collector import collect_process_data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COLLECTION_DURATION = 120   # seconds to observe processes (increase for richer data)
COLLECTION_INTERVAL = 1.0   # sampling interval in seconds
MODEL_PATH           = "process_classifier.pkl"

FEATURE_NAMES = ["idle_duration", "focus_time", "session_age"]
#                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#  idle_duration  – seconds since the process last held window focus
#  focus_time     – cumulative seconds the process was the focused window
#  session_age    – seconds since the process was created (context)

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
print(f"Collecting behaviour data for {COLLECTION_DURATION}s "
      f"(sampling every {COLLECTION_INTERVAL}s)…")

raw = collect_process_data(
    duration=COLLECTION_DURATION,
    interval=COLLECTION_INTERVAL,
)

if len(raw) == 0:
    raise RuntimeError("No data was collected. Is psutil able to read processes?")

data = np.array(raw, dtype=float)

X = data[:, :3]   # idle_duration | focus_time | session_age
y = data[:,  3]   # 0 = active, 1 = suspend candidate

print(f"Collected {len(X)} samples  |  "
      f"suspend={int(y.sum())}  active={int((y==0).sum())}")

# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 1 else None
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight="balanced",   # handles imbalanced active/idle ratio
    random_state=42,
)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)
print("\nValidation report:")
print(classification_report(y_test, y_pred, target_names=["active", "suspend"]))

importances = model.feature_importances_
print("Feature importances:")
for name, imp in zip(FEATURE_NAMES, importances):
    print(f"  {name:<16} {imp:.4f}")

# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to '{MODEL_PATH}'")
