import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from data_collector import collect_process_data

print("Collecting real-time process data...")
data = collect_process_data()

data = np.array(data)
X = data[:, :3]   # cpu, memory, runtime
y = data[:, 3]    # active / inactive

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "process_classifier.pkl")
print("ML model trained and saved")
