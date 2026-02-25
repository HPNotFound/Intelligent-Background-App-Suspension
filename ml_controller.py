import psutil
import time
import joblib
from process_manager import suspend_process

model = joblib.load("process_classifier.pkl")

def run_ml_controller(callback):
    while True:
        for p in psutil.process_iter(['pid', 'cpu_percent', 'memory_percent', 'create_time']):
            try:
                cpu = p.cpu_percent(interval=0.1)
                mem = p.memory_percent()
                runtime = time.time() - p.create_time()

                prediction = model.predict([[cpu, mem, runtime]])

                if prediction[0] == 1 and p.pid > 1000:
                    suspend_process(p.pid)
                    callback(f"Suspended {p.name()} (PID {p.pid})")
            except:
                pass

        time.sleep(5)
