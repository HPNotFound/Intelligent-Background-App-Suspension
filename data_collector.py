import psutil
import time

def collect_process_data(duration=60):
    data = []
    start = time.time()

    while time.time() - start < duration:
        for p in psutil.process_iter(['pid', 'cpu_percent', 'memory_percent', 'create_time']):
            try:
                cpu = p.info['cpu_percent']
                mem = p.info['memory_percent']
                runtime = time.time() - p.info['create_time']

                # Simple labeling logic (used only for training)
                label = 1 if cpu < 1 and mem < 1 else 0  # 1 = inactive

                data.append([cpu, mem, runtime, label])
            except:
                pass

        time.sleep(1)

    return data
