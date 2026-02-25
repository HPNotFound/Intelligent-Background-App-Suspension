import os
import signal
import psutil

suspended = {}

def suspend_process(pid):
    try:
        os.kill(pid, signal.SIGSTOP)
        suspended[pid] = psutil.Process(pid).name()
    except:
        pass

def resume_all():
    for pid in list(suspended.keys()):
        try:
            os.kill(pid, signal.SIGCONT)
            suspended.pop(pid)
        except:
            pass
