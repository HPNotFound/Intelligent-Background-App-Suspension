import tkinter as tk
from threading import Thread
from ml_controller import run_ml_controller

root = tk.Tk()
root.title("Intelligent Process Suspension System")
root.geometry("600x400")

log_box = tk.Listbox(root, width=80, height=20)
log_box.pack(pady=10)

def log(message):
    log_box.insert(tk.END, message)

def start_system():
    t = Thread(target=run_ml_controller, args=(log,), daemon=True)
    t.start()
    log("System initialized...")

start_btn = tk.Button(root, text="Start / Initialize", command=start_system)
start_btn.pack(pady=10)

root.mainloop()
