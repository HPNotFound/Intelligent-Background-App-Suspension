import tkinter as tk
from tkinter import ttk
from threading import Thread
from ml_controller import run_ml_controller, stop_ml_controller
from process_manager import resume_process, resume_all, get_suspended

# ---------------------------------------------------------------------------
# Window setup
# ---------------------------------------------------------------------------
root = tk.Tk()
root.title("Intelligent Process Suspension System")
root.geometry("720x520")
root.resizable(True, True)
root.configure(bg="#1a1a2e")

DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
ACCENT    = "#0f3460"
GREEN     = "#00b894"
RED       = "#d63031"
AMBER     = "#fdcb6e"
TEXT      = "#e2e2e2"
MONO      = ("Courier New", 10)
LABEL_FNT = ("Helvetica", 10, "bold")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
header = tk.Frame(root, bg=ACCENT, pady=8)
header.pack(fill=tk.X)
tk.Label(
    header, text="⚡ Intelligent Process Suspension System",
    bg=ACCENT, fg=TEXT, font=("Helvetica", 13, "bold")
).pack()

# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------
status_var = tk.StringVar(value="Status: Idle")
status_bar = tk.Label(
    root, textvariable=status_var,
    bg=PANEL_BG, fg=AMBER, font=LABEL_FNT, anchor="w", padx=10, pady=4
)
status_bar.pack(fill=tk.X)

# ---------------------------------------------------------------------------
# Main content area (log + suspended panel side by side)
# ---------------------------------------------------------------------------
content = tk.Frame(root, bg=DARK_BG)
content.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

# ── Activity log ────────────────────────────────────────────────────────────
log_frame = tk.LabelFrame(
    content, text=" Activity Log ", bg=DARK_BG, fg=TEXT,
    font=LABEL_FNT, bd=1, relief=tk.GROOVE
)
log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

log_box = tk.Listbox(
    log_frame, bg="#0d0d1a", fg=TEXT, font=MONO,
    selectbackground=ACCENT, activestyle="none",
    relief=tk.FLAT, bd=0, highlightthickness=0
)
log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_box.yview)
log_box.configure(yscrollcommand=log_scroll.set)
log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
log_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

# ── Suspended processes panel ────────────────────────────────────────────────
susp_frame = tk.LabelFrame(
    content, text=" Suspended Processes ", bg=DARK_BG, fg=TEXT,
    font=LABEL_FNT, bd=1, relief=tk.GROOVE, width=220
)
susp_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
susp_frame.pack_propagate(False)

susp_list = tk.Listbox(
    susp_frame, bg="#0d0d1a", fg=AMBER, font=MONO,
    selectbackground=ACCENT, activestyle="none",
    relief=tk.FLAT, bd=0, highlightthickness=0
)
susp_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

resume_sel_btn = tk.Button(
    susp_frame, text="Resume Selected",
    bg=ACCENT, fg=TEXT, font=LABEL_FNT, relief=tk.FLAT,
    activebackground=GREEN, cursor="hand2",
    command=lambda: _resume_selected()
)
resume_sel_btn.pack(fill=tk.X, padx=4, pady=(0, 2))

resume_all_btn = tk.Button(
    susp_frame, text="Resume All",
    bg=ACCENT, fg=TEXT, font=LABEL_FNT, relief=tk.FLAT,
    activebackground=GREEN, cursor="hand2",
    command=lambda: _resume_all()
)
resume_all_btn.pack(fill=tk.X, padx=4, pady=(0, 4))

# ---------------------------------------------------------------------------
# Control buttons
# ---------------------------------------------------------------------------
btn_row = tk.Frame(root, bg=DARK_BG)
btn_row.pack(pady=8)

start_btn = tk.Button(
    btn_row, text="▶  Start System",
    bg=GREEN, fg="#000", font=LABEL_FNT, relief=tk.FLAT,
    padx=16, pady=6, cursor="hand2",
    command=lambda: _start()
)
start_btn.pack(side=tk.LEFT, padx=8)

stop_btn = tk.Button(
    btn_row, text="■  Stop System",
    bg=RED, fg=TEXT, font=LABEL_FNT, relief=tk.FLAT,
    padx=16, pady=6, cursor="hand2", state=tk.DISABLED,
    command=lambda: _stop()
)
stop_btn.pack(side=tk.LEFT, padx=8)

clear_btn = tk.Button(
    btn_row, text="🗑  Clear Log",
    bg=ACCENT, fg=TEXT, font=LABEL_FNT, relief=tk.FLAT,
    padx=16, pady=6, cursor="hand2",
    command=lambda: log_box.delete(0, tk.END)
)
clear_btn.pack(side=tk.LEFT, padx=8)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(message: str):
    """Thread-safe log append (called from background thread via root.after)."""
    root.after(0, _log_main, message)

def _log_main(message: str):
    log_box.insert(tk.END, message)
    log_box.see(tk.END)
    # Colour-code entries
    idx = log_box.size() - 1
    if "[SUSPEND]" in message:
        log_box.itemconfig(idx, fg=RED)
    elif "[RESUME]" in message:
        log_box.itemconfig(idx, fg=GREEN)
    elif "[ERROR]" in message:
        log_box.itemconfig(idx, fg=AMBER)


def _refresh_suspended():
    """Periodically refresh the suspended-processes panel."""
    susp_list.delete(0, tk.END)
    for pid, name in get_suspended().items():
        susp_list.insert(tk.END, f"{name}  [{pid}]")
    root.after(2000, _refresh_suspended)   # refresh every 2 s


def _resume_selected():
    sel = susp_list.curselection()
    if not sel:
        return
    entry = susp_list.get(sel[0])          # e.g. "firefox  [12345]"
    try:
        pid = int(entry.split("[")[-1].rstrip("]"))
        if resume_process(pid):
            log(f"[RESUME]  PID {pid} — manually resumed")
    except (ValueError, IndexError):
        pass


def _resume_all():
    pids = resume_all()
    log(f"[RESUME]  Resumed {len(pids)} process(es) manually")


def _start():
    start_btn.config(state=tk.DISABLED)
    stop_btn.config(state=tk.NORMAL)
    status_var.set("Status: Running")
    log("[SYSTEM] ML controller starting…")
    t = Thread(target=run_ml_controller, args=(log,), daemon=True)
    t.start()


def _stop():
    stop_ml_controller()
    start_btn.config(state=tk.NORMAL)
    stop_btn.config(state=tk.DISABLED)
    status_var.set("Status: Stopped")

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_refresh_suspended()   # start the suspended-list refresh loop
root.mainloop()
