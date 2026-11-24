"""Native Windows console for launching and monitoring public jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from .public_jobs import list_jobs, request_stop
from .public_runs import build_plan, launch


def open_console(jobs):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Fly Touhou | Original-game jobs")
    root.minsize(910, 570)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    form = ttk.Frame(root, padding=14)
    form.grid(sticky="ew")
    form.columnconfigure(1, weight=1)
    form.columnconfigure(3, weight=1)
    choices = {
        "operation": ("train", "evaluate", "fit-attack"),
        "game": ("th14", "th08"),
        "objective": ("nbnm", "clear"),
        "difficulty": ("lunatic", "hard", "normal", "easy"),
        "train_scope": ("joint", "encoder-heads"),
    }
    values = {key: tk.StringVar(value=options[0]) for key, options in choices.items()}
    fields = ("executable", "checkpoint", "connectome", "spec", "output", "resume",
              "guide", "workers", "decisions", "seed")
    values.update({key: tk.StringVar(value="8" if key == "workers" else "") for key in fields})
    labels = {
        "operation": "Job", "game": "Original", "objective": "Objective", "difficulty": "Difficulty",
        "train_scope": "Train scope", "executable": "Original EXE", "checkpoint": "Parameters",
        "connectome": "Connectome", "spec": "Policy spec", "output": "Output directory", "resume": "Resume state",
        "guide": "Verified guide", "workers": "Workers", "decisions": "Decisions", "seed": "Seed",
    }
    for index, key in enumerate((*choices, *fields)):
        row, pair = divmod(index, 2)
        column = pair * 2
        ttk.Label(form, text=labels[key]).grid(row=row, column=column, padx=(0, 8), pady=4, sticky="w")
        box = (ttk.Combobox(form, values=choices[key], state="readonly", textvariable=values[key])
               if key in choices else ttk.Entry(form, textvariable=values[key]))
        box.grid(row=row, column=column + 1, padx=(0, 16), pady=4, sticky="ew")
        if key in ("executable", "checkpoint", "connectome", "spec", "output", "resume", "guide"):
            def browse(field=key):
                if field in ("connectome", "output", "guide"):
                    selected = filedialog.askdirectory(parent=root)
                else:
                    selected = filedialog.askopenfilename(parent=root)
                if selected:
                    values[field].set(selected)
            box.bind("<Double-1>", lambda _event, choose=browse: choose())

    note = tk.StringVar()
    show_finished = tk.BooleanVar()
    buttons = ttk.Frame(form)
    buttons.grid(row=8, columnspan=4, sticky="ew", pady=(10, 0))
    table = ttk.Frame(root, padding=(14, 0, 14, 8))
    table.grid(row=1, column=0, sticky="nsew")
    table.columnconfigure(0, weight=1)
    table.rowconfigure(0, weight=1)
    columns = ("operation", "game", "objective", "status", "phase", "stage", "frame", "decisions", "updates")
    listing = ttk.Treeview(table, columns=columns, show="headings", height=9)
    for key in columns:
        listing.heading(key, text=key.title())
        listing.column(key, width=85 if key != "phase" else 155, anchor="center")
    listing.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(table, orient="vertical", command=listing.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    listing.configure(yscrollcommand=scrollbar.set)
    ttk.Label(root, textvariable=note, padding=(14, 0, 14, 10), wraplength=870).grid(row=2, column=0, sticky="w")
    current = {}

    def selection():
        return current.get(next(iter(listing.selection()), ""))

    def perform(function):
        try:
            function()
        except (OSError, ValueError, KeyError) as error:
            messagebox.showerror("Fly Touhou", str(error), parent=root)

    def start():
        options = {key: variable.get().strip() or None for key, variable in values.items()}
        for key in ("workers", "decisions", "seed"):
            if options[key] is not None:
                options[key] = int(options[key])
        if options["operation"] != "train":
            options.update(resume=None, workers=None, decisions=None)
        options.update(jobs=jobs, d3d8to9=None, profile=None, environment=None, segment=None,
                       course=None, trainable_heads=None, fit_steps=640, record_observations=False)
        path = launch(build_plan(SimpleNamespace(**options)))
        note.set(f"Dispatched {path.parent.name}")

    def stop():
        row = selection()
        if row and messagebox.askyesno("Stop job", "Save and stop at the next supported boundary?", parent=root):
            target = request_stop(Path(row["job"]))
            note.set(f"Stop requested for {target.parent.name}")

    def open_output():
        row = selection()
        if row:
            output = Path(row["output"])
            os.startfile(output if output.exists() else Path(row["job"]).parent)

    def details(_event=None):
        row = selection()
        if row:
            note.set(json.dumps({key: row[key] for key in ("output", "miss", "bomb", "contact", "error")
                                 if row.get(key) is not None}, ensure_ascii=False))

    def refresh():
        try:
            rows = list_jobs(jobs, active_only=not show_finished.get())
            displayed = set(listing.get_children())
            current.clear()
            for row in rows:
                key = row["job"]
                current[key] = row
                row_values = tuple(row[k] if row[k] is not None else "" for k in columns)
                if key in displayed:
                    listing.item(key, values=row_values)
                else:
                    listing.insert("", "end", iid=key, values=row_values)
            for key in displayed - current.keys():
                listing.delete(key)
        except (OSError, ValueError, KeyError) as error:
            note.set(str(error))
        root.after(1500, refresh)

    ttk.Button(buttons, text="Start job", command=lambda: perform(start)).pack(side="left", padx=(0, 6))
    ttk.Button(buttons, text="Stop at boundary", command=lambda: perform(stop)).pack(side="left", padx=6)
    ttk.Button(buttons, text="Open output", command=lambda: perform(open_output)).pack(side="left", padx=6)
    ttk.Checkbutton(buttons, text="Show completed jobs", variable=show_finished).pack(side="right")
    listing.bind("<<TreeviewSelect>>", details)
    refresh()
    root.mainloop()
    return 0
