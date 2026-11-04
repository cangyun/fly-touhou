"""Small native entry to the same demonstration tab used by the main panel."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    import tkinter as tk
    from tkinter import ttk
    from fly_touhou.thrl_demo_ui import DemoTab
    root=tk.Tk();root.title('东方原作 · 神经网络实机演示')
    root.geometry('1120x720');root.minsize(900,600)
    notebook=ttk.Notebook(root);notebook.pack(fill='both',expand=True,padx=10,pady=10)
    demo=DemoTab(notebook,ROOT/'artifacts/engineering/real-model-demonstration-v1')
    root.mainloop()
