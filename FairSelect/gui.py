import os
import sys
import traceback
from typing import Dict, Tuple, Any, List
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from .deps import SKLEARN_OK, AIF360_OK, IMBLEARN_OK, FAIRLEARN_OK
from .params import PARAM_SPECS, AVAILABLE_MODELS
from .core import split_data, RunResult
from .techniques_pre import run_reweighting, run_smote_or_ros, run_local_massaging
from .techniques_in import (
    run_baseline, run_compositional_models, run_group_balanced_ensemble,
    run_multicalibration, run_reductions_meta, run_prejudice_remover,
)
from .techniques_post import (
    run_group_youden_postproc, run_multiaccuracy_boost,
    run_reject_option_shift, run_input_repair,
)
from .techniques_combined import run_combined_pipeline
from .utils import coerce_value, eval_tuple

"""
This file contains the code to generate the GUI for the tool
"""

#-------GUI---------
class FairnessToolGUI(tk.Tk):
    '''
    Defines the actual user interface window and its behavior.
    '''
    def __init__(self):
        super().__init__()
        self.title("Fairness Tool – Assess Different Techniques for Optimal Fairness")
        self.minsize(1000, 800)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.resizable(True, True)

        self.csv_path = tk.StringVar()
        self.selected_model = tk.StringVar()
        self.param_widgets: Dict[str, Tuple[tk.Widget, Dict[str, Any]]] = {}
        self.status_text = tk.StringVar()
        self.columns: List[str] = []

        self.target_var = tk.StringVar()
        self.features_listbox: tk.Listbox = None  # type: ignore
        self.protected_listbox: tk.Listbox = None  # type: ignore

        ttk.Label(self, text="Fairness Technique Assessment Tool", font=("Segoe UI", 16, "bold")).pack(pady=10)

        self._build_file_picker()
        self._build_variable_selector()
        self._build_model_chooser()

        self.params_frame = ttk.LabelFrame(self, text="Model Parameters")
        self.params_frame.pack(fill="both", expand=False, padx=12, pady=8)

        self._build_technique_selector()
        self._build_bottom_buttons()

        self.status_text.set("Select CSV → choose variables → select model & params → pick techniques → Run.")

    # ---------- File + Variables ----------
    def _build_file_picker(self):
        lf = ttk.LabelFrame(self, text="Dataset (CSV)")
        lf.pack(fill="x", padx=12, pady=6)
        row = ttk.Frame(lf); row.pack(fill="x", padx=8, pady=6)
        ttk.Label(row, text="File:").pack(side="left")
        ttk.Entry(row, textvariable=self.csv_path, width=90).pack(side="left", padx=8, fill="x", expand=True)
        ttk.Button(row, text="Browse…", command=self.on_browse_csv).pack(side="left")
        ttk.Button(row, text="Refresh Columns", command=self.load_columns).pack(side="left", padx=6)

    def on_browse_csv(self):
        path = filedialog.askopenfilename(title="Choose dataset (CSV)", filetypes=[("CSV Files", "*.csv")])
        if path and path.lower().endswith(".csv"):
            self.csv_path.set(path); self.load_columns()
        elif path:
            messagebox.showerror("Invalid file", "Please select a .csv file.")

    def load_columns(self):
        try:
            df_head = pd.read_csv(self.csv_path.get().strip(), nrows=0)
            self.columns = list(df_head.columns)
            self._populate_variable_widgets()
            self.status_text.set(f"Found {len(self.columns)} columns.")
        except Exception as e:
            self.columns = []
            self._populate_variable_widgets()
            self.status_text.set(f"\U0001F4A5 Could not read columns: {e}")
            messagebox.showerror("Header Read Error", f"{e}")

    def _build_variable_selector(self):
        lf = ttk.LabelFrame(self, text="Variables")
        lf.pack(fill="x", padx=12, pady=6)

        r1 = ttk.Frame(lf); r1.pack(fill="x", padx=8, pady=6)
        ttk.Label(r1, text="Target:").pack(side="left")
        self.target_combo = ttk.Combobox(r1, textvariable=self.target_var, values=[], state="readonly", width=40)
        self.target_combo.pack(side="left", padx=8)

        body = ttk.Frame(lf); body.pack(fill="both", padx=8, pady=(0,8))

        left = ttk.Frame(body); left.pack(side="left", fill="both", expand=True, padx=(0,8))
        ttk.Label(left, text="Protected characteristics (1+):").pack(anchor="w")
        pcont = ttk.Frame(left); pcont.pack(fill="both", expand=True, pady=4)
        self.prot_scroll_y = ttk.Scrollbar(pcont, orient="vertical")
        self.protected_listbox = tk.Listbox(pcont, selectmode="extended", exportselection=False)
        self.protected_listbox.pack(side="left", fill="both", expand=True)
        self.prot_scroll_y.config(command=self.protected_listbox.yview); self.prot_scroll_y.pack(side="right", fill="y")
        self.protected_listbox.config(yscrollcommand=self.prot_scroll_y.set, height=10)
        ttk.Button(left, text="Clear", command=lambda: self._clear_listbox(self.protected_listbox)).pack(anchor="w")

        right = ttk.Frame(body); right.pack(side="left", fill="both", expand=True, padx=(8,0))
        ttk.Label(right, text="Features (1+):").pack(anchor="w")
        fcont = ttk.Frame(right); fcont.pack(fill="both", expand=True, pady=4)
        self.feat_scroll_y = ttk.Scrollbar(fcont, orient="vertical")
        self.features_listbox = tk.Listbox(fcont, selectmode="extended", exportselection=False)
        self.features_listbox.pack(side="left", fill="both", expand=True)
        self.feat_scroll_y.config(command=self.features_listbox.yview); self.feat_scroll_y.pack(side="right", fill="y")
        self.features_listbox.config(yscrollcommand=self.feat_scroll_y.set, height=10)
        b = ttk.Frame(right); b.pack(fill="x")
        ttk.Button(b, text="Select All (except target/protected)", command=self.auto_select_features).pack(side="left")
        ttk.Button(b, text="Clear", command=lambda: self._clear_listbox(self.features_listbox)).pack(side="left", padx=6)

    def _populate_variable_widgets(self):
        cols = self.columns or []
        self.target_combo["values"] = cols
        if cols:
            if self.target_var.get() not in cols:
                self.target_var.set(cols[0])
        else:
            self.target_var.set("")
        self._refill_listbox(self.protected_listbox, cols)
        self._refill_listbox(self.features_listbox, cols)

    def _refill_listbox(self, lb: tk.Listbox, items):
        lb.delete(0, tk.END)
        for c in items:
            lb.insert(tk.END, c)

    def _clear_listbox(self, lb: tk.Listbox):
        lb.selection_clear(0, tk.END)

    def auto_select_features(self):
        tgt = self.target_var.get()
        prot = set(self._get_selected(self.protected_listbox))
        for i, c in enumerate(self.columns):
            if c==tgt or c in prot:
                self.features_listbox.selection_clear(i)
            else:
                self.features_listbox.selection_set(i)

    def _get_selected(self, lb: tk.Listbox):
        return [lb.get(i) for i in lb.curselection()]

    # ---------- Model + Params ----------
    def _build_model_chooser(self):
        lf = ttk.LabelFrame(self, text="Model")
        lf.pack(fill="x", padx=12, pady=6)
        r = ttk.Frame(lf); r.pack(fill="x", padx=8, pady=6)
        ttk.Label(r, text="Type:").pack(side="left")
        vals = AVAILABLE_MODELS if AVAILABLE_MODELS else ["(no models)"]
        self.model_combo = ttk.Combobox(r, textvariable=self.selected_model, values=vals, state="readonly", width=40)
        if AVAILABLE_MODELS:
            self.model_combo.current(0)
        self.model_combo.pack(side="left", padx=8)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)
        ttk.Button(r, text="Configure Parameters", command=self.render_param_form).pack(side="left", padx=6)

    def on_model_change(self, event=None):
        self.render_param_form()

    def clear_params_frame(self):
        for c in self.params_frame.winfo_children():
            c.destroy()
        self.param_widgets.clear()

    def render_param_form(self):
        self.clear_params_frame()
        mname = self.selected_model.get()
        specs = PARAM_SPECS.get(mname, [])
        if not specs:
            ttk.Label(self.params_frame, text="No parameters for this model.").pack(padx=8, pady=8)
            return
        grid = ttk.Frame(self.params_frame); grid.pack(fill="both", expand=True, padx=8, pady=8)
        ttk.Label(grid, text="Parameter", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(grid, text="Value",     font=("Segoe UI", 10, "bold")).grid(row=0, column=1, sticky="w")
        ttk.Label(grid, text="Help",      font=("Segoe UI", 10, "bold")).grid(row=0, column=2, sticky="w")

        for r, spec in enumerate(specs, start=1):
            name = spec["name"]; required=spec["required"]; default=spec["default"]; ptype=spec["type"]; choices=spec.get("choices",[])
            ttk.Label(grid, text=f"{name}{' *' if required else ''}").grid(row=r, column=0, sticky="w", padx=4, pady=4)
            if ptype == bool:
                var = tk.BooleanVar(value=bool(default) if default is not None else False)
                w = ttk.Checkbutton(grid, variable=var); w.var=var
                w.grid(row=r, column=1, sticky="w", padx=4, pady=4)
            elif ptype == "choice":
                var = tk.StringVar(value=str(default) if default is not None else "")
                w = ttk.Combobox(grid, textvariable=var, values=choices, state="readonly", width=20); w.var=var; w.choices=choices
                if default in choices: w.set(str(default))
                w.grid(row=r, column=1, sticky="w", padx=4, pady=4)
            else:
                var = tk.StringVar(value="" if default is None else str(default))
                w = ttk.Entry(grid, textvariable=var, width=24); w.var=var
                w.grid(row=r, column=1, sticky="w", padx=4, pady=4)
            ttk.Label(grid, text=spec["help"], foreground="#555").grid(row=r, column=2, sticky="w", padx=4, pady=4)
            self.param_widgets[name] = (w, spec)
        ttk.Label(self.params_frame, text="* Required fields. Leave blank to use library defaults for optional parameters.",
                  foreground="#555").pack(anchor="w", padx=8, pady=(0,8))

    # ---------- Technique selection & Run ----------
    def _build_technique_selector(self):
        lf = ttk.LabelFrame(self, text="Techniques to Run")
        lf.pack(fill="x", padx=12, pady=6)

        self.tech_vars: Dict[str, tk.BooleanVar] = {}

        # Pre
        pre_row = ttk.Frame(lf); pre_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(pre_row, text="Pre-processing:", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0,8))
        for name in ["Reweight (y,a)", "SMOTE / Oversample", "Local Massaging"]:
            v = tk.BooleanVar(value=False); self.tech_vars[f"Pre:{name}"]=v
            ttk.Checkbutton(pre_row, text=name, variable=v).pack(side="left", padx=6)

        # In
        in_row = ttk.Frame(lf); in_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(in_row, text="In-processing:", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0,8))
        for name in ["Compositional per-group", "Ensemble (K=5)", "Multicalibration (isotonic)", "Reductions (EO)", "Fairness Regularization (Prejudice Remover)"]:
            v = tk.BooleanVar(value=False); self.tech_vars[f"In:{name}"]=v
            cb = ttk.Checkbutton(in_row, text=name, variable=v)
            cb.pack(side="left", padx=6)
            # disable if AIF360 not available
            if name == "Fairness Regularization (Prejudice Remover)" and not AIF360_OK:
                cb.state(["disabled"])

        # Post
        post_row = ttk.Frame(lf); post_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(post_row, text="Post-processing:", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0,8))
        for name in ["Youden per group", "Multiaccuracy Boost", "Reject-Option Shift", "Input Repair"]:
            v = tk.BooleanVar(value=False); self.tech_vars[f"Post:{name}"]=v
            ttk.Checkbutton(post_row, text=name, variable=v).pack(side="left", padx=6)

        # Baseline toggle
        base_row = ttk.Frame(lf); base_row.pack(fill="x", padx=8, pady=4)
        self.base_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(base_row, text="Include Baseline", variable=self.base_var).pack(side="left")

        # Combined pipeline toggle
        combo_row = ttk.Frame(lf); combo_row.pack(fill="x", padx=8, pady=4)
        self.combo_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(combo_row, text="Run Combined Pipeline of selected techniques", variable=self.combo_var).pack(side="left")
        ttk.Label(combo_row, text="(Applies Pre ➜ In ➜ Post, on the same model)", foreground="#555").pack(side="left", padx=8)


        # Note about optional deps
        note = []
        if not IMBLEARN_OK: note.append("SMOTE→fallback (RandomOverSampler)")
        if not FAIRLEARN_OK: note.append("Reductions disabled (install fairlearn)")
        if not AIF360_OK: note.append("Prejudice Remover disabled (install aif360)")
        if note:
            ttk.Label(lf, text="Note: " + "; ".join(note), foreground="#a33").pack(anchor="w", padx=8, pady=(4,0))


    def _build_bottom_buttons(self):
        bar = ttk.Frame(self); bar.pack(fill="x", padx=12, pady=10)
        ttk.Button(bar, text="Run Experiments", command=self.on_run).pack(side="left")
        ttk.Button(bar, text="Quit", command=self.destroy).pack(side="right")
        ttk.Label(self, textvariable=self.status_text, foreground="#444", anchor="w").pack(fill="x", padx=16, pady=(0, 8))

    def collect_params(self) -> Dict[str, Any]:
        params = {}
        for name, (widget, spec) in self.param_widgets.items():
            ptype = spec["type"]; choices = spec.get("choices", [])
            if ptype == bool:
                raw = widget.var.get()
            elif ptype == "choice":
                raw = widget.var.get().strip()
            else:
                raw = widget.var.get().strip()
            if self.selected_model.get() == "Neural Network" and name == "hidden_layer_sizes":
                coerced = None if raw.strip()=="" else eval_tuple(raw)
            else:
                coerced = coerce_value(ptype, raw, choices=choices)
                if ptype == "choice" and raw == "None":
                    coerced = None
            params[name] = coerced
        return params

    def on_run(self):
        try:
            # Validate config
            path = self.csv_path.get().strip()
            if not path or not os.path.isfile(path) or not path.lower().endswith(".csv"):
                raise ValueError("Please choose a valid CSV file.")
            df = pd.read_csv(path)

            target = self.target_var.get().strip()
            if not target:
                raise ValueError("Choose a target column.")
            protected = self._get_selected(self.protected_listbox)
            if len(protected)<1: raise ValueError("Select at least one protected characteristic.")
            features = self._get_selected(self.features_listbox)
            features = [f for f in features if f != target and f not in protected]
            if len(features)<1: raise ValueError("Select at least one feature (not including target/protected).")

            model_name = self.selected_model.get().strip()
            if not model_name: raise ValueError("Choose a model type.")
            params = self.collect_params()

            # Prepare data & splits
            X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te = split_data(
                df[[*features, *protected, target]], target, protected, features, test_size=0.25, val_size=0.2, random_state=42
            )
            # Keep full train (for some repairs)
            all_df_train = pd.concat([X_tr, X_va], axis=0)

            # Run baseline + techniques
            results: List[RunResult] = []
            if self.base_var.get():
                results.append(run_baseline(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train))

            # Pre
            if self.tech_vars["Pre:Reweight (y,a)"].get():
                results.append(run_reweighting(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train))
            if self.tech_vars["Pre:SMOTE / Oversample"].get():
                results.append(run_smote_or_ros(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train))
            if self.tech_vars["Pre:Local Massaging"].get():
                results.append(run_local_massaging(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train))

            # In
            if self.tech_vars["In:Compositional per-group"].get():
                results.append(run_compositional_models(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train))
            if self.tech_vars["In:Ensemble (K=5)"].get():
                results.append(run_group_balanced_ensemble(model_name, params, 5, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train))
            if self.tech_vars["In:Multicalibration (isotonic)"].get():
                results.append(run_multicalibration(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train))
            if self.tech_vars["In:Reductions (EO)"].get():
                results.append(run_reductions_meta(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train, constraint="EO"))
            if self.tech_vars["In:Fairness Regularization (Prejudice Remover)"].get():
                results.append(run_prejudice_remover(
                    model_name, params,
                    X_tr, X_va, X_te, y_tr, y_va, y_te,
                    A_tr, A_va, A_te,
                    protected, all_df_train,
                    eta=25.0  # tweakable strength of the regularizer
                ))


            # Post
            if self.tech_vars["Post:Youden per group"].get():
                results.append(run_group_youden_postproc(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train))
            if self.tech_vars["Post:Multiaccuracy Boost"].get():
                results.append(run_multiaccuracy_boost(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train))
            if self.tech_vars["Post:Reject-Option Shift"].get():
                results.append(run_reject_option_shift(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train))
            if self.tech_vars["Post:Input Repair"].get():
                results.append(run_input_repair(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train))

            #After individual technique runs, run combined pipeline if selected
            if self.combo_var.get():
                #Build a simple “selected” dict with the same keys used in the technique selector
                selected = {k: v.get() for k, v in self.tech_vars.items()}
                try:
                    combined_rr = run_combined_pipeline(
                        model_name, params,
                        X_tr, X_va, X_te, y_tr, y_va, y_te,
                        A_tr, A_va, A_te, protected, all_df_train,
                        selected
                    )
                    results.append(combined_rr)
                except Exception as combo_err:
                    #Dont crash the session if a combo is incompatible
                    print("[Combined pipeline] Failed:", combo_err, file=sys.stderr)


            if len(results)==0:
                raise ValueError("No techniques selected. Include Baseline and/or choose at least one technique.")

            self.show_dashboard(results)

        except Exception as e:
            tb = traceback.format_exc(limit=1)
            self.status_text.set(f"\U0001F4A5 {e}")
            messagebox.showerror("Run Error", f"{e}\n\nDetails:\n{tb}")

    # ---------- Dashboard ----------
    def show_dashboard(self, results: List[RunResult]):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image, ImageTk

        # Small helpers
        def _fmt(x):
            return "NA" if pd.isna(x) else f"{x:.3f}"

        def _fmt_delta(curr, base, *, invert=False):
            if pd.isna(curr) or pd.isna(base):
                return "NA"
            d = (curr - base)
            if invert:
                d = -d
            return f"{d:+.3f}"

        win = tk.Toplevel(self)
        win.title("Results Dashboard")
        win.minsize(1100, 800)
        win.rowconfigure(0, weight=1)
        win.columnconfigure(0, weight=1)

        nb = ttk.Notebook(win)
        nb.grid(row=0, column=0, sticky="nsew")

        # ------------- Overview -------------
        overview = ttk.Frame(nb)
        nb.add(overview, text="Overview")
        overview.rowconfigure(0, weight=1)
        overview.columnconfigure(0, weight=1)

        baseline = next((r for r in results if r.name.lower().startswith("baseline")), None)
        base = baseline.overall if baseline is not None else {}

        cols = [
            "Technique",
            "ACC", "ACCΔ",
            "AUROC", "AUROCΔ",
            "AUPRC", "AUPRCΔ",
            "F1", "F1Δ",
            "Brier", "BrierΔ",
            "ECE", "ECEΔ",
            "PPR_diff", "PPRΔ",
            "DP_diff", "DPΔ",
            "TPR_diff", "TPRΔ",
            "EOp_diff", "EOpΔ",
            "FPR_diff", "FPRΔ",
            "EO_diff", "EOΔ",
        ]

        ov_frame = ttk.Frame(overview)
        ov_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        ov_frame.rowconfigure(0, weight=1)
        ov_frame.columnconfigure(0, weight=1)

        ov_tree = ttk.Treeview(ov_frame, columns=cols, show="headings")
        vsb = ttk.Scrollbar(ov_frame, orient="vertical", command=ov_tree.yview)
        hsb = ttk.Scrollbar(ov_frame, orient="horizontal", command=ov_tree.xview)
        ov_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        ov_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        for c in cols:
            ov_tree.heading(c, text=c)

        ov_tree.column("Technique", width=360, anchor="w")
        for c in cols[1:]:
            ov_tree.column(c, width=110, anchor="center")

        def _b(k):
            return float(base.get(k, np.nan)) if base else np.nan

        for rr in results:
            ov = {k: float(v) if v is not None else np.nan for k, v in rr.overall.items()}
            row = [
                rr.name,
                _fmt(ov.get("ACC", np.nan)),      _fmt_delta(ov.get("ACC", np.nan),      _b("ACC")),
                _fmt(ov.get("AUROC", np.nan)),    _fmt_delta(ov.get("AUROC", np.nan),    _b("AUROC")),
                _fmt(ov.get("AUPRC", np.nan)),    _fmt_delta(ov.get("AUPRC", np.nan),    _b("AUPRC")),
                _fmt(ov.get("F1", np.nan)),       _fmt_delta(ov.get("F1", np.nan),       _b("F1")),
                _fmt(ov.get("Brier", np.nan)),    _fmt_delta(ov.get("Brier", np.nan),    _b("Brier"), invert=True),
                _fmt(ov.get("ECE", np.nan)),      _fmt_delta(ov.get("ECE", np.nan),      _b("ECE"), invert=True),
                _fmt(ov.get("PPR_diff", np.nan)), _fmt_delta(ov.get("PPR_diff", np.nan), _b("PPR_diff"), invert=True),
                _fmt(ov.get("DP_diff", np.nan)),  _fmt_delta(ov.get("DP_diff", np.nan),  _b("DP_diff"), invert=True),
                _fmt(ov.get("TPR_diff", np.nan)), _fmt_delta(ov.get("TPR_diff", np.nan), _b("TPR_diff"), invert=True),
                _fmt(ov.get("EOp_diff", np.nan)), _fmt_delta(ov.get("EOp_diff", np.nan), _b("EOp_diff"), invert=True),
                _fmt(ov.get("FPR_diff", np.nan)), _fmt_delta(ov.get("FPR_diff", np.nan), _b("FPR_diff"), invert=True),
                _fmt(ov.get("EO_diff", np.nan)),  _fmt_delta(ov.get("EO_diff", np.nan),  _b("EO_diff"), invert=True),
            ]
            ov_tree.insert("", "end", values=row)

        ttk.Label(
            overview,
            text=(
                "Δ vs Baseline: positive means improvement "
                "(Brier/ECE/fairness-gap metrics inverted so lower is better \u2192 positive Δ)."
            ),
            foreground="#555"
        ).grid(row=1, column=0, sticky="w", padx=8, pady=(0, 8))

        def _autosize_overview(_evt=None):
            total = ov_tree.winfo_width()
            base_w = 360
            n = len(cols) - 1
            if total > (base_w + 100):
                w = max(90, (total - base_w - 20) // n)
                for c in cols[1:]:
                    ov_tree.column(c, width=w)

        overview.bind("<Configure>", _autosize_overview)

        # ------------- Group Metrics -------------
        group_tab = ttk.Frame(nb)
        nb.add(group_tab, text="Group Metrics")
        group_tab.rowconfigure(1, weight=1)
        group_tab.columnconfigure(0, weight=1)

        top = ttk.Frame(group_tab)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        ttk.Label(top, text="Technique:").pack(side="left")
        sel_var = tk.StringVar(value=results[0].name)
        sel_combo = ttk.Combobox(
            top,
            textvariable=sel_var,
            values=[r.name for r in results],
            state="readonly",
            width=60,
        )
        sel_combo.pack(side="left", padx=8)

        g_cols = ["group", "n", "TPR", "FPR", "PPV", "NPV", "PPR", "ECE"]
        g_frame = ttk.Frame(group_tab)
        g_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        g_frame.rowconfigure(0, weight=1)
        g_frame.columnconfigure(0, weight=1)

        g_tree = ttk.Treeview(g_frame, columns=g_cols, show="headings")
        g_vsb = ttk.Scrollbar(g_frame, orient="vertical", command=g_tree.yview)
        g_hsb = ttk.Scrollbar(g_frame, orient="horizontal", command=g_tree.xview)
        g_tree.configure(yscrollcommand=g_vsb.set, xscrollcommand=g_hsb.set)

        g_tree.grid(row=0, column=0, sticky="nsew")
        g_vsb.grid(row=0, column=1, sticky="ns")
        g_hsb.grid(row=1, column=0, sticky="ew")

        for c in g_cols:
            g_tree.heading(c, text=c)
            g_tree.column(c, width=140 if c != "group" else 280, anchor="center")
        g_tree.column("group", anchor="w")

        def refresh_group_table(name):
            g_tree.delete(*g_tree.get_children())
            rr = next(r for r in results if r.name == name)
            df = rr.group_stats.copy()
            for _, row in df.iterrows():
                g_tree.insert("", "end", values=[
                    row["group"],
                    int(row["n"]),
                    _fmt(row["TPR"]),
                    _fmt(row["FPR"]),
                    _fmt(row["PPV"]),
                    _fmt(row["NPV"]),
                    _fmt(row["PPR"]),
                    _fmt(row["ECE"]),
                ])

        sel_combo.bind("<<ComboboxSelected>>", lambda _e: refresh_group_table(sel_var.get()))
        refresh_group_table(sel_var.get())

        # ------------- Plots -------------
        plot_tab = ttk.Frame(nb)
        nb.add(plot_tab, text="Plots")
        plot_tab.rowconfigure(1, weight=1)
        plot_tab.columnconfigure(0, weight=1)

        pm = ttk.Frame(plot_tab)
        pm.grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        ttk.Label(pm, text="Metric:").pack(side="left")
        metric_var = tk.StringVar(value="AUROC")
        metric_combo = ttk.Combobox(
            pm,
            textvariable=metric_var,
            values=[
                "ACC", "AUROC", "AUPRC", "F1", "Brier", "ECE",
                "PPR_diff", "DP_diff", "TPR_diff", "EOp_diff", "FPR_diff", "EO_diff"
            ],
            state="readonly",
            width=18
        )
        metric_combo.pack(side="left", padx=8)

        canvas = tk.Label(plot_tab)
        canvas.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        def render_barplot(metric="AUROC", px=1000):
            fig_w = max(6.0, px / 130.0)
            fig = plt.figure(figsize=(fig_w, 4.2), dpi=120)
            names = [r.name for r in results]
            vals = [r.overall.get(metric, np.nan) for r in results]
            plt.bar(range(len(vals)), vals)
            plt.xticks(range(len(vals)), names, rotation=35, ha="right")
            plt.ylabel(metric)
            plt.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            im = Image.open(buf)
            return ImageTk.PhotoImage(im)

        def refresh_plot(*_):
            img = render_barplot(metric_var.get(), px=plot_tab.winfo_width())
            canvas.img = img
            canvas.configure(image=img)

        metric_combo.bind("<<ComboboxSelected>>", refresh_plot)
        plot_tab.bind("<Configure>", refresh_plot)
        refresh_plot()

        #------------- Export -------------
        export_bar = ttk.Frame(win); export_bar.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        def export_json():
            import json
            from tkinter import filedialog, messagebox
            data = []
            for rr in results:
                data.append({
                    "technique": rr.name,
                    "overall": rr.overall,
                    "group_stats": rr.group_stats.to_dict(orient="records"),
                    "notes": rr.notes
                })
            out = {"results": data}
            out_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")], title="Save results as")
            if out_path:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)
                messagebox.showinfo("Saved", f"Results saved to:\n{out_path}")
        ttk.Button(export_bar, text="Export JSON…", command=export_json).pack(side="left")
