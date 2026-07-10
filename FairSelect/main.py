from tkinter import messagebox
from .deps import SKLEARN_OK
from .gui import FairnessToolGUI

from .runner import PipelineConfig, run_pipeline

'''
if __name__ == "__main__":
    cfg = PipelineConfig(
        df_or_path="path/to/data.csv",
        target="y",
        protected=["sex", "race"],
        features=["age", "bmi", "hdl", "ldl", "sex", "race"],  # target/protected will be removed automatically
        model_name="logreg",
        model_params={"max_iter": 1000},
        techniques=[
            "Pre:Reweight (y,a)",
            "In:Reductions (EO)",
            "Post:Youden per group",
        ],
        run_baseline=True,
        run_combined=True,
    )

for seed in args.seeds:

    cfg = PipelineConfig(
        df_or_path=str(csv_path),
        target=target,
        protected=protected,
        features=features,
        model_name=args.model_name,
        model_params={},
        techniques=plan["techniques"],
        run_baseline=plan["run_baseline"],
        run_combined=plan["run_combined"],
        random_state=seed,
    )

    results = run_pipeline(cfg)

    for rr in results:
        rec = rr_to_record(
            rr=rr,
            dataset_name=dataset_name,
            plan_name=plan["plan_name"],
            model_name=args.model_name,
            target=target,
            protected=protected,
            features=features,
        )

        rec["seed"] = seed
        rec["selected_techniques"] = plan["techniques"]

        all_records.append(rec)
'''


### Below runs the GUI application ###
      
def main():
    if not SKLEARN_OK:
        messagebox.showerror("Missing dependency", "scikit-learn is required. Please install it first.")
        return
    app = FairnessToolGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
