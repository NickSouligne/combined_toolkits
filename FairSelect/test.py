from itertools import product
import argparse
import json
import os
import traceback
import yaml
import pandas as pd

from runner import PipelineConfig, run_pipeline
from params import AVAILABLE_MODELS

PRE_TECHNIQUES = [
    "Pre:Reweight (y,a)",
    "Pre:SMOTE / Oversample",
    "Pre:Local Massaging",
]

IN_TECHNIQUES = [
    "In:Compositional per-group",
    "In:Ensemble (K=5)",
    "In:Multicalibration (isotonic)",
    "In:Reductions (EO)",
    "In:Fairness Regularization (Prejudice Remover)",
]

POST_TECHNIQUES = [
    "Post:Youden per group",
    "Post:Multiaccuracy Boost",
    "Post:Reject-Option Shift",
    "Post:Reject-Option Kamiran",
]

SINGLE_TECHNIQUES = PRE_TECHNIQUES + IN_TECHNIQUES + POST_TECHNIQUES


MODEL_PARAMS = {
    "Logistic Regression": {
        "max_iter": 200,
        "random_state": 42,
    },
    "Neural Network": {
        "hidden_layer_sizes": (100,),
        "max_iter": 200,
        "random_state": 42,
    },
    "Random Forest": {
        "n_estimators": 200,
        "random_state": 42,
    },
    "Decision Tree": {
        "random_state": 42,
    },
    "SVM": {
        "probability": True,
    },
    "XGBoost": {
        "n_estimators": 300,
        "random_state": 42,
    },
    "LightGBM": {
        "n_estimators": 300,
        "random_state": 42,
    },
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_list(value):
    """
    Allows CLI inputs like:
    --protected race,gender
    --features age,bmi,diabetes
    """
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [x.strip() for x in value.split(",") if x.strip()]


def make_plans(include_baseline_only=True, include_single=True, include_combined=True):
    plans = []

    if include_baseline_only:
        plans.append({
            "plan_name": "baseline_only",
            "techniques": [],
            "run_baseline": True,
            "run_combined": False,
        })

    if include_single:
        for tech in SINGLE_TECHNIQUES:
            plans.append({
                "plan_name": f"single__{tech}",
                "techniques": [tech],
                "run_baseline": True,
                "run_combined": False,
            })

    if include_combined:
        for i, (pre, in_, post) in enumerate(
            product(PRE_TECHNIQUES, IN_TECHNIQUES, POST_TECHNIQUES),
            start=1,
        ):
            plans.append({
                "plan_name": f"combined_{i:03d}",
                "techniques": [pre, in_, post],
                "run_baseline": True,
                "run_combined": True,
            })

    return plans


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def rr_to_flat_record(rr, dataset_name, plan_name, model_name, target, protected, features):
    row = {
        "dataset": dataset_name,
        "plan": plan_name,
        "technique": getattr(rr, "name", None),
        "model_name": model_name,
        "target": target,
        "protected": "|".join(protected),
        "n_features": len(features),
        "notes": getattr(rr, "notes", None),
    }

    overall = getattr(rr, "overall", {}) or {}
    for k, v in overall.items():
        row[k] = safe_float(v)

    return row


def rr_to_group_record(rr, dataset_name, plan_name, model_name):
    gs = getattr(rr, "group_stats", None)
    if gs is None or len(gs) == 0:
        return pd.DataFrame()

    out = gs.copy()
    out["dataset"] = dataset_name
    out["plan"] = plan_name
    out["technique"] = getattr(rr, "name", None)
    out["model_name"] = model_name
    return out


def apply_cli_overrides(config: dict, args) -> dict:
    """
    CLI arguments override the YAML config.
    This makes the script usable both locally and inside GitHub Actions.
    """
    if args.data_path:
        config["data_path"] = args.data_path

    if args.dataset_name:
        config["dataset_name"] = args.dataset_name

    if args.target:
        config["target"] = args.target

    if args.protected:
        config["protected"] = parse_list(args.protected)

    if args.features:
        config["features"] = parse_list(args.features)

    if args.models:
        config["models"] = parse_list(args.models)

    if args.min_group_size is not None:
        config.setdefault("pipeline", {})["min_group_size"] = args.min_group_size

    if args.test_size is not None:
        config.setdefault("pipeline", {})["test_size"] = args.test_size

    if args.val_size is not None:
        config.setdefault("pipeline", {})["val_size"] = args.val_size

    if args.random_state is not None:
        config.setdefault("pipeline", {})["random_state"] = args.random_state

    return config


def validate_config(config: dict):
    required = ["dataset_name", "data_path", "target", "protected", "features"]
    missing = [key for key in required if key not in config or config[key] in [None, ""]]

    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    if not os.path.exists(config["data_path"]):
        raise FileNotFoundError(f"Dataset not found: {config['data_path']}")

    df = pd.read_csv(config["data_path"], nrows=5)
    columns = set(df.columns)

    required_columns = [config["target"]] + config["protected"] + config["features"]
    missing_columns = [col for col in required_columns if col not in columns]

    if missing_columns:
        raise ValueError(
            "The following target/protected/feature columns are missing from the dataset: "
            f"{missing_columns}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run FairSelect/FairLogue benchmark pipeline.")

    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--protected", default=None)
    parser.add_argument("--features", default=None)
    parser.add_argument("--models", default=None)

    parser.add_argument("--min-group-size", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--val-size", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)

    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    validate_config(config)

    dataset_name = config["dataset_name"]
    data_path = config["data_path"]
    target = config["target"]
    protected = config["protected"]
    features = config["features"]

    model_names = config.get("models", AVAILABLE_MODELS)

    pipeline_cfg = config.get("pipeline", {})
    plan_cfg = config.get("plans", {})
    output_cfg = config.get("outputs", {})

    plans = make_plans(
        include_baseline_only=plan_cfg.get("include_baseline_only", True),
        include_single=plan_cfg.get("include_single", True),
        include_combined=plan_cfg.get("include_combined", True),
    )

    print(f"Dataset: {dataset_name}")
    print(f"Data path: {data_path}")
    print(f"Target: {target}")
    print(f"Protected: {protected}")
    print(f"Number of features: {len(features)}")
    print(f"Models: {model_names}")
    print(f"Number of plans: {len(plans)}")

    all_flat_rows = []
    all_group_rows = []
    failures = []

    for model_name in model_names:
        if model_name not in AVAILABLE_MODELS:
            print(f"Skipping unavailable model: {model_name}")
            continue

        model_params = MODEL_PARAMS.get(model_name, {})

        for i, plan in enumerate(plans, start=1):
            print(f"[{model_name}] [{i}/{len(plans)}] {plan['plan_name']}")

            cfg = PipelineConfig(
                df_or_path=data_path,
                target=target,
                protected=protected,
                features=features,
                model_name=model_name,
                model_params=model_params,
                min_group_size=pipeline_cfg.get("min_group_size", 40),
                techniques=plan["techniques"],
                run_baseline=plan["run_baseline"],
                run_combined=plan["run_combined"],
                test_size=pipeline_cfg.get("test_size", 0.25),
                val_size=pipeline_cfg.get("val_size", 0.20),
                random_state=pipeline_cfg.get("random_state", 42),
            )

            try:
                results = run_pipeline(cfg)

                for rr in results:
                    all_flat_rows.append(
                        rr_to_flat_record(
                            rr=rr,
                            dataset_name=dataset_name,
                            plan_name=plan["plan_name"],
                            model_name=model_name,
                            target=target,
                            protected=protected,
                            features=features,
                        )
                    )

                    gs = rr_to_group_record(rr, dataset_name, plan["plan_name"], model_name)
                    if not gs.empty:
                        all_group_rows.append(gs)

            except Exception as e:
                failures.append({
                    "dataset": dataset_name,
                    "model_name": model_name,
                    "plan": plan["plan_name"],
                    "techniques": json.dumps(plan["techniques"]),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                print(f"FAILED: {plan['plan_name']}")

    os.makedirs("results", exist_ok=True)

    flat_path = output_cfg.get("flat_results_path", "results/results_flat.csv")
    group_path = output_cfg.get("group_results_path", "results/results_group.csv")
    failures_path = output_cfg.get("failures_path", "results/failures.csv")

    pd.DataFrame(all_flat_rows).to_csv(flat_path, index=False)

    if all_group_rows:
        pd.concat(all_group_rows, ignore_index=True).to_csv(group_path, index=False)
    else:
        pd.DataFrame().to_csv(group_path, index=False)

    pd.DataFrame(failures).to_csv(failures_path, index=False)

    print(f"Saved flat results to: {flat_path}")
    print(f"Saved group results to: {group_path}")
    print(f"Saved failures to: {failures_path}")

    if failures:
        print(f"Completed with {len(failures)} failed runs.")
    else:
        print("Completed with no failed runs.")


if __name__ == "__main__":
    main()