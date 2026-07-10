from html import parser
from pathlib import Path

import argparse
import importlib
import json
import sys
import traceback
import types
from itertools import product
from pathlib import Path
from .runner import PipelineConfig, run_pipeline

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ..FairModel import FairModel


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def rr_to_record(rr, dataset_name, plan_name, model_name, target, protected, features):
    record = {
        "dataset": dataset_name,
        "plan": plan_name,
        "technique": getattr(rr, "name", None),
        "model_name": model_name,
        "target": target,
        "protected": list(protected),
        "n_features": len(features),

        # FairSelect-native outputs
        "fairselect_overall": {},
        "fairselect_group_stats": [],

        # FairLogue outputs
        "fairlogue": {},

        "notes": getattr(rr, "notes", None),
        "has_fair_model": getattr(rr, "fair_model", None) is not None,
    }

    # -----------------------------
    # FairSelect metrics
    # -----------------------------
    overall = getattr(rr, "overall", {}) or {}
    for k, v in overall.items():
        record["fairselect_overall"][k] = safe_float(v)

    gs = getattr(rr, "group_stats", None)
    if gs is not None:
        try:
            record["fairselect_group_stats"] = gs.to_dict(orient="records")
        except Exception:
            record["fairselect_group_stats"] = []

    # -----------------------------
    # FairLogue metrics
    # -----------------------------
    fairlogue = getattr(rr, "fairlogue", None)

    if isinstance(fairlogue, dict):
        for comp_name, comp_res in fairlogue.items():

            if not isinstance(comp_res, dict):
                record["fairlogue"][comp_name] = {
                    "status": "unparsed",
                    "raw": str(comp_res),
                }
                continue

            comp_out = {}

            # Keep basic metadata
            for meta_key in [
                "status",
                "component",
                "audit_source",
                "model_name",
                "error",
                "reason",
            ]:
                if meta_key in comp_res:
                    comp_out[meta_key] = comp_res[meta_key]

            # Component 1 group stats, if present
            if "group_stats" in comp_res:
                try:
                    comp_out["group_stats"] = comp_res["group_stats"].to_dict(
                        orient="records"
                    )
                except Exception:
                    comp_out["group_stats"] = str(comp_res["group_stats"])

            # Component 3 summary table, if present
            if "summary" in comp_res:
                try:
                    comp_out["summary"] = comp_res["summary"].to_dict(
                        orient="records"
                    )
                except Exception:
                    comp_out["summary"] = str(comp_res["summary"])

            # Store scalar items from results if available
            if "results" in comp_res and isinstance(comp_res["results"], dict):
                results_obj = comp_res["results"]

                for key, val in results_obj.items():
                    if key in {"defs", "est_choice"}:
                        continue

                    if isinstance(val, (str, int, float, bool)) or val is None:
                        comp_out[key] = val
                    else:
                        comp_out[key] = str(val)

            record["fairlogue"][comp_name] = comp_out

    return record


def flatten_record(record):
    flat = {
        "dataset": record["dataset"],
        "plan": record["plan"],
        "technique": record["technique"],
        "model_name": record["model_name"],
        "target": record["target"],
        "protected": "|".join(record["protected"]),
        "n_features": record["n_features"],
        "notes": record["notes"],
        "seed": record.get("seed"),
        "has_fair_model": record.get("has_fair_model", False),
    }

    # -----------------------------
    # FairSelect overall metrics
    # -----------------------------
    for k, v in record.get("fairselect_overall", {}).items():
        flat[f"fairselect_{k}"] = v

    # -----------------------------
    # FairLogue outputs
    # -----------------------------
    fairlogue = record.get("fairlogue", {}) or {}

    for comp_name, comp_res in fairlogue.items():
        prefix = f"fairlogue_{comp_name}"

        if not isinstance(comp_res, dict):
            flat[f"{prefix}_raw"] = str(comp_res)
            continue

        # Metadata/status
        for k in ["status", "component", "audit_source", "model_name", "error", "reason"]:
            if k in comp_res:
                flat[f"{prefix}_{k}"] = comp_res[k]

        # Component 3 summary often comes as rows with stat/value columns.
        summary = comp_res.get("summary")
        if isinstance(summary, list):
            for row in summary:
                if not isinstance(row, dict):
                    continue

                stat_name = (
                    row.get("stat")
                    or row.get("metric")
                    or row.get("name")
                    or row.get("parameter")
                )

                if stat_name is None:
                    continue

                stat_name = str(stat_name)

                for value_key in ["estimate", "value", "est", "theta", "mean"]:
                    if value_key in row:
                        flat[f"{prefix}_{stat_name}_{value_key}"] = safe_float(row[value_key])
                        break

                # Also preserve confidence interval / p-value style fields if present
                for extra_key in ["se", "lower", "upper", "p", "p_value", "ci_low", "ci_high"]:
                    if extra_key in row:
                        flat[f"{prefix}_{stat_name}_{extra_key}"] = safe_float(row[extra_key])

        # Component 1 group stats: summarize as gaps so it fits flat CSV
        group_stats = comp_res.get("group_stats")
        if isinstance(group_stats, list) and len(group_stats) > 0:
            gdf = pd.DataFrame(group_stats)

            for metric in [
                "predicted_positive_rate",
                "mean_score",
                "TPR",
                "FPR",
                "FNR",
                "TNR",
                "PPV",
                "NPV",
            ]:
                if metric in gdf.columns:
                    vals = pd.to_numeric(gdf[metric], errors="coerce")
                    if vals.notna().sum() > 0:
                        flat[f"{prefix}_{metric}_max"] = float(vals.max())
                        flat[f"{prefix}_{metric}_min"] = float(vals.min())
                        flat[f"{prefix}_{metric}_diff"] = float(vals.max() - vals.min())

            if "n" in gdf.columns:
                flat[f"{prefix}_n_groups"] = int(len(gdf))
                flat[f"{prefix}_min_group_n"] = int(pd.to_numeric(gdf["n"], errors="coerce").min())
                flat[f"{prefix}_max_group_n"] = int(pd.to_numeric(gdf["n"], errors="coerce").max())

        # Scalar results
        for k, v in comp_res.items():
            if k in {"summary", "group_stats", "results"}:
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                flat[f"{prefix}_{k}"] = v

    return flat


def records_to_long_metrics(all_records):
    rows = []

    for record in all_records:
        base = {
            "dataset": record["dataset"],
            "plan": record["plan"],
            "technique": record["technique"],
            "model_name": record["model_name"],
            "target": record["target"],
            "protected": "|".join(record["protected"]),
            "seed": record.get("seed"),
        }

        # FairSelect overall metrics
        for metric, value in record.get("fairselect_overall", {}).items():
            rows.append({
                **base,
                "source": "FairSelect",
                "component": "overall",
                "level": "overall",
                "group": None,
                "metric": metric,
                "value": safe_float(value),
            })

        # FairSelect group stats
        for row in record.get("fairselect_group_stats", []):
            group = row.get("group")
            for metric, value in row.items():
                if metric == "group":
                    continue
                rows.append({
                    **base,
                    "source": "FairSelect",
                    "component": "group_stats",
                    "level": "group",
                    "group": group,
                    "metric": metric,
                    "value": safe_float(value),
                })

        # FairLogue
        fairlogue = record.get("fairlogue", {}) or {}

        for comp_name, comp_res in fairlogue.items():
            if not isinstance(comp_res, dict):
                continue

            # Component 1 group stats
            for row in comp_res.get("group_stats", []) or []:
                group = row.get("group")
                for metric, value in row.items():
                    if metric == "group":
                        continue
                    rows.append({
                        **base,
                        "source": "FairLogue",
                        "component": comp_name,
                        "level": "group",
                        "group": group,
                        "metric": metric,
                        "value": safe_float(value),
                    })

            # Component 3 summary
            for row in comp_res.get("summary", []) or []:
                if not isinstance(row, dict):
                    continue

                metric = (
                    row.get("stat")
                    or row.get("metric")
                    or row.get("name")
                    or row.get("parameter")
                )

                if metric is None:
                    continue

                for value_key in ["estimate", "value", "est", "theta", "mean"]:
                    if value_key in row:
                        rows.append({
                            **base,
                            "source": "FairLogue",
                            "component": comp_name,
                            "level": "overall",
                            "group": None,
                            "metric": str(metric),
                            "value": safe_float(row[value_key]),
                        })

    return pd.DataFrame(rows)



def infer_columns(df: pd.DataFrame):
    candidate_targets = [
        "glaucoma_intervention",
        "outcome",
        "target",
        "label",
        "y",
    ]
    candidate_protected = [
        ["Race", "Gender"],
        ["race", "sex"],
        ["race", "gender"],
        ["Race", "Sex"],
    ]

    target = next((c for c in candidate_targets if c in df.columns), None)
    if target is None:
        raise ValueError("Could not infer target column. Pass --target explicitly.")

    protected = next(
        (cols for cols in candidate_protected if all(c in df.columns for c in cols)),
        None,
    )
    if protected is None:
        raise ValueError("Could not infer protected columns. Pass --protected explicitly.")

    features = [c for c in df.columns if c != target and c not in protected]
    if not features:
        raise ValueError("No usable feature columns found after excluding target/protected.")

    return target, protected, features


def compute_deltas_vs_baseline(df_flat: pd.DataFrame):
    fairness_metrics = [
        "fairselect_PPR_diff",
        "fairselect_DP_diff",
        "fairselect_TPR_diff",
        "fairselect_EOp_diff",
        "fairselect_FPR_diff",
        "fairselect_EO_diff",
    ]

    utility_metrics = [
        "fairselect_ACC",
        "fairselect_AUROC",
        "fairselect_AUPRC",
        "fairselect_F1",
        "fairselect_Brier",
        "fairselect_ECE",
    ]

    out = df_flat.copy()
    baseline_rows = out[out["technique"] == "Baseline"].copy()

    if baseline_rows.empty:
        return out

    baseline_cols = ["dataset", "plan"] + fairness_metrics + utility_metrics
    baseline_rows = baseline_rows[[c for c in baseline_cols if c in baseline_rows.columns]].copy()
    baseline_rows = baseline_rows.rename(
        columns={c: f"{c}_baseline" for c in baseline_rows.columns if c not in ["dataset", "plan", "seed"]}
    )

    out = out.merge(baseline_rows, on=["dataset", "plan"], how="left")

    for m in fairness_metrics + utility_metrics:
        if m in out.columns and f"{m}_baseline" in out.columns:
            out[f"delta_{m}"] = out[m] - out[f"{m}_baseline"]

    return out

PRE_TECHNIQUES = [
    "Pre:Reweight (y,a)",
    "Pre:SMOTE / Oversample",
]

IN_TECHNIQUES = [
    "In:Multicalibration",
    "In:Prejudice Remover",
]

POST_TECHNIQUES = [
    "Post:Youden per group",
    "Post:Multiaccuracy Boost",
]


SINGLE_TECHNIQUES = PRE_TECHNIQUES + IN_TECHNIQUES + POST_TECHNIQUES

"""
PRE_TECHNIQUES = [
    "Pre:Reweight (y,a)",
    "Pre:SMOTE / Oversample",
    "Pre:Local Massaging",
]

IN_TECHNIQUES = [
    "In:Compositional per-group",
    "In:Group-balanced ensemble",
    "In:Multicalibration",
    "In:Reductions (EO)",
    "In:Prejudice Remover",
]

POST_TECHNIQUES = [
    "Post:Youden per group",
    "Post:Multiaccuracy Boost",
    "Post:Reject-option shift",
    "Post:Input Repair (z-align)",
    "Post:Kamiran Reject Option",
]

SCENARIO_MATCH = {
    "pre_reweight_y_a": ["Pre:Reweight (y,a)"],
    "pre_smote_rare_positive": ["Pre:SMOTE / Oversample"],
    "pre_local_massaging_label_bias": ["Pre:Local Massaging"],
    "in_compositional_group_specific": ["In:Compositional per-group"],
    "in_group_balanced_ensemble": ["In:Group-balanced ensemble"],
    "in_multicalibration_group_miscalibration": ["In:Multicalibration"],
    "in_reductions_equalized_odds": ["In:Reductions (EO)"],
    "in_prejudice_remover_sensitive_attribute": ["In:Prejudice Remover"],
    "post_youden_threshold_heterogeneity": ["Post:Youden per group"],
    "post_multiaccuracy_residual_structure": ["Post:Multiaccuracy Boost"],
    "post_reject_option_around_boundary": ["Post:Reject-option shift"],
    "post_input_repair_measurement_shift": ["Post:Input Repair (z-align)"],
}
"""

SCENARIO_MATCH = {
    "pre_reweight_y_a": ["Pre:Reweight (y,a)"],
    "pre_local_massaging_label_bias": ["Pre:Local Massaging"],
    "in_reductions_equalized_odds": ["In:Reductions (EO)"],
    "in_prejudice_remover_sensitive_attribute": ["In:Prejudice Remover"],
    "post_reject_option_around_boundary": ["Post:Reject-option shift"],
    "post_input_repair_measurement_shift": ["Post:Input Repair (z-align)"],
}


def make_all_combined_plans():
    combos = []
    for i, (pre, in_, post) in enumerate(product(PRE_TECHNIQUES, IN_TECHNIQUES, POST_TECHNIQUES), start=1):
        combos.append({
            "plan_name": f"combined_{i:03d}",
            "techniques": [pre, in_, post],
            "run_baseline": True,
            "run_combined": True,
        })
    return combos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Folder containing generated scenario CSVs")
    parser.add_argument("--toolkit-dir", required=True, help="Folder containing runner.py, core.py, deps.py, etc.")
    parser.add_argument("--out-dir", default="benchmark_results_all_combos", help="Output folder")
    parser.add_argument("--include", nargs="*", default=None, help="Optional dataset stems to include")
    parser.add_argument("--target", default=None, help="Override target column")
    parser.add_argument("--protected", nargs="*", default=None, help="Override protected columns")
    parser.add_argument("--model-name", default="LightGBM", help="Toolkit model name")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 11, 9, 21, 7, 92, 55, 63, 3, 17],
        help="Random seeds for repeated train/validation/test splits"
    )
    parser.add_argument("--skip-single", action="store_true", help="Skip single-technique plans")
    parser.add_argument("--skip-scenario-match", action="store_true", help="Skip scenario-matched plans")
    parser.add_argument("--skip-combined", action="store_true", help="Skip all combined plans")
    parser.add_argument("--no-baseline-only", action="store_true", help="Skip baseline-only plan")
    parser.add_argument("--fairlogue-comp1", action="store_true")
    parser.add_argument("--fairlogue-comp3", action="store_true")

    parser.add_argument("--write-stats", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    toolkit_path = Path(args.toolkit_dir).resolve()

    if not toolkit_path.exists():
        raise FileNotFoundError(f"Toolkit directory not found: {toolkit_path}")

    runner_file = toolkit_path / "runner.py"
    if not runner_file.exists():
        raise FileNotFoundError(
            f"Could not find runner.py in toolkit directory: {toolkit_path}"
        )

    data_dir = Path(args.data_dir)
    csvs = sorted([p for p in data_dir.glob("*.csv") if p.name.lower() != "manifest.json"])

    if args.include:
        include_set = set(args.include)
        csvs = [p for p in csvs if p.stem in include_set]

    all_records = []
    failures = []
    combined_plans = [] if args.skip_combined else make_all_combined_plans()

    for csv_path in csvs:
        dataset_name = csv_path.stem
        print(f"\\n=== Running dataset: {dataset_name} ===")

        try:
            df = pd.read_csv(csv_path)

            if args.target and args.protected:
                target = args.target
                protected = args.protected
                features = [c for c in df.columns if c != target and c not in protected]
            else:
                target, protected, features = infer_columns(df)

            plans = []

            if not args.no_baseline_only:
                plans.append({
                    "plan_name": "baseline_only",
                    "techniques": [],
                    "run_baseline": True,
                    "run_combined": False,
                })

            if not args.skip_single:
                for tech in SINGLE_TECHNIQUES:
                    plans.append({
                        "plan_name": f"single__{tech}",
                        "techniques": [tech],
                        "run_baseline": True,
                        "run_combined": False,
                    })

            if (not args.skip_scenario_match) and (dataset_name in SCENARIO_MATCH):
                plans.append({
                    "plan_name": f"scenario_match__{dataset_name}",
                    "techniques": SCENARIO_MATCH[dataset_name],
                    "run_baseline": True,
                    "run_combined": False,
                })

            plans.extend(combined_plans)

            for idx, plan in enumerate(plans, start=1):
                print(f"  -> [{idx}/{len(plans)}] {plan['plan_name']}")

                for seed in args.seeds:
                    print(f"     seed={seed}")

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
                        fairlogue_comp1=args.fairlogue_comp1,
                        fairlogue_comp3=args.fairlogue_comp3,
                    )

                    try:
                        results = run_pipeline(cfg)
                    except Exception as e:
                        failures.append({
                            "dataset": dataset_name,
                            "plan": plan["plan_name"],
                            "seed": seed,
                            "techniques": plan["techniques"],
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        })
                        print(f"     !! PLAN FAILED: {plan['plan_name']} seed={seed}")
                        continue

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
                        all_records.append(rec)

        except Exception as e:
            failures.append({
                "dataset": dataset_name,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            print(f"  !! DATASET FAILED: {dataset_name}")
            print(traceback.format_exc())

    jsonl_path = out_dir / "benchmark_results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    flat_rows = [flatten_record(r) for r in all_records]
    df_flat = pd.DataFrame(flat_rows)
    if not df_flat.empty:
        df_flat = compute_deltas_vs_baseline(df_flat)

    flat_csv_path = out_dir / "benchmark_results_flat.csv"
    df_flat.to_csv(flat_csv_path, index=False)

    summary_cols = [
        "dataset", "plan", "technique", "seed",
        "has_fair_model",

        # FairSelect utility
        "fairselect_ACC",
        "fairselect_AUROC",
        "fairselect_AUPRC",
        "fairselect_F1",
        "fairselect_Brier",
        "fairselect_ECE",

        # FairSelect fairness
        "fairselect_PPR_diff",
        "fairselect_DP_diff",
        "fairselect_TPR_diff",
        "fairselect_EOp_diff",
        "fairselect_FPR_diff",
        "fairselect_EO_diff",

        # FairLogue Component 1 observed summaries
        "fairlogue_component1_status",
        "fairlogue_component1_predicted_positive_rate_diff",
        "fairlogue_component1_mean_score_diff",
        "fairlogue_component1_TPR_diff",
        "fairlogue_component1_FPR_diff",
        "fairlogue_component1_FNR_diff",

        # FairLogue Component 3 counterfactual summaries
        "fairlogue_component3_status",
        "fairlogue_component3_audit_source",
        "fairlogue_component3_tau",
        "fairlogue_component3_groups",

        # Deltas
        "delta_fairselect_ACC",
        "delta_fairselect_AUROC",
        "delta_fairselect_AUPRC",
        "delta_fairselect_F1",
        "delta_fairselect_Brier",
        "delta_fairselect_ECE",
        "delta_fairselect_PPR_diff",
        "delta_fairselect_DP_diff",
        "delta_fairselect_TPR_diff",
        "delta_fairselect_EOp_diff",
        "delta_fairselect_FPR_diff",
        "delta_fairselect_EO_diff",
    ]
    summary_df = df_flat[[c for c in summary_cols if c in df_flat.columns]].copy()
    summary_path = out_dir / "benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    long_metrics_df = records_to_long_metrics(all_records)
    long_metrics_path = out_dir / "benchmark_metrics_long.csv"
    long_metrics_df.to_csv(long_metrics_path, index=False)
    print(f"Long metrics:      {long_metrics_path}")

    failures_path = out_dir / "failures.json"
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    manifest = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "toolkit_dir": str(Path(args.toolkit_dir).resolve()),
        "out_dir": str(out_dir.resolve()),
        "n_datasets": len(csvs),
        "n_records": len(all_records),
        "n_failures": len(failures),
        "model_name": args.model_name,
        "n_pre": len(PRE_TECHNIQUES),
        "n_in": len(IN_TECHNIQUES),
        "n_post": len(POST_TECHNIQUES),
        "n_combined_plans": len(combined_plans),
    }
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\\nDone.")
    print(f"Detailed results: {jsonl_path}")
    print(f"Flat results:     {flat_csv_path}")
    print(f"Summary:          {summary_path}")
    print(f"Failures:         {failures_path}")


if __name__ == "__main__":
    main()

