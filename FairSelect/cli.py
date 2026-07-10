import argparse
import json
import pandas as pd

from .runner import PipelineConfig, run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run fairness pipeline from command line"
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=20,
        help="Minimum acceptable intersectional group size. Default: 20."
    )

    parser.add_argument(
        "--allow-incomplete-outcome-coverage",
        action="store_true",
        help="Allow groups that do not contain both outcome classes."
    )

    parser.add_argument(
        "--no-group-filter",
        action="store_true",
        help="Disable filtering of small or incomplete intersectional groups."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV dataset"
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target column name"
    )

    parser.add_argument(
        "--protected",
        nargs="+",
        required=True,
        help="Protected attribute columns"
    )

    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Feature columns"
    )

    parser.add_argument(
        "--model",
        default="Logistic Regression",
        help="Model name"
    )

    parser.add_argument(
        "--params",
        default="{}",
        help="JSON string of model parameters"
    )

    parser.add_argument(
        "--techniques",
        nargs="*",
        default=[],
        help="Selected fairness techniques"
    )

    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Disable baseline run"
    )

    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Disable combined run"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save results CSV"
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1,2,3,4,5,6,7,8,9,10],
        help="Random seeds for repeated train/validation/test splits"
    )

    args = parser.parse_args()

    model_params = json.loads(args.params)
    
    # Print results
    rows = []

    for seed in args.seeds:

        cfg = PipelineConfig(
            df_or_path=str(model_params.get("data", args.data)),
            target=model_params.get("target", args.target),
            protected=model_params.get("protected", args.protected),
            features=model_params.get("features", args.features),
            model_name=model_params.get("model", args.model),
            model_params=model_params,
            techniques=model_params.get("techniques", args.techniques),
            run_baseline=model_params.get("run_baseline", not args.no_baseline),
            run_combined=model_params.get("run_combined", not args.no_combined) ,
            random_state=seed,
        )

        results = run_pipeline(cfg)



        for r in results:
            print("\n=========================")
            print(r.name)

            for k, v in r.overall.items():
                print(f"{k}: {v:.4f}")

            row = {"Technique": r.name}
            row.update(r.overall)
            rows.append(row)

    # Optional save
    if args.output:
        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()