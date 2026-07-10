from pathlib import Path
import sys

import pandas as pd
from matplotlib import pyplot as plt

from intersectional_metrics import cross_validate_intersectional_fairness, evaluate_intersectional_fairness


# ---------------------------------------------------------------------
# Import shared FairModel from:
# Combined_Toolkits/FairModel.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FairModel import FairModel

#-----Example usage (synthetic)------

if __name__ == "__main__":
    #Load the synthetic data (Replace with file path)
    df = pd.read_csv("C:\\Users\\nicks\\Desktop\\combined_toolkits\\FairLogue\\Component1\\synthetic_data.csv")
    
    # Define model parameters for LightGBM
    MODEL_PARAMS = {
        "objective": "binary",
        "n_estimators": 600,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "class_weight": "balanced",
    }

    # Define model object that stores the Component 1 configuration
    fair_model = FairModel(
        name="LightGBM baseline glaucoma model",
        features=None,  # Let Component 1 infer features by dropping outcome/protected cols
        protected_cols=["Race", "MALE_GENDER"],

        outcome_col="SURGERY",
        positive_label=1,

        model_type="lgbm",
        model_params=MODEL_PARAMS,
        test_size=0.3,
        random_state=42,
        threshold=0.5,

        require_class_balance=True,
        min_group_size=20,
        make_plots=True,
        return_intermediates=True,
        return_non_intersectional=True,

        metadata={
            "component": "FairLogue Component 1",
            "dataset": "glaucoma_synth_component3.csv",
            "purpose": "initial FairModel compatibility test",
        },
    )

    # Component 1 currently supports two protected columns
    if len(fair_model.protected_cols) != 2:
        raise ValueError(
            "FairLogue Component 1 currently expects exactly two protected columns."
        )

    protected_1, protected_2 = fair_model.protected_cols

    result = cross_validate_intersectional_fairness(
        df=df,
        outcome=fair_model.outcome_col,
        protected_1=protected_1,
        protected_2=protected_2,
        features=fair_model.features,
        model_type=fair_model.model_type,
        model_params=fair_model.model_params,
        k=5,
        test_size=fair_model.test_size,
        random_state=fair_model.random_state,
        positive_label=fair_model.positive_label,
        threshold=fair_model.threshold,
        min_group_size=fair_model.min_group_size,
        require_class_balance=fair_model.require_class_balance,
        make_plots=fair_model.make_plots,
        return_intermediates=fair_model.return_intermediates,
        return_non_intersectional=fair_model.return_non_intersectional,
    )

    # Run the original Component 1 function using values stored in FairModel
    '''
    result = evaluate_intersectional_fairness(
        df=df,
        outcome=fair_model.outcome_col,
        protected_1=protected_1,
        protected_2=protected_2,
        features=fair_model.features,
        model_type=fair_model.model_type,
        model_params=fair_model.model_params,
        test_size=fair_model.test_size,
        random_state=fair_model.random_state,
        positive_label=fair_model.positive_label,
        threshold=fair_model.threshold,
        require_class_balance=fair_model.require_class_balance,
        min_group_size=fair_model.min_group_size,
        make_plots=fair_model.make_plots,
        return_intermediates=fair_model.return_intermediates,
        return_non_intersectional=fair_model.return_non_intersectional,
    )
    '''


    # Handle return shape depending on return_intermediates
    if fair_model.return_intermediates:
        res, figs, inter = result
    else:
        res, figs = result
        inter = None

    # Store FairLogue Component 1 outputs back onto FairModel
    fair_model.fairlogue_component1_results = res
    fair_model.fairlogue_component1_figs = figs
    fair_model.fairlogue_component1_intermediates = inter

    # Print the same summary as before
    print("Model:", res.model)
    print("Demographic parity gap (max-min P(Ŷ=1)):", res.demographic_parity_gap)
    print("Equalized odds TPR gap:", res.equalized_odds_gap_tpr)
    print("Equalized odds FPR gap:", res.equalized_odds_gap_fpr)
    print("Equal opportunity gap (TPR gap):", res.equal_opportunity_gap)
    print("Results per fold: ", result['fold_metrics'])
    print("Bootstrap results: ", result['bootstrap_overall'])

    print("\nPer-group metrics:")
    print(res.per_group_df)

    if inter is not None and inter.get("non_intersectional") is not None:
        print("\nRace non-intersectional metrics:")
        print(inter["non_intersectional"]["Race"].per_group_df)

        print("\nMALE_GENDER demographic parity gap:")
        print(inter["non_intersectional"]["MALE_GENDER"].demographic_parity_gap)

    # Show plots when running as a script
    for name, fig in figs.items():
        print(f"Showing plot: {name}")
        fig.show()
        plt.show()