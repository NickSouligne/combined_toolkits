from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class GroupRates:
    group: str
    n: int
    positive_rate: float   #P(Ŷ=1)
    tpr: float             #P(Ŷ=1 | Y=1)
    fpr: float             #P(Ŷ=1 | Y=0)
    pos_true: int          #count of Y=1 in this group's TEST set
    neg_true: int          #count of Y=0 in this group's TEST set
    TP: int
    FP: int
    TN: int
    FN: int

@dataclass
class FairnessResults:
    model: str
    groups: List[GroupRates]
    demographic_parity_gap: float
    equalized_odds_gap_tpr: float
    equalized_odds_gap_fpr: float
    equal_opportunity_gap: float
    per_group_df: pd.DataFrame
    dropped_groups: List[str]            #groups dropped due to insufficient data
    kept_groups_summary: pd.DataFrame    #n/pos/neg for kept groups