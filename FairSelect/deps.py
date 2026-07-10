
"""
This file contains the dependency checks and imports for the package
"""
import warnings
import numpy as np

#NumPy has dependency issues with aif360, this helps smooth out data type issues arising from that
if not hasattr(np, "float"):   np.float   = float
if not hasattr(np, "int"):     np.int     = int
if not hasattr(np, "bool"):    np.bool    = bool
if not hasattr(np, "object"):  np.object  = object
if not hasattr(np, "complex"): np.complex = complex

warnings.filterwarnings("ignore", category=UserWarning)

AIF360_OK = True
try:
    from aif360.datasets import BinaryLabelDataset  
    from aif360.algorithms.inprocessing import PrejudiceRemover  
except Exception:
    AIF360_OK = False

SKLEARN_OK = True
try:
    from sklearn.preprocessing import OneHotEncoder, StandardScaler  
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline  
    from sklearn.linear_model import LogisticRegression 
    from sklearn.neural_network import MLPClassifier 
    from sklearn.ensemble import RandomForestClassifier  
    from sklearn.tree import DecisionTreeClassifier 
    from sklearn.svm import SVC 
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, average_precision_score, f1_score,
        brier_score_loss,
    ) 
    from sklearn.calibration import CalibrationDisplay 
    from sklearn.isotonic import IsotonicRegression  
    from sklearn.model_selection import train_test_split
except Exception:
    SKLEARN_OK = False

IMBLEARN_OK = True
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler  
except Exception:
    IMBLEARN_OK = False

FAIRLEARN_OK = True
try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
except Exception:
    FAIRLEARN_OK = False

XGB_OK = True
try:
    from xgboost import XGBClassifier
except Exception:
    XGB_OK = False

LGBM_OK = True
try:
    from lightgbm import LGBMClassifier  
except Exception:
    LGBM_OK = False

#plotting backend for GUI
import matplotlib
matplotlib.use("Agg")
