import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


NUMERICAL_COLS = [
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7",
    "v8",
    "v9",
    "v10",
    "v11",
    "v12",
    "v13",
    "v14",
    "v15",
    "v16",
    "v17",
    "v18",
    "v19",
    "v20",
    "v21",
    "v22",
    "v23",
    "v24",
    "v25",
    "v26",
    "v27",
    "v28",
    "amount",
]

# pipeline for numerical cols
numerical_pipeline = Pipeline([("impute", KNNImputer()), ("scale", RobustScaler())])

# full preprocessing pipeline
preprocessing_pipeline = ColumnTransformer(
    [("numerical", numerical_pipeline, NUMERICAL_COLS)]
)

models = {
    # base models
    "dummy": DummyClassifier(strategy="stratified"),
    "logres": LogisticRegression(max_iter=10_000),
    "sgd": SGDClassifier(),
    "svc": LinearSVC(dual="auto"),
    # "knn": KNeighborsClassifier(),
    "dt": DecisionTreeClassifier(),
    "ada": AdaBoostClassifier(),
    "gb": GradientBoostingClassifier(),
    "rf": RandomForestClassifier(),
    "xgb": XGBClassifier(),
    "lgb": LGBMClassifier(),
    "mlp": MLPClassifier(max_iter=10_000),
    "lgb_best": LGBMClassifier(
        **{
            "lambda_l1": 0.001749391593798918,
            "lambda_l2": 0.0001016528692597652,
            "num_leaves": 255,
            "feature_fraction": 0.48235552109791346,
            "bagging_fraction": 0.7627578419183237,
            "bagging_freq": 2,
            "min_child_samples": 43,
            "loss": "hinge",
            "penalty": None,
            "alpha": 2.4109193827960137e-05,
            "C": 0.6426793738292348,
            "solver": "sag",
        }
    ),
}
