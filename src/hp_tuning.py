import time

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from tabulate import tabulate

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import optuna

import config
import model
import module

x, y = module.load_train_dataset()


def objective(trial):
        clf_name = trial.suggest_categorical("classifier", ["lgb", "sgd", "logress"])
        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        sgd_params = {
            "max_iter": 10_000, 
            "verbose": 0, 
            "loss": trial.suggest_categorical("loss",
                ["hinge", "log_loss", "modified_huber",
                 "squared_hinge", "perceptron", "squared_error",
                 "huber", "epsilon_insensitive", "squared_epsilon_insensitive",
                ],
            ), 
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None]), 
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        }
        logress_params = {
            'C': trial.suggest_float('C', 0.00001, 10, log= True),
            'solver': trial.suggest_categorical('solver', [
                'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 
                'sag', 'saga'
            ]), 
        }

        if clf_name == "lgb":
            clf = LGBMClassifier(**lgb_params)
            clf_pipeline = Pipeline(
                [("preprocessing", model.preprocessing_pipeline), ("clf", clf)]
            )
        elif clf_name == "sgd":
            clf = SGDClassifier(**sgd_params)
            clf_pipeline = Pipeline(
                [("preprocessing", model.preprocessing_pipeline), ("clf", clf)]
            )
        elif clf_name == "logress":
            clf = LogisticRegression(**logress_params)
            clf_pipeline = Pipeline(
                [("preprocessing", model.preprocessing_pipeline), ("clf", clf)]
            )

        y_pred = cross_val_predict(
            estimator=clf_pipeline,
            X=x,
            y=y,
            cv=5,
            n_jobs=-1,
        )
        f1 = f1_score(y, y_pred)
        return f1


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    print(f"Best params: {study.best_params}")