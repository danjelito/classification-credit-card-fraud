import time

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from tabulate import tabulate

import config
import model
import module


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        # return the original result along with execution time
        return result + (execution_time,)
    return wrapper


@timing_decorator
def train(x, y, clf_model):
    clf = model.models.get(clf_model)
    y_pred = cross_val_predict(
        estimator=clf,
        X=x,
        y=y,
        cv=5,
        n_jobs=-1,
    )
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    return acc, f1, roc_auc


if __name__ == "__main__":
    x, y = module.load_train_dataset()

    result_file = pd.read_csv(config.TRAIN_RESULT)
    trained_models = result_file.loc[:, "model"].values

    # train for each model if not yet trained
    for clf_model in model.models.keys():
        if clf_model not in trained_models:

            curr_time = time.strftime("%H:%M:%S", time.localtime())
            print(f"Training {clf_model} at {curr_time}.")
            
            scores = train(x, y, clf_model)  # contains scores
            
            # append model name and scores to result file
            # overwrite the result file with the new one
            result_file.loc[len(result_file.index)] = (clf_model,) + scores
            result_file.to_csv(config.TRAIN_RESULT, index=False)

    result_file = result_file.sort_values("acc", ascending=False, ignore_index=True)
    print(tabulate(result_file, headers="keys"))
