import pandas as pd
import config

def load_train_dataset():
    dataset = pd.read_parquet(config.TRAIN_FILEPATH)
    x = dataset.drop(columns="target").copy()
    y = dataset["target"].copy()
    return x, y
