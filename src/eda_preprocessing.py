import pandas as pd
import numpy as np

def load_clean_data(path):
    df = pd.read_csv(path, encoding="latin-1")

    df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
    df.columns = ["label", "message"]
    df = df.drop_duplicates()

    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    return df

