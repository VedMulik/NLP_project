
from datasets import load_dataset
import pandas as pd

LABEL_MAP = {
    "Clear Reply": 0,
    "Clear Non-Reply": 1,
    "Ambivalent": 2
}

def load_qevasion():
    """
    Loads QEvasion dataset and extracts:
        text  = interview_answer
        label = clarity_label (mapped to 0/1/2)
    """

    ds = load_dataset("ailsntua/QEvasion")

    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    train_df['clarity_label'] = train_df['clarity_label'].str.strip()
    test_df['clarity_label'] = test_df['clarity_label'].str.strip()

    train_df = train_df.rename(columns={"interview_answer": "text"})
    test_df = test_df.rename(columns={"interview_answer": "text"})

    train_df = train_df.dropna(subset=["text", "clarity_label"])
    test_df = test_df.dropna(subset=["text", "clarity_label"])

    train_df['label'] = train_df['clarity_label'].map(LABEL_MAP)
    test_df['label'] = test_df['clarity_label'].map(LABEL_MAP)

    train_df = train_df.dropna(subset=["label"])
    test_df = test_df.dropna(subset=["label"])

    train_df["label"] = train_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    print("\n=== QEvasion Dataset Loaded ===")
    print("Train:", train_df.shape)
    print("Test:", test_df.shape)
    print("\nLabel Distribution (Train):")
    print(train_df["label"].value_counts())
    print("\nLabel Distribution (Test):")
    print(test_df["label"].value_counts())

    return train_df, test_df


if __name__ == "__main__":
    load_qevasion()
