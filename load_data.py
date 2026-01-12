import os
import kagglehub
import pandas as pd

SMARTPHONE_DATASET = "garvchawla78/cleaned-smartphone-specs-dataset-2025-958-phones"
AMAZON_REVIEWS_DATASET = "PromptCloudHQ/amazon-reviews-unlocked-mobile-phones"


def _load_csv_from_path(path: str) -> pd.DataFrame:
    """Load the first CSV file found in the given directory."""
    csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
    return pd.read_csv(os.path.join(path, csv_file))


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load datasets (downloads if not cached, otherwise uses cache)."""
    path1 = kagglehub.dataset_download(SMARTPHONE_DATASET)
    path2 = kagglehub.dataset_download(AMAZON_REVIEWS_DATASET)

    df_smartphones = _load_csv_from_path(path1)
    df_amazon_reviews = _load_csv_from_path(path2)

    return df_smartphones, df_amazon_reviews


if __name__ == "__main__":
    df_smartphones, df_amazon_reviews = load_datasets()
    print("Smartphone Specs:")
    print(df_smartphones.head())
    print()
    print("Amazon Reviews:")
    print(df_amazon_reviews.head())
