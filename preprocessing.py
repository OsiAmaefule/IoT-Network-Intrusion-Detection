"""
UNR-IDD dataset preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the UNR-IDD dataset from CSV."""
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    print(f"Dataset loaded: {df.shape[0]} samples")
    print(f"Class distribution:\n{df['Label'].value_counts().sort_index()}")
    return df


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Preprocess the UNR-IDD dataset.
    Returns X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # Drop label columns and zero-variance columns
    drop_cols = [
        "Label", "Binary Label",
        "Packets Rx Dropped", "Packets Tx Dropped",
        "Packets Rx Errors", "Packets Tx Errors",
        "Table ID", "Max Size"
    ]
    X = df.drop(columns=drop_cols).copy()
    y = df["Label"].copy()

    # Encode string and bool columns
    for col in X.select_dtypes(include=["object", "bool"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names, index=X_test.index
    )

    print(f"Features: {len(feature_names)}")
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test, scaler, feature_names