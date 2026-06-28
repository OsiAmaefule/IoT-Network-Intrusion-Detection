"""
Classifier definitions and hyperparameter search spaces
for the 5 CSA-optimized lightweight ML models.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier


CLASSIFIER_CONFIGS = {
    "KNN": {
        "class": KNeighborsClassifier,
        "param_grid": {
            "n_neighbors": {"type": "int", "low": 1, "high": 30},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
        },
    },
    "LR": {
        "class": LogisticRegression,
        "param_grid": {
            "C": {"type": "float", "low": 0.001, "high": 100.0},
            "max_iter": {"type": "int", "low": 100, "high": 1000},
        },
    },
    "RF": {
        "class": RandomForestClassifier,
        "param_grid": {
            "n_estimators": {"type": "int", "low": 50, "high": 300},
            "max_depth": {"type": "int", "low": 5, "high": 30},
            "min_samples_split": {"type": "int", "low": 2, "high": 10},
        },
    },
    "DT": {
        "class": DecisionTreeClassifier,
        "param_grid": {
            "max_depth": {"type": "int", "low": 3, "high": 30},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
        },
    },
    "AB": {
        "class": AdaBoostClassifier,
        "param_grid": {
            "n_estimators": {"type": "int", "low": 50, "high": 300},
            "learning_rate": {"type": "float", "low": 0.01, "high": 2.0},
        },
    },
}


def get_classifier_config(name: str) -> dict:
    """Return classifier class and param_grid by short name."""
    if name not in CLASSIFIER_CONFIGS:
        raise ValueError(f"Unknown classifier: {name}. Choose from {list(CLASSIFIER_CONFIGS.keys())}")
    return CLASSIFIER_CONFIGS[name]
