"""
Explainable AI (XAI) analysis module using SHAP and LIME.

Provides functions for global/local explanations of trained classifiers
on the UNR-IDD intrusion detection dataset.
"""

import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


CLASS_NAMES = ["Normal", "TCP-SYN", "PortScan", "Overflow", "Blackhole", "Diversion"]


def compute_shap_tree(model, X_test, model_name: str, save_dir: str = "results"):
    """Compute SHAP values using TreeExplainer (for RF, DT, AdaBoost)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"shap_values_{model_name}.npy"), shap_values)
    print(f"SHAP values computed and saved for {model_name}")
    return explainer, shap_values


def compute_shap_linear(model, X_train, X_test, model_name: str, save_dir: str = "results"):
    """Compute SHAP values using LinearExplainer (for LR)."""
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"shap_values_{model_name}.npy"), shap_values)
    print(f"SHAP values computed and saved for {model_name}")
    return explainer, shap_values


def compute_shap_kernel(model, X_train, X_test, model_name: str,
                        n_background: int = 100, n_explain: int = 300,
                        save_dir: str = "results"):
    """Compute SHAP values using KernelExplainer (for KNN). Uses sampling for speed."""
    background = shap.sample(X_train, n_background)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_test[:n_explain])
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"shap_values_{model_name}.npy"), shap_values)
    print(f"SHAP values computed and saved for {model_name} ({n_explain} samples)")
    return explainer, shap_values


def plot_shap_global(shap_values, X_test, model_name: str, fig_dir: str = "figures"):
    """Generate and save SHAP global bar plot and beeswarm plot."""
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar",
                      class_names=CLASS_NAMES, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"shap_global_bar_{model_name}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test, class_names=CLASS_NAMES, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"shap_beeswarm_{model_name}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    print(f"SHAP global plots saved for {model_name}")


def plot_shap_per_class(shap_values, X_test, model_name: str, fig_dir: str = "figures"):
    """Generate per-class SHAP bar plots (one subplot per attack type)."""
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, (ax, class_name) in enumerate(zip(axes.flatten(), CLASS_NAMES)):
        top_features = np.argsort(np.abs(shap_values[idx]).mean(axis=0))[-10:]
        feature_names = X_test.columns[top_features]
        feature_importance = np.abs(shap_values[idx]).mean(axis=0)[top_features]

        ax.barh(range(len(feature_names)), feature_importance)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=8)
        ax.set_title(f"Class {idx}: {class_name}", fontsize=10)
        ax.set_xlabel("Mean |SHAP value|", fontsize=8)

    plt.suptitle(f"Per-Class Feature Importance — {model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"shap_per_class_{model_name}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Per-class SHAP plots saved for {model_name}")


def get_top_features_per_class(shap_values, feature_names, top_n: int = 5) -> pd.DataFrame:
    """Return a DataFrame with top-N features per class ranked by mean |SHAP|."""
    rows = []
    for idx, class_name in enumerate(CLASS_NAMES):
        importances = np.abs(shap_values[idx]).mean(axis=0)
        top_idx = np.argsort(importances)[-top_n:][::-1]
        for rank, fi in enumerate(top_idx, 1):
            rows.append({
                "Class": class_name,
                "Rank": rank,
                "Feature": feature_names[fi],
                "Mean |SHAP|": round(importances[fi], 4),
            })
    return pd.DataFrame(rows)


def explain_with_lime(model, X_train, X_test, sample_idx: int,
                      feature_names: list, model_name: str,
                      fig_dir: str = "figures"):
    """Generate and save a LIME explanation for a single sample."""
    os.makedirs(fig_dir, exist_ok=True)

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values if hasattr(X_train, "values") else X_train,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        mode="classification",
    )

    exp = lime_explainer.explain_instance(
        X_test.iloc[sample_idx].values if hasattr(X_test, "iloc") else X_test[sample_idx],
        model.predict_proba,
        num_features=10,
        top_labels=3,
    )

    fig = exp.as_pyplot_figure(label=exp.top_labels[0])
    fig.savefig(
        os.path.join(fig_dir, f"lime_{model_name}_sample{sample_idx}.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    print(f"LIME explanation saved for {model_name}, sample {sample_idx}")
    return exp
