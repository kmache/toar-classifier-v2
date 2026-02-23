"""
evaluator.py
============
Centralises all evaluation utilities for the TOAR Classifier V2 project.

Functions
---------
Clustering evaluation:
  ari_nmi_clustering(y_true, y_pred)
      Compute Adjusted Rand Index (ARI) and Normalised Mutual Information (NMI).

  evaluate_clustering(X, y_pred, y_true=None)
      Return a dict with silhouette (always) plus ARI/NMI when y_true is given.

  clustering_evaluation(df_to_pred, clustering_model, labels_map, ...)
      Full clustering report printed to stdout; returns a labelled DataFrame.

Classifier evaluation:
  statistics_evaluation(df_spice, clf, spice)
      Box-plot comparison of a pollution species split by predicted station type.

  feature_importance(trained_clf) -> pd.DataFrame
      Return a sorted feature-importance DataFrame for a tree-based classifier.

  classifier_evaluation(df_to_pred, trained_clfs, apply_threshold=False)
      Full per-model classification report; returns a DataFrame with predictions.
      (Renamed from ``full_classification_report`` in the notebook.)
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)


# ---------------------------------------------------------------------------
# Private helpers (ported / cleaned up from the notebook)
# ---------------------------------------------------------------------------

def _get_features_from_clf(clf) -> list:
    """Extract the feature names a fitted sklearn-compatible classifier was trained on."""
    if hasattr(clf, "feature_names_in_"):
        return list(clf.feature_names_in_)
    if hasattr(clf, "feature_names_"):
        return list(clf.feature_names_)
    raise AttributeError(
        f"Classifier {type(clf).__name__!r} exposes no known feature-name attribute "
        "('feature_names_in_' or 'feature_names_')."
    )

def _predict(clf, X: pd.DataFrame) -> np.ndarray:
    """Return class predictions, selecting only the features the classifier was trained on."""
    features = _get_features_from_clf(clf)
    return clf.predict(X[features]).reshape(len(X),)

def _predict_proba(clf, X: pd.DataFrame) -> np.ndarray:
    """Return class-probability estimates."""
    features = _get_features_from_clf(clf)
    return clf.predict_proba(X[features])

def _threshold_clf(y_proba: np.ndarray, thd: float = 0.5) -> np.ndarray:
    """Assign *suburban* when the winning class probability is below *thd*."""
    y_pred = []
    for proba in y_proba:
        p_max = max(proba)
        if p_max == proba[1]:
            y_pred.append("suburban")
        elif p_max == proba[0] and p_max >= thd:
            y_pred.append("rural")
        elif p_max == proba[2] and p_max >= thd:
            y_pred.append("urban")
        else:
            y_pred.append("suburban")
    return np.asarray(y_pred, dtype=object)


def _grid_search_threshold_clf(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "balanced_accuracy",
    n_thresholds: int = 50,
):
    """Grid-search the best confidence threshold for :func:`_threshold_clf`.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_proba : ndarray of shape (n_samples, n_classes)
        Class probability estimates from ``predict_proba``.
    metric : {'f1_macro', 'balanced_accuracy'}
        Scoring metric to maximise.
    n_thresholds : int
        Number of threshold values to try in [0.35, 0.65].

    Returns
    -------
    best_thd : float
    best_score : float
    best_y_pred : ndarray
    """
    thresholds = np.linspace(0.35, 0.65, n_thresholds)
    best_thd, best_score, best_y_pred = 0.5, -1.0, None
    for t in thresholds:
        y_pred = _threshold_clf(y_proba, thd=t)
        if metric == "f1_macro":
            score = f1_score(y_true, y_pred, average="macro")
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError("metric must be 'f1_macro' or 'balanced_accuracy'.")
        if score > best_score:
            best_score, best_thd, best_y_pred = score, t, y_pred
    return best_thd, best_score, best_y_pred

def _cluster_predict(X: pd.DataFrame, clustering_model) -> np.ndarray:
    """Run inference with *clustering_model*.

    Accepts either:
    * the legacy notebook dict  ``{'model': ..., 'pca': ..., 'use_pca': ...}``
    * any object with a ``.predict()`` method (e.g. ``TOARClustering``).
    """
    if isinstance(clustering_model, dict):
        model = clustering_model["model"]
        pca = clustering_model.get("pca")
        X_transformed = pca.transform(X) if pca is not None else X
        if hasattr(model, "predict"):
            return model.predict(X_transformed)
        return model.predict_proba(X_transformed).argmax(axis=1)
    # Class-based API (TOARClustering)
    return clustering_model.predict(X)


def _cf_matrix(y_true, y_pred, fig_name: str = "_") -> None:
    """Plot and save a labelled confusion-matrix heatmap."""
    print("Accuracy: {:.2f}%".format(accuracy_score(y_true, y_pred) * 100))
    cm = confusion_matrix(y_true, y_pred)
    class_labels = np.array(["rural", "suburban", "urban"])
    plt.figure(figsize=(11, 9), facecolor="white")
    sns.set(font_scale=1.4)
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="g",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel("Predicted classes")
    plt.ylabel("True classes")
    plt.title("Confusion Matrix")
    os.makedirs("figures", exist_ok=True)
    fig_path = os.path.abspath(os.path.join("figures", fig_name + ".jpg"))
    plt.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 1. Clustering helpers
# ---------------------------------------------------------------------------

def ari_nmi_clustering(y_true, y_pred) -> tuple:
    """Compute Adjusted Rand Index (ARI) and Normalised Mutual Information (NMI).

    Parameters
    ----------
    y_true : array-like
        Ground-truth class labels.
    y_pred : array-like
        Predicted cluster assignments.

    Returns
    -------
    ari : float
    nmi : float
    """
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return ari, nmi


def evaluate_clustering(X, y_pred, y_true=None) -> dict:
    """Compute clustering evaluation metrics.

    Always computes the silhouette score (unsupervised).  When *y_true* is
    supplied, ARI and NMI are added as well.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix used for clustering.
    y_pred : array-like of shape (n_samples,)
        Predicted cluster labels.
    y_true : array-like of shape (n_samples,), optional
        Ground-truth labels for supervised metrics.

    Returns
    -------
    metrics : dict
        Keys: ``'silhouette'`` always; ``'ari'`` and ``'nmi'`` when y_true given.
    """
    metrics = {"silhouette": silhouette_score(X, y_pred)}
    if y_true is not None:
        metrics["ari"] = adjusted_rand_score(y_true, y_pred)
        metrics["nmi"] = normalized_mutual_info_score(y_true, y_pred)
    return metrics


def clustering_evaluation(
    df_to_pred: pd.DataFrame,
    clustering_model,
    labels_map: dict = None,
    evaluate_on_hand_label_data: bool = False,
    cm_plot: bool = False,
) -> pd.DataFrame:
    """Run a full clustering evaluation report.

    Prints accuracy, balanced accuracy, ARI, NMI, and a sklearn classification
    report to stdout.  Optionally shows and saves a confusion-matrix figure.

    Parameters
    ----------
    df_to_pred : pd.DataFrame
        DataFrame containing at minimum ``'area_code'`` and ``'type_of_area'``
        columns plus all predictor features.  When *evaluate_on_hand_label_data*
        is True, ``'type_of_area_gmap'`` must also be present.
    clustering_model : dict or TOARClustering
        Fitted clustering model – either the legacy ``{model, pca, use_pca}``
        dict or a ``TOARClustering`` instance.
    labels_map : dict
        Mapping from integer cluster labels to string class names,
        e.g. ``{0: 'urban', 1: 'suburban', 2: 'rural'}``.
    evaluate_on_hand_label_data : bool
        If True, also evaluate against the ``'type_of_area_gmap'`` column.
    cm_plot : bool
        If True, display and save a confusion-matrix figure.

    Returns
    -------
    pd.DataFrame
        Copy of *df_to_pred* with columns ``['area_code', 'type_of_area',
        'type_of_area_pred']``.
    """
    if labels_map is None:
        raise ValueError("labels_map must be provided.")

    map_func = np.vectorize(labels_map.get)
    exclude_cols = ["area_code", "type_of_area", "type_of_area_toar", "type_of_area_gmap"]
    predictor_cols = [c for c in df_to_pred.columns if c not in exclude_cols]
    X_to_pred = df_to_pred[predictor_cols]
    Y_true = df_to_pred["type_of_area"].values
    Y_pred = map_func(_cluster_predict(X_to_pred, clustering_model))

    acc = accuracy_score(Y_true, Y_pred)
    bal_acc = balanced_accuracy_score(Y_true, Y_pred)
    ari, nmi = ari_nmi_clustering(Y_true, Y_pred)

    print()
    print("global accuracy: ", acc)
    print("Balanced accuracy score:", bal_acc)
    print(f"ARI-score: {ari}")
    print(f"NMI-score: {nmi}")
    print()
    print("classification report")
    print(classification_report(Y_true, Y_pred, digits=4))

    if cm_plot:
        _cf_matrix(Y_true, Y_pred)

    if evaluate_on_hand_label_data and "type_of_area_gmap" in df_to_pred.columns:
        print("Evaluation on hand-labelled data")
        Y_true_gmap = df_to_pred["type_of_area_gmap"].values
        acc_gmap = accuracy_score(Y_true_gmap, Y_pred)
        bal_acc_gmap = balanced_accuracy_score(Y_true_gmap, Y_pred)
        ari_gmap, nmi_gmap = ari_nmi_clustering(Y_true_gmap, Y_pred)
        print()
        print("global accuracy: ", acc_gmap)
        print("Balanced accuracy score:", bal_acc_gmap)
        print(f"ARI-score: {ari_gmap}")
        print(f"NMI-score: {nmi_gmap}")
        print()
        print("classification report")
        print(classification_report(Y_true_gmap, Y_pred, digits=4))
        if cm_plot:
            _cf_matrix(Y_true_gmap, Y_pred, fig_name="km_hand_labeled")

    df_result = df_to_pred[["area_code", "type_of_area"]].copy()
    df_result["type_of_area_pred"] = Y_pred
    return df_result


# ---------------------------------------------------------------------------
# 2. Statistics evaluation
# ---------------------------------------------------------------------------

def statistics_evaluation(df_spice: pd.DataFrame, clf, spice: str) -> None:
    """Evaluate a classifier via box-plot comparison of a pollution species.

    Predicts station types with *clf*, then plots the distribution of *spice*
    (e.g. ``'nox'``, ``'no2'``, ``'pm2p5'``) across the three predicted
    station-type groups to provide a physical-plausibility sanity check.

    Parameters
    ----------
    df_spice : pd.DataFrame
        DataFrame containing the features expected by *clf* **and** a column
        named after *spice* plus a ``'type_of_area'`` column for accuracy
        reporting.
    clf : fitted sklearn-compatible classifier
        Must expose ``predict()`` and ``feature_names_in_`` / ``feature_names_``.
    spice : str
        Column name of the species to plot (e.g. ``'nox'``).
    """
    features = _get_features_from_clf(clf)
    y_pred = clf.predict(df_spice[features])
    df_plot = df_spice.copy()
    df_plot["type_of_area_pred"] = y_pred

    if "type_of_area" in df_plot.columns:
        acc = accuracy_score(df_plot["type_of_area"].values, y_pred)
        print(f"accuracy: {acc:.3f}")

    data_clf = [
        df_plot.loc[df_plot["type_of_area_pred"] == "urban",    spice].values,
        df_plot.loc[df_plot["type_of_area_pred"] == "suburban", spice].values,
        df_plot.loc[df_plot["type_of_area_pred"] == "rural",    spice].values,
    ]

    plt.figure(figsize=(9, 8), facecolor="white")
    plt.boxplot(
        data_clf,
        patch_artist=True,
        notch=False,
        vert=True,
        showmeans=False,
        widths=0.15,
        positions=[0.25, 0.75, 1.25],
    )
    plt.title(f"Statistics evaluation – {spice}")
    plt.xlabel("Station location")
    plt.ylabel(f"p75 of {spice}")
    plt.xticks([0.25, 0.75, 1.25], ["urban", "suburban", "rural"])
    plt.grid(True, alpha=0.3)
    os.makedirs("figures", exist_ok=True)
    plt.savefig(
        os.path.join("figures", f"box_{spice}.jpg"), dpi=400, bbox_inches="tight"
    )
    plt.show()


# ---------------------------------------------------------------------------
# 3. Feature importance
# ---------------------------------------------------------------------------

def feature_importance(trained_clf) -> pd.DataFrame:
    """Compute feature importance for a fitted tree-based classifier.

    Parameters
    ----------
    trained_clf : fitted sklearn-compatible classifier
        Must expose ``feature_importances_`` and ``feature_names_in_`` /
        ``feature_names_``.

    Returns
    -------
    pd.DataFrame
        Columns: ``['feature', 'importance']``, sorted by importance descending.
    """
    features = _get_features_from_clf(trained_clf)
    importances = trained_clf.feature_importances_
    rank = pd.DataFrame({"feature": features, "importance": importances})
    return rank.sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Full classifier evaluation (renamed from full_classification_report)
# ---------------------------------------------------------------------------

def classifier_evaluation(
    df_to_pred: pd.DataFrame,
    trained_clfs: dict,
    apply_threshold: bool = False,
) -> pd.DataFrame:
    """Run a full multi-model classification evaluation report.

    Prints per-model accuracy (global and per class), sklearn classification
    report, and balanced accuracy to stdout.  A majority-vote prediction column
    is also appended to the returned DataFrame.

    Parameters
    ----------
    df_to_pred : pd.DataFrame
        Labelled DataFrame with at least ``'area_code'`` and ``'type_of_area'``
        columns plus all predictor features.
    trained_clfs : dict
        ``{model_name: fitted_classifier}`` mapping, e.g.
        ``{'rf': rf_clf, 'cboost': cboost_clf, 'lgbm': lgbm_clf}``.
    apply_threshold : bool
        If True, apply a grid-searched confidence threshold (may improve macro-F1
        on imbalanced data).

    Returns
    -------
    pd.DataFrame
        Copy of *df_to_pred* restricted to ``['area_code', 'type_of_area']``
        plus one prediction column per model and a
        ``'type_of_area_pred_voting'`` majority-vote ensemble column.
    """
    X_to_pred = df_to_pred.drop(columns=["area_code", "type_of_area"])
    Y_true = df_to_pred["type_of_area"].values
    df_result = df_to_pred[["area_code", "type_of_area"]].copy()
    pred_cols = []

    for model_name, clf in trained_clfs.items():
        y_pred = _predict(clf, X_to_pred)
        if apply_threshold:
            y_proba = _predict_proba(clf, X_to_pred)
            best_thd, best_score, best_y_pred = _grid_search_threshold_clf(
                Y_true, y_proba, metric="f1_macro", n_thresholds=50
            )
            y_pred = best_y_pred
            print(
                f"best threshold for {model_name}: {best_thd:.3f}, "
                f"best f1_macro: {best_score:.4f}"
            )
        col = f"type_of_area_pred_{model_name}"
        df_result[col] = y_pred
        pred_cols.append(col)

    # Majority-vote ensemble
    df_result["type_of_area_pred_voting"] = df_result[pred_cols].apply(
        lambda row: Counter(row).most_common(1)[0][0], axis=1
    )

    # Per-class masks for targeted accuracy
    urban_mask    = df_result["type_of_area"] == "urban"
    suburban_mask = df_result["type_of_area"] == "suburban"
    rural_mask    = df_result["type_of_area"] == "rural"

    for model_name, clf in trained_clfs.items():
        col = f"type_of_area_pred_{model_name}"
        acc          = accuracy_score(df_result["type_of_area"],           df_result[col])
        acc_urban    = accuracy_score(df_result.loc[urban_mask,    "type_of_area"], df_result.loc[urban_mask,    col])
        acc_suburban = accuracy_score(df_result.loc[suburban_mask, "type_of_area"], df_result.loc[suburban_mask, col])
        acc_rural    = accuracy_score(df_result.loc[rural_mask,    "type_of_area"], df_result.loc[rural_mask,    col])
        bal_acc      = balanced_accuracy_score(df_result["type_of_area"], df_result[col])

        print()
        print("=" * 52)
        print(f"  Model: {model_name}")
        print("=" * 52)
        print(f"  global accuracy:                {acc:.4f}")
        print(f"  accuracy for predicting urban:    {acc_urban:.4f}")
        print(f"  accuracy for predicting suburban: {acc_suburban:.4f}")
        print(f"  accuracy for predicting rural:    {acc_rural:.4f}")
        print()
        print("  classification report")
        print(classification_report(
            df_result["type_of_area"], df_result[col],
            target_names=clf.classes_, digits=4,
        ))
        print(f"  Balanced accuracy score: {bal_acc:.4f}")
        print()

    print("✅ Successfully evaluated trained models!")
    return df_result
if __name__ == "__main__":
    print("This module defines evaluation utilities for clustering and classification.")

