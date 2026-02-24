import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

# Project root: one level up from this file's directory (src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FIGURES_DIR = _PROJECT_ROOT / "figures"

# ── global style ─────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
font_title = {"family": "serif", "color": "#0b5394", "weight": "bold", "size": 14}

plt.style.use("fivethirtyeight")
plt.rcParams.update(
    {
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 14,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

# numeric feature names used for KDE / boxplot filtering
num_vars = [
    "altitude",
    "mean_topography_srtm_alt_90m_year1994",
    "mean_topography_srtm_alt_1km_year1994",
    "max_topography_srtm_relative_alt_5km_year1994",
    "min_topography_srtm_relative_alt_5km_year1994",
    "stddev_topography_srtm_relative_alt_5km_year1994",
    "distance_to_major_road_year2020",
    "mean_stable_nightlights_1km_year2013",
    "mean_stable_nightlights_5km_year2013",
    "max_stable_nightlights_25km_year2013",
    "max_stable_nightlights_25km_year1992",
    "mean_population_density_250m_year2015",
    "mean_population_density_5km_year2015",
    "max_population_density_25km_year2015",
    "mean_population_density_250m_year1990",
    "mean_population_density_5km_year1990",
    "max_population_density_25km_year1990",
    "mean_nox_emissions_10km_year2015",
    "mean_nox_emissions_10km_year2000",
]


# ── helpers ───────────────────────────────────────────────────────────────────
def _savefig(fig_path: str) -> None:
    """Create parent directories and save the current figure."""
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=400, bbox_inches="tight")


# ── plotting functions ────────────────────────────────────────────────────────

def plot_correlation(data, fig_name: str = "") -> None:
    """Plot a lower-triangle correlation heatmap for all numeric columns.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    fig_name : str
        Prefix appended to the saved file name (e.g. ``"eda_"`` →
        ``figures/eda_correlation.jpg``).
    """
    df = data.select_dtypes(include=["int64", "float64"])
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(22, 14))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 9},
        mask=mask,
        center=0,
        cmap="coolwarm",
        ax=ax,
    )
    plt.title("Linear correlation heatmap")
    fig_path = str(_FIGURES_DIR / (fig_name + "correlation.jpg"))
    _savefig(fig_path)
    plt.show()


def boxplot(df) -> None:
    """Display an interactive Plotly box-plot for every column in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Data to visualise (numeric columns recommended).
    """
    import plotly.express as px

    for col in df.columns:
        fig = px.box(df, y=col, width=400, height=400)
        fig.show()


def cf_matrix_plot(y_true, y_pred, fig_name: str = "confusion_matrix", title: str = "Confusion Matrix") -> None:
    """Pretty-print and save a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    fig_name : str
        Base file name (without extension) saved under ``figures/``.
    title : str
        Title of the confusion matrix plot.
    """
    print("Accuracy: {:.2f}%".format(accuracy_score(y_true, y_pred) * 100))
    cm = confusion_matrix(y_true, y_pred)
    class_labels = np.array(["rural", "suburban", "urban"])

    fig = plt.figure(figsize=(11, 9), facecolor="white")
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
    plt.title(title, fontdict=font_title)

    fig_path = str(_FIGURES_DIR / (fig_name + ".jpg"))
    _savefig(fig_path)
    plt.show()


def plot_count_kde(
    df,
    num_cols=None,
    cat_cols=None,
    plot_numeric: bool = True,
    plot_cat: bool = True,
    plot_target: bool = True,
    target: str = "type_of_area",
) -> None:
    """Plot histogram + KDE and boxplot for numeric features, count-plot for
    categorical features, and a count-plot for the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    num_cols : list, optional
        Numeric columns to plot.  Defaults to ``num_vars`` columns present in *df*.
    cat_cols : list, optional
        Categorical columns to plot.  Defaults to all object columns except
        ``area_code``.
    plot_numeric : bool
        Whether to plot numeric features.
    plot_cat : bool
        Whether to plot categorical features.
    plot_target : bool
        Whether to plot the target variable.
    target : str
        Name of the target column.
    """
    if plot_numeric:
        if num_cols is None:
            num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c in num_vars]
        for col in num_cols:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            sns.histplot(df[col], kde=True, bins=10, color="skyblue")
            plt.title(f"{col} - Histogram & KDE")
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col], color="lightgreen")
            plt.title(f"{col} - Boxplot")
            plt.tight_layout()
            fig_path = str(_FIGURES_DIR / f"{col}.jpg")
            _savefig(fig_path)
            plt.show()
        print("✅ Successfully plotted numeric features!")

    if plot_cat:
        if cat_cols is None:
            cat_cols = [
                c for c in df.select_dtypes(include=["object"]).columns if c != "area_code"
            ]
        for col in cat_cols:
            plt.figure(figsize=(8, 4))
            sns.countplot(x=col, data=df, palette="pastel")
            plt.title(f"{col} - Countplot")
            plt.xticks(rotation=45)
            fig_path = str(_FIGURES_DIR / f"{col}.jpg")
            _savefig(fig_path)
            plt.show()
        print("✅ Successfully plotted categorical features!")

    if plot_target:
        if target in df.columns:
            ax = sns.countplot(data=df, x=target)
            plt.title(f"Value Counts of {target}")
            plt.xlabel("Labels")
            plt.ylabel("Counts")
            for p in ax.patches:
                ax.annotate(
                    format(p.get_height(), ".0f"),
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                    xytext=(0, 5),
                    textcoords="offset points",
                )
            fig_path = str(_FIGURES_DIR / f"value_counts_{target}.jpg")
            _savefig(fig_path)
            plt.show()
            print(f"✅ Successfully plotted target variable '{target}'!")
        else:
            print(f"Warning: '{target}' not in df.columns!")


def plot_clusters(
    trained_model,
    df: pd.DataFrame,
    y_pred: np.ndarray | None = None,
    exclude_cols: list[str] | None = None,
    labels_map: dict = None,
    save_path: str = None,
) -> None:
    """Visualise cluster assignments in a 2-D PCA space.

    A fresh 2-component PCA is always fitted on the numeric features of *df*
    for visualisation so the axis labels can report the explained variance of
    each component.

    Parameters
    ----------
    trained_model : TOARClustering or similar
        A fitted clustering model that exposes a ``predict(X)`` method.
        Used to compute *y_pred* when it is not supplied, and to plot
        centroids when the underlying model has ``cluster_centers_``.
    df : pd.DataFrame
        Feature DataFrame.  Non-predictor columns listed in *exclude_cols*
        are dropped before PCA projection and prediction.
    y_pred : array-like, optional
        Integer cluster labels for each row of *df*.  If ``None``, labels
        are computed as ``trained_model.predict(X)``.
    exclude_cols : list[str], optional
        Columns to drop from *df* before prediction / projection.
        Defaults to ``['area_code', 'type_of_area', 'type_of_area_gmap']``.
    labels_map : dict, optional
        Mapping from cluster integer to human-readable string,
        e.g. ``{0: 'suburban', 1: 'rural', 2: 'urban'}``.
    save_path : str, optional
        File path to save the figure (uses ``_savefig``); skipped when ``None``.
    """
    if exclude_cols is None:
        exclude_cols = ["area_code", "type_of_area", "type_of_area_gmap"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]

    if y_pred is None:
        y_pred = trained_model.predict(X)

    # Always use a fresh 2-D PCA so we can report explained variance
    pca_2d = PCA(n_components=2)
    data_2d = pca_2d.fit_transform(X)
    explained_variance = pca_2d.explained_variance_ratio_

    colors = ["darkblue", "purple", "darkorange", "green", "red", "cyan"]
    unique_clusters = np.unique(y_pred)
    cmap = ListedColormap(colors[: len(unique_clusters)])

    plt.figure(figsize=(10, 8), facecolor="white")
    plt.scatter(
        data_2d[:, 0], data_2d[:, 1],
        c=y_pred, s=85, cmap=cmap, alpha=0.7, edgecolors="w", linewidth=0.5,
    )

    # Plot centroids if the model exposes them (KMeans)
    _model = getattr(trained_model, "model_", trained_model)
    if hasattr(_model, "cluster_centers_"):
        centers = _model.cluster_centers_
        centers_2d = pca_2d.transform(centers)
        plt.scatter(
            centers_2d[:, 0], centers_2d[:, 1],
            c="red", marker="X", s=200, alpha=1.0,
            label="Centroids", edgecolors="black", linewidth=1,
        )

    plt.xlabel(f"Principal Component 1 ({explained_variance[0]:.2%} variance)")
    plt.ylabel(f"Principal Component 2 ({explained_variance[1]:.2%} variance)")
    plt.title("Clustering: Rural, Urban, and Suburban Areas")
    if labels_map:
        legend_elements = [
            plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=colors[i % len(colors)],
                markersize=10, label=labels_map[cluster_idx],
            )
            for i, cluster_idx in enumerate(sorted(unique_clusters))
        ]
        legend_elements.append(
            plt.Line2D([0], [0], marker="X", color="w",
                       markerfacecolor="red", markersize=10, label="Centroids")
        )
        plt.legend(handles=legend_elements, loc="best")
    else:
        plt.legend(loc="best", title="Clusters")

    x_min, x_max = data_2d[:, 0].min(), data_2d[:, 0].max()
    y_min, y_max = data_2d[:, 1].min(), data_2d[:, 1].max()
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor("white")
    plt.tight_layout()

    if save_path:
        _savefig(save_path) 
    plt.show()


def plot_feature_importance(trained_clf, model_name: str = "", save_plot: bool = False) -> None:
    """Bar chart of feature importances for tree-based classifiers.

    Parameters
    ----------
    trained_clf : fitted sklearn estimator
        Must expose ``feature_importances_`` and ``feature_names_in_``
        (or ``feature_names_``).
    model_name : str
        Label used in the title and (optionally) the saved file name.
    save_plot : bool
        If ``True``, save to ``figures/features_importances_<model_name>.jpg``.
    """
    if hasattr(trained_clf, "feature_names_in_"):
        features = list(trained_clf.feature_names_in_)
    elif hasattr(trained_clf, "feature_names_"):
        features = list(trained_clf.feature_names_)
    else:
        raise AttributeError("Classifier does not expose feature name attributes.")

    import pandas as pd

    feature_rank = pd.DataFrame(
        {"features": features, "importance": trained_clf.feature_importances_}
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(12, 12), facecolor="white")
    sns.barplot(y="features", x="importance", data=feature_rank)
    plt.title(f"Feature importance – {model_name}", size=12)
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor("white")
    plt.tight_layout()

    if save_plot:
        fig_path = str(_FIGURES_DIR / f"features_importances_{model_name}.jpg")
        _savefig(fig_path)
    plt.show()


def plot_data_distribution_bar(
    data_fe,
    df_train,
    df_test,
    target: str = "type_of_area",
    categories: list = None,
    save_plot: bool = True,
) -> None:
    """Grouped bar chart showing class counts in full, train, and test data.

    Parameters
    ----------
    data_fe : pd.DataFrame
        Full (feature-engineered) dataset.
    df_train : pd.DataFrame
        Training set (must contain *target* column).
    df_test : pd.DataFrame
        Test set.
    target : str
        Column name for the class labels.
    categories : list, optional
        Ordered list of category labels for the x-axis. If ``None``,
        defaults to ``['unknown', 'urban', 'suburban', 'rural']``.
    save_plot : bool
        Save to ``figures/data_distribution_bar.jpg`` when ``True``.
    """
    if categories is None:
        categories = ["unknown", "urban", "suburban", "rural"]

    full_counts = data_fe[target].value_counts().reindex(categories)
    counts_train = df_train[target].value_counts().reindex(categories)
    counts_test = df_test[target].value_counts().reindex(categories)

    plt.figure(figsize=(12, 6), facecolor="white")

    bar_width = 0.3
    x_pos = np.arange(len(categories))

    bars1 = plt.bar(x_pos - bar_width, full_counts.values, width=bar_width, label="Full Data")
    bars2 = plt.bar(x_pos, counts_train.values, width=bar_width, label="Training Data", alpha=0.85, color="coral")
    bars3 = plt.bar(x_pos + bar_width, counts_test.values, width=bar_width, label="Test Data", alpha=0.7, color="green")

    plt.xlabel("Area Categories")
    plt.ylabel("Counts")
    plt.title(f"Distribution of {target} in datasets")
    plt.xticks(x_pos, categories)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.gca().set_facecolor("white")

    # add count labels on top of each bar
    def _add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 5,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    _add_value_labels(bars1)
    _add_value_labels(bars2)
    _add_value_labels(bars3)

    plt.tight_layout()
    if save_plot:
        fig_path = str(_FIGURES_DIR / "data_distribution_bar.jpg")
        _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=600, transparent=False, facecolor="white", bbox_inches="tight")
    plt.show()


def plot_data_distribution_map(
    df_train,
    df_test,
    df_val=None,
    save_plot: bool = True,
) -> None:
    """World map (Plate Carrée projection) showing the geographic distribution
    of train, test, and optionally val/unlabeled station locations.

    The DataFrames must have a MultiIndex of ``(lat, lon)`` or contain
    ``lat`` / ``lon`` columns.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training set.
    df_test : pd.DataFrame
        Test set.
    df_val : pd.DataFrame, optional
        Validation / unlabeled set. Skipped when ``None``.
    save_plot : bool
        Save to ``figures/data_distribution_map.jpg`` when ``True``.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    def _coords(df):
        if "lat" in df.columns and "lon" in df.columns:
            return df["lat"].tolist(), df["lon"].tolist()
        # MultiIndex (lat, lon)
        return (
            list(df.index.get_level_values("lat")),
            list(df.index.get_level_values("lon")),
        )

    lat_train, lon_train = _coords(df_train)
    lat_test, lon_test = _coords(df_test)

    plt.figure(figsize=(14, 7), facecolor="white")
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    if df_val is not None:
        lat_val, lon_val = _coords(df_val)
        ax.scatter(lon_val, lat_val, s=10, color="orange", alpha=0.7,
                   label="Val / Unlabeled", transform=ccrs.PlateCarree())

    ax.scatter(lon_train, lat_train, s=10, color="blue", alpha=0.7,
               label="Train", transform=ccrs.PlateCarree())
    ax.scatter(lon_test, lat_test, s=10, color="red", alpha=0.7,
               label="Test", transform=ccrs.PlateCarree())

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Distribution of train, test and predict station locations", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor("white")
    plt.tight_layout()

    if save_plot:
        fig_path = str(_FIGURES_DIR / "data_distribution_map.jpg")
        _savefig(fig_path)
    plt.show()

if __name__ == "__main__":
    print("This module contains plotting functions for EDA and model evaluation. "
          "Import and call the desired functions from your notebook or script.")