"""
modeling.py
===========
Two main modeling classes for the TOAR Classifier V2 project:

  * TOARClustering  – Unsupervised learning via KMeans or GaussianMixture,
                      with optional PCA pre-processing.

  * TOARClassifier  – Supervised classification via RandomForest, CatBoost,
                      LightGBM, and XGBoost.

Both classes expose:
  fit()              – train the model(s)
  predict()          – class predictions
  predict_proba()    – class-probability estimates  (TOARClassifier only)
  threshold_predict() – confidence-threshold inference (TOARClassifier only)
  save()             – joblib.dump to <save_dir>/<model_name>_model.pkl
  load()             – class-method to restore from disk

Naming convention for saved files
----------------------------------
  KMeans          → kmean_model.pkl
  GaussianMixture → gmm_model.pkl
  RandomForest    → rf_model.pkl
  CatBoost        → catboost_model.pkl
  LightGBM        → lgbm_model.pkl
  XGBoost         → xgb_model.pkl
"""
import warnings
from collections import Counter
from pathlib import Path

import joblib

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    warnings.warn(
        "xgboost is not installed. The 'xgb' model will be unavailable. "
        "Install it with: pip install xgboost",
        ImportWarning,
        stacklevel=2,
    )

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)

__all__ = ["TOARClustering", "TOARClassifier"]
# ---------------------------------------------------------------------------
# Internal name → filename mapping
# ---------------------------------------------------------------------------
_CLUSTERING_SAVE_NAMES: dict[str, str] = {
    "kmeans": "kmean_model",
    "gmm":    "gmm_model",
}

_CLASSIFIER_SAVE_NAMES: dict[str, str] = {
    "rf":       "rf_model",
    "catboost": "catboost_model",
    "lgbm":     "lgbm_model",
    "xgb":      "xgb_model",
}


# ===========================================================================
# TOARClustering
# ===========================================================================

class TOARClustering:
    """
    Unsupervised clustering for TOAR station classification.

    Supports KMeans and Gaussian Mixture Models (GMM) with optional PCA
    pre-processing (retaining 97 % of variance by default).

    Parameters
    ----------
    model_type : str
        Clustering algorithm.  One of {'kmeans', 'gmm'}.
    num_clusters : int
        Number of clusters / components (default: 3).
    use_pca : bool
        If True, apply PCA before fitting the clustering model.
    **kwargs
        Extra keyword arguments forwarded to the underlying sklearn estimator.

    Attributes
    ----------
    model_ : fitted KMeans or GaussianMixture instance
    pca_ : fitted PCA instance or None
    is_fitted_ : bool
    """
    def __init__(
        self,
        model_type: str = "kmeans",
        num_clusters: int = 3,
        use_pca: bool = False,
        **kwargs,
    ) -> None:
        if model_type.lower() not in _CLUSTERING_SAVE_NAMES:
            raise ValueError(
                f"model_type must be one of {list(_CLUSTERING_SAVE_NAMES)}. "
                f"Got '{model_type}'."
            )
        if num_clusters < 2:
            raise ValueError("num_clusters must be at least 2.")

        self.model_type:   str  = model_type.lower()
        self.num_clusters: int  = num_clusters
        self.use_pca:      bool = use_pca
        self.kwargs:       dict = kwargs

        self.model_:     object | None = None
        self.pca_:       PCA | None    = None
        self.is_fitted_: bool          = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> object:
        """Instantiate (unfitted) the underlying sklearn estimator."""
        if self.model_type == "kmeans":
            return KMeans(
                n_clusters=self.num_clusters,
                n_init=self.kwargs.get("n_init", "auto"),
                init=self.kwargs.get("init", "k-means++"),
                random_state=42,
            )
        # gmm
        return GaussianMixture(
            n_components=self.num_clusters,
            random_state=42,
            covariance_type=self.kwargs.get("covariance_type", "full"),
            max_iter=self.kwargs.get("max_iter", 100),
        )

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply PCA projection if a fitted PCA is available."""
        if self.pca_ is not None:
            return self.pca_.transform(X)
        return np.asarray(X)

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "This TOARClustering instance is not fitted yet. "
                "Call fit() before using this method."
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "TOARClustering":  # noqa: F821
        """
        Fit the clustering model on X.

        Parameters
        ----------
        X : pd.DataFrame
            Pre-processed feature matrix (already scaled if desired).

        Returns
        -------
        self
        """
        if self.use_pca:
            self.pca_ = PCA(n_components=0.97, random_state=42)
            print(f"Original shape: {X.shape}")
            X_transformed = self.pca_.fit_transform(X)
            print(f"Shape after PCA: {X_transformed.shape}")
        else:
            X_transformed = np.asarray(X)

        self.model_ = self._build_model()
        self.model_.fit(X_transformed)
        self.is_fitted_ = True
        print(f"✅ {self.model_type.upper()} model trained successfully!")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster indices for X.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray of integer cluster indices.
        """
        self._check_fitted()
        X_t = self._transform(X)
        if hasattr(self.model_, "predict"):
            return self.model_.predict(X_t)
        # GMM fallback: argmax of responsibilities
        return self.model_.predict_proba(X_t).argmax(axis=1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_dir: str | Path = "models") -> Path:
        """
        Persist the fitted model to a .pkl file via joblib.

        Saved as  <save_dir>/<model_type>_model.pkl
        e.g.  models/kmean_model.pkl   or   models/gmm_model.pkl

        Parameters
        ----------
        save_dir : str | Path
            Directory to save into (created if it does not exist).

        Returns
        -------
        Path : absolute path to the saved file.
        """
        self._check_fitted()
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / (_CLUSTERING_SAVE_NAMES[self.model_type] + ".pkl")

        state = {
            "model_type":   self.model_type,
            "num_clusters": self.num_clusters,
            "use_pca":      self.use_pca,
            "kwargs":       self.kwargs,
            "model_":       self.model_,
            "pca_":         self.pca_,
        }
        joblib.dump(state, file_path)
        print(f"✅ {self.model_type.upper()} model saved to: {file_path.resolve()}")
        return file_path.resolve()

    @classmethod
    def load(cls, file_path: str | Path) -> "TOARClustering":
        """
        Load a previously saved TOARClustering model.

        Parameters
        ----------
        file_path : str | Path
            Path to the .pkl file produced by save().

        Returns
        -------
        TOARClustering (fitted instance).

        Example
        -------
        >>> clustering = TOARClustering.load('models/kmean_model.pkl')
        >>> labels = clustering.predict(X_test)
        """
        file_path = Path(file_path)
        state = joblib.load(file_path)

        instance = cls(
            model_type=state["model_type"],
            num_clusters=state["num_clusters"],
            use_pca=state["use_pca"],
            **state["kwargs"],
        )
        instance.model_     = state["model_"]
        instance.pca_       = state["pca_"]
        instance.is_fitted_ = True
        print(f"✅ {state['model_type'].upper()} model loaded from: {file_path.resolve()}")
        return instance


# ===========================================================================
# TOARClassifier  (supervised classification)
# ===========================================================================

class TOARClassifier:
    """
    Supervised classification for TOAR station type prediction.

    Wraps RandomForest, CatBoost, LightGBM, and XGBoost classifiers.
    Each model is trained independently; predictions can be obtained
    individually or via a majority-vote ensemble.

    Parameters
    ----------
    models : list of str, optional
        Which models to train.  Any subset of {'rf', 'catboost', 'lgbm', 'xgb'}.
        Defaults to all four (xgb excluded if xgboost is not installed).
    use_smote : bool
        If True, apply SMOTE oversampling before training (default: True).
    rf_params : dict, optional
        Keyword arguments forwarded to RandomForestClassifier.
    catboost_params : dict, optional
        Keyword arguments forwarded to CatBoostClassifier.
    lgbm_params : dict, optional
        Keyword arguments forwarded to LGBMClassifier.
    xgb_params : dict, optional
        Keyword arguments forwarded to XGBClassifier.

    Attributes
    ----------
    estimators_ : dict
        Fitted classifiers keyed by model name after calling fit().
    is_fitted_ : bool
    """

    _DEFAULT_MODELS: list[str] = (
        ["rf", "catboost", "lgbm", "xgb"] if _HAS_XGB
        else ["rf", "catboost", "lgbm"]
    )

    _DEFAULT_RF: dict = dict(n_estimators=500, random_state=42)

    _DEFAULT_CATBOOST: dict = dict(
        n_estimators=500,
        learning_rate=0.03986794927756705,
        depth=9,
        l2_leaf_reg=6,
        random_strength=3.4439060846939396,
        min_data_in_leaf=49,
        bootstrap_type="MVS",
        verbose=False,
    )

    _DEFAULT_LGBM: dict = dict(n_estimators=500, max_depth=20, verbose=0)

    _DEFAULT_XGB: dict = dict(
        n_estimators=500,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )

    def __init__(
        self,
        models: list[str] | None = None,
        use_smote: bool = True,
        rf_params:       dict | None = None,
        catboost_params: dict | None = None,
        lgbm_params:     dict | None = None,
        xgb_params:      dict | None = None,
    ) -> None:
        self.models    = [m.lower() for m in (models or self._DEFAULT_MODELS)]
        self.use_smote = use_smote

        self._params: dict[str, dict] = {
            "rf":       {**self._DEFAULT_RF,       **(rf_params       or {})},
            "catboost": {**self._DEFAULT_CATBOOST,  **(catboost_params or {})},
            "lgbm":     {**self._DEFAULT_LGBM,      **(lgbm_params     or {})},
            "xgb":      {**self._DEFAULT_XGB,       **(xgb_params      or {})},
        }

        self.estimators_: dict[str, object] = {}
        self.is_fitted_:  bool              = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_estimators(self) -> dict[str, object]:
        built: dict[str, object] = {}
        for name in self.models:
            if name == "rf":
                built["rf"] = RandomForestClassifier(**self._params["rf"])
            elif name == "catboost":
                built["catboost"] = CatBoostClassifier(**self._params["catboost"])
            elif name == "lgbm":
                built["lgbm"] = LGBMClassifier(**self._params["lgbm"])
            elif name == "xgb":
                if not _HAS_XGB:
                    raise ImportError(
                        "xgboost is not installed. "
                        "Install via: pip install xgboost"
                    )
                built["xgb"] = XGBClassifier(**self._params["xgb"])
            else:
                raise ValueError(
                    f"Unknown model '{name}'. "
                    f"Choose from {self._DEFAULT_MODELS}."
                )
        return built

    @staticmethod
    def _get_features(clf) -> list[str]:
        """Extract feature names stored inside a fitted estimator."""
        if hasattr(clf, "feature_names_in_"):
            return list(clf.feature_names_in_)
        if hasattr(clf, "feature_names_"):
            return list(clf.feature_names_)
        raise AttributeError(
            f"{type(clf).__name__} does not expose known feature-name attributes."
        )

    def _predict_one(self, clf, X: pd.DataFrame) -> np.ndarray:
        features = self._get_features(clf)
        return clf.predict(X[features]).reshape(-1)

    def _predict_proba_one(self, clf, X: pd.DataFrame) -> np.ndarray:
        features = self._get_features(clf)
        return clf.predict_proba(X[features])

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "This TOARClassifier instance is not fitted yet. "
                "Call fit() before using this method."
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train,
    ) -> "TOARClassifier":  # noqa: F821
        """
        Train all configured classifiers.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature matrix.
        y_train : array-like
            Target class labels.

        Returns
        -------
        self
        """
        estimators = self._build_estimators()

        if self.use_smote:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print("✅ SMOTE applied to balance classes!")

        for name, clf in estimators.items():
            clf.fit(X_train, y_train)
            print(f"✅ {name.upper()} trained successfully!")
            self.estimators_[name] = clf

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame,
        model: str | None = None,
    ) -> np.ndarray | dict[str, np.ndarray]:

        """
        Generate class predictions.

        Parameters
        ----------
        X : pd.DataFrame
        model : str, optional
            Name of a single model (e.g. 'rf').  When None, predictions from
            all models are returned together with a majority-vote ensemble.

        Returns
        -------
        np.ndarray when *model* is specified;
        dict {model_name: predictions, 'voting': voting_predictions} otherwise.
        """
        self._check_fitted()

        if model is not None:
            return self._predict_one(self.estimators_[model], X)

        preds = {
            name: self._predict_one(clf, X)
            for name, clf in self.estimators_.items()
        }
        # Majority-vote ensemble
        pred_matrix = np.column_stack(list(preds.values()))
        preds["voting"] = np.array(
            [Counter(row).most_common(1)[0][0] for row in pred_matrix],
            dtype=object,
        )
        return preds

    def predict_proba(
        self,
        X: pd.DataFrame,
        model: str | None = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """
        Class-probability estimates.

        Parameters
        ----------
        X : pd.DataFrame
        model : str, optional
            Single model name; returns a dict for all models when None.

        Returns
        -------
        (n_samples, n_classes) array when *model* is specified;
        dict {model_name: proba_array} otherwise.
        """
        self._check_fitted()

        if model is not None:
            return self._predict_proba_one(self.estimators_[model], X)
        return {
            name: self._predict_proba_one(clf, X)
            for name, clf in self.estimators_.items()
        }

    def threshold_predict(
        self,
        X: pd.DataFrame,
        thd: float = 0.5,
        model: str | None = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """
        Predict classes with a confidence threshold.

        Samples whose maximum class probability is below *thd* are assigned
        to 'suburban' (the most ambiguous class).

        Parameters
        ----------
        X : pd.DataFrame
        thd : float
            Minimum probability required to assign 'rural' or 'urban'.
        model : str, optional
            Single model; returns a dict for all when None.
        """
        self._check_fitted()
        classes = np.array(["rural", "suburban", "urban"], dtype=object)

        def _apply(y_proba: np.ndarray) -> np.ndarray:
            result = []
            for proba in y_proba:
                idx = int(np.argmax(proba))
                if idx == 1:                  # suburban
                    result.append("suburban")
                elif proba[idx] >= thd:
                    result.append(classes[idx])
                else:
                    result.append("suburban")
            return np.asarray(result, dtype=object)

        all_proba = self.predict_proba(X, model=model)
        if model is not None:
            return _apply(all_proba)
        return {name: _apply(proba) for name, proba in all_proba.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        model: str | None = None,
        save_dir: str | Path = "models",
    ) -> list[Path]:
        """
        Save one or all trained classifiers via joblib.

        File-naming convention:
          rf       → models/rf_model.pkl
          catboost → models/catboost_model.pkl
          lgbm     → models/lgbm_model.pkl
          xgb      → models/xgb_model.pkl

        Parameters
        ----------
        model : str | None
            Which estimator to save.  Saves all when None.
        save_dir : str | Path
            Output directory (created if needed).

        Returns
        -------
        list of absolute Path objects for every saved file.
        """
        self._check_fitted()
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        targets = (
            {model: self.estimators_[model]} if model
            else self.estimators_
        )
        saved: list[Path] = []
        for name, clf in targets.items():
            file_path = save_dir / (_CLASSIFIER_SAVE_NAMES[name] + ".pkl")
            joblib.dump(clf, file_path)
            print(f"✅ {name.upper()} saved to: {file_path.resolve()}")
            saved.append(file_path.resolve())
        return saved

    @classmethod
    def load(cls, model: str, file_path: str | Path) -> "TOARClassifier":
        """
        Load a single previously saved classifier and return a TOARClassifier
        instance that contains only that estimator.

        Parameters
        ----------
        model : str
            Key identifying the model type (e.g. 'rf', 'lgbm', 'catboost', 'xgb').
        file_path : str | Path
            Path to the .pkl file produced by save().

        Returns
        -------
        TOARClassifier (fitted instance with one estimator).

        Example
        -------
        >>> clf = TOARClassifier.load('rf', 'models/rf_model.pkl')
        >>> preds = clf.predict(X_test, model='rf')
        """
        file_path = Path(file_path)
        clf = joblib.load(file_path)

        instance = cls(models=[model], use_smote=False)
        instance.estimators_[model] = clf
        instance.is_fitted_ = True
        print(f"✅ {model.upper()} loaded from: {file_path.resolve()}")
        return instance


if __name__ == "__main__":
    print("This module defines the TOARClustering and TOARClassifier classes for unsupervised and supervised modeling, respectively.")