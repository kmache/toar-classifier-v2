"""
inference.py
------------
End-to-end inference pipeline for TOAR station area-type classification.

The ``TOARInference`` class ties together the saved artifacts produced during
training (processor, feature engineer, and one or more classifiers) into a
single, easy-to-use prediction interface.

Typical workflow
----------------
    infer = TOARInference(models_dir="models", best_model="voting")
    result = infer.predict(["DE0001A", "FR0123X"])   # station codes
    result = infer.predict(df_raw)                   # raw DataFrame
    result = infer.predict({"area_code": "IT0001A", "lat": 41.9, ...})

Input formats accepted by ``predict()``
----------------------------------------
- ``pd.DataFrame``        – One row per station, with raw metadata columns.
- ``dict``                – Single station as a flat key/value dict.
- ``list[dict]``          – Multiple stations, each as a flat dict.
- ``str``                 – Single TOAR station code (fetched from API).
- ``list[str]``           – Multiple TOAR station codes (fetched from API).

Output
------
A ``pd.DataFrame`` (default) or ``list[dict]`` with columns:
    lat, lon, area_code,
    pred_<model_name> for every loaded model,
    pred_voting        (majority-vote ensemble across all classifiers),
    type_of_area       (TOAR-labelled ground truth, kept when present).

When *best_model* is set, only ``pred_<best_model>`` is returned instead of
all per-model columns (plus the ``voting`` column is omitted).
"""

import sys
import os
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
# Support running from project root or from inside src/
_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
for _p in [str(_ROOT), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.processing import TOARProcessing          # noqa: E402
from src.feature import TOARFeature                # noqa: E402
from script.dataloader import get_data_from_station_codes  # noqa: E402

# ---------------------------------------------------------------------------
# Known model artifact names (must stay in sync with modeling.py)
# ---------------------------------------------------------------------------
_CLASSIFIER_FILES: dict[str, str] = {
    "rf":       "rf_model.pkl",
    "catboost": "catboost_model.pkl",
    "lgbm":     "lgbm_model.pkl",
    "xgb":      "xgb_model.pkl",
}

# Output columns that carry identifiers / ground truth and should be
# preserved through the pipeline without being model inputs.
_ID_COLS = ["lat", "lon", "area_code"]
_GROUND_TRUTH_COL = "type_of_area"


# ===========================================================================
# TOARInference
# ===========================================================================

class TOARInference:
    """Load trained TOAR artifacts and run area-type classification.

    Parameters
    ----------
    models_dir : str | Path
        Directory that contains ``processor.pkl``, ``feature_engineer.pkl``,
        and the classifier ``.pkl`` files produced by the training pipeline.
        Defaults to ``"models"``.
    best_model : str | None
        When set, ``predict()`` returns only the prediction column for this
        model (e.g. ``"rf"``, ``"lgbm"``, ``"catboost"``, ``"xgb"``,
        ``"voting"``).  When ``None`` (default) predictions from **all**
        available models plus the majority-vote ensemble are returned.
    output_format : str
        ``"dataframe"`` (default) – return a ``pd.DataFrame``.
        ``"dicts"``               – return a ``list[dict]``.

    Attributes
    ----------
    models_ : dict[str, object]
        Fitted classifier objects keyed by their short name (``"rf"`` etc.).
    processor_ : TOARProcessing
        Fitted preprocessing transformer.
    feature_engineer_ : TOARFeature
        Fitted feature engineering transformer.
    """

    def __init__(
        self,
        models_dir: str | Path = "models",
        best_model: str | None = None,
        output_format: str = "dataframe",
    ) -> None:
        if output_format not in ("dataframe", "dicts"):
            raise ValueError("output_format must be 'dataframe' or 'dicts'.")
        self.models_dir = Path(models_dir)
        self.best_model = best_model
        self.output_format = output_format

        # Populated by _load_artifacts
        self.models_: dict[str, object] = {}
        self.processor_: TOARProcessing | None = None
        self.feature_engineer_: TOARFeature | None = None

        self._load_artifacts()

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------

    def _load_artifacts(self) -> None:
        """Load all available artifacts from ``self.models_dir``.

        Steps performed
        ---------------
        1. Load ``processor.pkl``         → ``self.processor_``
        2. Load ``feature_engineer.pkl``  → ``self.feature_engineer_``
        3. Scan for classifier ``.pkl`` files and load each one found.
        """
        models_dir = self.models_dir

        # -- 1. Processor -------------------------------------------------
        processor_path = models_dir / "processor.pkl"
        if not processor_path.exists():
            raise FileNotFoundError(
                f"processor.pkl not found in '{models_dir}'. "
                "Train and save the processor first."
            )
        self.processor_ = TOARProcessing.load(processor_path)

        # -- 2. Feature engineer ------------------------------------------
        fe_path = models_dir / "feature_engineer.pkl"
        if not fe_path.exists():
            raise FileNotFoundError(
                f"feature_engineer.pkl not found in '{models_dir}'. "
                "Train and save the feature engineer first."
            )
        self.feature_engineer_ = TOARFeature.load(fe_path)

        # -- 3. Classifier models -----------------------------------------
        for name, filename in _CLASSIFIER_FILES.items():
            path = models_dir / filename
            if path.exists():
                clf = joblib.load(path)
                self.models_[name] = clf
                print(f"✅ {name.upper()} loaded from: {path.resolve()}")

        if not self.models_:
            raise FileNotFoundError(
                f"No classifier model files found in '{models_dir}'. "
                f"Expected one or more of: {list(_CLASSIFIER_FILES.values())}"
            )

        # Validate best_model is among loaded models (or 'voting')
        if self.best_model is not None:
            valid = set(self.models_.keys()) | {"voting"}
            if self.best_model not in valid:
                raise ValueError(
                    f"best_model='{self.best_model}' is not among loaded models "
                    f"or 'voting'. Available: {sorted(valid)}"
                )

    # ------------------------------------------------------------------
    # Input normalisation
    # ------------------------------------------------------------------

    def _normalise_input(
        self,
        data: pd.DataFrame | dict | list | str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert any supported input type to a raw DataFrame.

        Returns
        -------
        raw_df : pd.DataFrame
            Full raw DataFrame (all columns, including identifier columns).
        identifiers : pd.DataFrame
            Sub-DataFrame with only the identifier / ground-truth columns
            (``lat``, ``lon``, ``area_code``, ``type_of_area`` if present).
            Used to reconstruct the output after prediction.
        """
        # -- station code(s) → fetch from API ---------------------------
        if isinstance(data, str):
            data = get_data_from_station_codes([data])
        elif isinstance(data, list) and data and isinstance(data[0], str):
            data = get_data_from_station_codes(data)

        # -- dict → DataFrame -------------------------------------------
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            data = pd.DataFrame(data)

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "predict() accepts: pd.DataFrame, dict, list[dict], "
                "str (station code), or list[str] (station codes). "
                f"Got: {type(data).__name__}"
            )

        # -- extract identifier columns so we can reattach them ---------
        id_cols = [c for c in _ID_COLS if c in data.columns]
        gt_cols = [_GROUND_TRUTH_COL] if _GROUND_TRUTH_COL in data.columns else []
        identifiers = data[id_cols + gt_cols].copy().reset_index(drop=True)

        return data.reset_index(drop=True), identifiers

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _preprocess(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Apply processor → feature engineer to produce the feature matrix."""
        df_clean = self.processor_.transform(raw_df)
        X = self.feature_engineer_.transform(df_clean)
        return X

    # ------------------------------------------------------------------
    # Per-model prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_feature_names(clf) -> list[str]:
        """Extract stored feature names from a fitted sklearn/boosting clf.

        Handles XGBoost specifically via ``get_booster().feature_names``,
        then falls back to the standard sklearn attributes.
        """
        # XGBoost: feature names live on the underlying Booster object
        if hasattr(clf, "get_booster"):
            names = clf.get_booster().feature_names
            if names is not None:
                return list(names)

        for attr in ("feature_names_in_", "feature_names_", "feature_name_"):
            if hasattr(clf, attr):
                return list(getattr(clf, attr))

        raise AttributeError(
            f"{type(clf).__name__} does not expose known feature-name attributes."
        )

    def _predict_one(self, clf, X: pd.DataFrame) -> np.ndarray:
        features = self._get_feature_names(clf)
        # Align columns; missing columns are filled with 0
        X_aligned = X.reindex(columns=features, fill_value=0)
        return clf.predict(X_aligned).reshape(-1)

    # ------------------------------------------------------------------
    # Public predict
    # ------------------------------------------------------------------

    def predict(
        self,
        data: pd.DataFrame | dict | list | str,
    ) -> pd.DataFrame | list[dict]:
        """Run the full inference pipeline.

        Parameters
        ----------
        data : pd.DataFrame | dict | list[dict] | str | list[str]
            Input data in any of the supported formats (see module docstring).

        Returns
        -------
        pd.DataFrame or list[dict]
            Columns: ``lat``, ``lon``, ``area_code``,
            ``type_of_area`` (if present in input),
            ``pred_<model>`` for each loaded model,
            ``pred_voting`` (majority-vote ensemble).
            When *best_model* is set only ``pred_<best_model>`` is returned.
        """
        raw_df, identifiers = self._normalise_input(data)

        # Build feature matrix
        X = self._preprocess(raw_df)

        # -- Collect per-model predictions --------------------------------
        preds: dict[str, np.ndarray] = {}
        for name, clf in self.models_.items():
            preds[name] = self._predict_one(clf, X)

        # -- Majority-vote ensemble ---------------------------------------
        if len(preds) > 1:
            pred_matrix = np.column_stack(list(preds.values()))
            preds["voting"] = np.array(
                [Counter(row).most_common(1)[0][0] for row in pred_matrix],
                dtype=object,
            )
        elif len(preds) == 1:
            # Only one model loaded – voting equals that model's predictions
            solo_key = next(iter(preds))
            preds["voting"] = preds[solo_key].copy()

        # -- Filter to best_model if requested ----------------------------
        if self.best_model is not None:
            if self.best_model == "voting":
                selected_preds = {"voting": preds["voting"]}
            else:
                selected_preds = {self.best_model: preds[self.best_model]}
        else:
            selected_preds = preds

        # -- Build result DataFrame ---------------------------------------
        result = identifiers.copy()
        for model_name, arr in selected_preds.items():
            result[f"pred_{model_name}"] = arr

        # -- Return -------------------------------------------------------
        if self.output_format == "dicts":
            return result.to_dict(orient="records")
        return result

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = list(self.models_.keys())
        return (
            f"TOARInference("
            f"models_dir='{self.models_dir}', "
            f"models={loaded}, "
            f"best_model={self.best_model!r}"
            f")"
        )
if __name__ == "__main__":
    print("This module defines the TOARInference class for end-to-end inference.")