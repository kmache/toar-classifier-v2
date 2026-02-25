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
- ``str``                 – Single TOAR station code (fetched from API),
                            **or** a path to a ``.json`` file containing
                            station record(s).
- ``list[str]``           – Multiple TOAR station codes (fetched from API).
- ``str`` / ``Path``      – Path to a ``.json`` file (dict or list of dicts).

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

import json
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
    verbose : bool
        If True (default), print progress messages during loading and
        prediction. Set to False to suppress all output (also propagated
        to the loaded processor and feature engineer).

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
        verbose: bool = True,
    ) -> None:
        if output_format not in ("dataframe", "dicts"):
            raise ValueError("output_format must be 'dataframe' or 'dicts'.")
        self.models_dir = Path(models_dir)
        self.best_model = best_model
        self.output_format = output_format
        self.verbose = verbose

        # Populated by _load_artifacts
        self.models_: dict[str, object] = {}
        self.processor_: TOARProcessing | None = None
        self.feature_engineer_: TOARFeature | None = None

        # Classifier metadata (categorical handling)
        self._cat_cols: list[str] = []
        self._rf_ohe = None
        self._label_encoder = None

        self._load_artifacts()

    def _log(self, msg: str) -> None:
        """Print *msg* only when verbose mode is enabled."""
        if self.verbose:
            print(msg)

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
        self.processor_.verbose = self.verbose

        # -- 2. Feature engineer ------------------------------------------
        fe_path = models_dir / "feature_engineer.pkl"
        if not fe_path.exists():
            raise FileNotFoundError(
                f"feature_engineer.pkl not found in '{models_dir}'. "
                "Train and save the feature engineer first."
            )
        self.feature_engineer_ = TOARFeature.load(fe_path)
        self.feature_engineer_.verbose = self.verbose

        # -- 3. Classifier models -----------------------------------------
        for name, filename in _CLASSIFIER_FILES.items():
            path = models_dir / filename
            if path.exists():
                clf = joblib.load(path)
                self.models_[name] = clf
                self._log(f"✅ {name.upper()} loaded from: {path.resolve()}")

        if not self.models_:
            raise FileNotFoundError(
                f"No classifier model files found in '{models_dir}'. "
                f"Expected one or more of: {list(_CLASSIFIER_FILES.values())}"
            )

        # -- 4. Classifier metadata (categorical handling) ----------------
        meta_path = models_dir / "classifier_meta.pkl"
        if meta_path.exists():
            meta = joblib.load(meta_path)
            self._cat_cols = meta.get("_cat_cols", [])
            self._rf_ohe = meta.get("_rf_ohe", None)
            self._label_encoder = meta.get("_label_encoder", None)
            self._log(f"✅ Classifier metadata loaded from: {meta_path.resolve()}")

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

        Supported inputs
        ----------------
        - ``pd.DataFrame`` – raw station metadata.
        - ``dict`` / ``list[dict]`` – station record(s).
        - ``str`` – a single station code **or** a path to a JSON file.
        - ``list[str]`` – multiple station codes.

        When station codes are provided, the data is fetched from the
        TOAR-II API.  Codes that cannot be fetched (HTTP error, missing
        data) are reported to the user and excluded from the result.

        Returns
        -------
        raw_df : pd.DataFrame
            Full raw DataFrame (all columns, including identifier columns).
        identifiers : pd.DataFrame
            Sub-DataFrame with only the identifier / ground-truth columns
            (``lat``, ``lon``, ``area_code``, ``type_of_area`` if present).
            Used to reconstruct the output after prediction.
        """
        # -- JSON file path → load from disk ----------------------------
        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.suffix == ".json" and path.is_file():
                with open(path) as f:
                    payload = json.load(f)
                # Accept both a single dict and a list of dicts
                if isinstance(payload, dict):
                    data = pd.DataFrame([payload])
                elif isinstance(payload, list):
                    data = pd.DataFrame(payload)
                else:
                    raise ValueError(
                        f"JSON file must contain a dict or list of dicts, "
                        f"got {type(payload).__name__}."
                    )

        # -- station code(s) → fetch from API ---------------------------
        if isinstance(data, str):
            requested_codes = [data]
            data = get_data_from_station_codes(requested_codes)
            self._report_skipped_codes(requested_codes, data)
        elif isinstance(data, list) and data and isinstance(data[0], str):
            requested_codes = list(data)
            data = get_data_from_station_codes(requested_codes)
            self._report_skipped_codes(requested_codes, data)

        # -- dict → DataFrame -------------------------------------------
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            data = pd.DataFrame(data)

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "predict() accepts: pd.DataFrame, dict, list[dict], "
                "str (station code or JSON path), list[str] (station codes). "
                f"Got: {type(data).__name__}"
            )

        if data.empty:
            raise ValueError(
                "No valid station data could be retrieved. "
                "Please verify the station codes and try again."
            )

        # -- extract identifier columns so we can reattach them ---------
        id_cols = [c for c in _ID_COLS if c in data.columns]
        gt_cols = [_GROUND_TRUTH_COL] if _GROUND_TRUTH_COL in data.columns else []
        identifiers = data[id_cols + gt_cols].copy().reset_index(drop=True)

        return data.reset_index(drop=True), identifiers

    @staticmethod
    def _report_skipped_codes(
        requested: list[str], fetched_df: pd.DataFrame
    ) -> None:
        """Print a warning listing any station codes that could not be fetched."""
        if fetched_df.empty:
            fetched_codes = set()
        else:
            fetched_codes = set(fetched_df["area_code"].values)
        skipped = [c for c in requested if c not in fetched_codes]
        if skipped:
            print(
                f"⚠️  The following station code(s) could not be retrieved "
                f"and were skipped — please verify them:\n  {skipped}"
            )

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _preprocess(self, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
        """Apply processor → feature engineer to produce the feature matrix.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix (rows may be fewer than input due to cleaning).
        surviving_idx : pd.Index
            Original integer positions (from ``raw_df``) that survived
            the preprocessing steps, used to align identifiers.
        """
        df_clean = self.processor_.transform(raw_df)

        X = self.feature_engineer_.transform(df_clean)

        # Map surviving (lat, lon) back to original row positions.
        # The processor deduplicates by keeping the first occurrence of each
        # (lat, lon) pair, so we must do the same when mapping back — using
        # ``isin`` alone would match *all* duplicates and inflate the count.
        raw_indexed = raw_df.reset_index(drop=True)
        first_occ = raw_indexed.drop_duplicates(subset=["lat", "lon"], keep="first")
        latlon_to_pos = {
            (row["lat"], row["lon"]): idx
            for idx, row in first_occ.iterrows()
        }

        surviving_positions = pd.Index(
            [latlon_to_pos[idx] for idx in X.index if idx in latlon_to_pos]
        )

        return X, surviving_positions

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

    def _prepare_X(self, name: str, X: pd.DataFrame) -> pd.DataFrame:
        """Return a model-specific copy of X with categoricals handled.

        Strategy per model
        ------------------
        rf       : OHE (sklearn has no native categorical support)
        catboost : pass strings as-is; handled natively
        lgbm     : cast to pd.Categorical
        xgb      : cast to pd.Categorical (requires enable_categorical=True)
        """
        cat_cols = [c for c in self._cat_cols if c in X.columns]
        if not cat_cols:
            return X
        X = X.copy()
        if name == "rf" and self._rf_ohe is not None:
            ohe_arr = self._rf_ohe.transform(X[cat_cols])
            ohe_cols = self._rf_ohe.get_feature_names_out(cat_cols).tolist()
            ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=X.index)
            X = pd.concat([X.drop(columns=cat_cols), ohe_df], axis=1)
        elif name in ("lgbm", "xgb"):
            for col in cat_cols:
                X[col] = X[col].astype("category")
        # catboost: strings are native – no transformation needed
        return X

    def _predict_one(self, name: str, clf, X: pd.DataFrame) -> np.ndarray:
        X = self._prepare_X(name, X)
        features = self._get_feature_names(clf)
        # Align columns; missing columns are filled with 0
        X_aligned = X.reindex(columns=features, fill_value=0)
        preds = clf.predict(X_aligned).reshape(-1)
        if self._label_encoder is not None:
            preds = self._label_encoder.inverse_transform(preds.astype(int))
        return preds

    # ------------------------------------------------------------------
    # Public predict
    # ------------------------------------------------------------------

    def predict(
        self,
        data: pd.DataFrame | dict | list | str,
        output_format: str | None = None,
    ) -> pd.DataFrame | list[dict]:
        """Run the full inference pipeline.

        Parameters
        ----------
        data : pd.DataFrame | dict | list[dict] | str | list[str]
            Input data in any of the supported formats (see module docstring).
        output_format : str | None
            ``"dataframe"`` or ``"dicts"``.  When ``None`` (default), uses
            the instance-level ``self.output_format`` set at init time.

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

        # Build feature matrix (may drop rows during cleaning)
        X, surviving_positions = self._preprocess(raw_df)

        # Separate surviving vs dropped identifiers
        all_positions = identifiers.index
        dropped_positions = all_positions.difference(surviving_positions)
        id_survived = identifiers.loc[surviving_positions].reset_index(drop=True)

        # -- Collect per-model predictions --------------------------------
        preds: dict[str, np.ndarray] = {}
        for name, clf in self.models_.items():
            preds[name] = self._predict_one(name, clf, X)

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

        # -- Build result DataFrame for survived rows ---------------------
        result = id_survived.copy()
        for model_name, arr in selected_preds.items():
            result[f"pred_{model_name}"] = arr

        # -- Re-attach dropped rows with sentinel message -----------------
        _NO_DATA_MSG = "not enough data for prediction"
        if not dropped_positions.empty:
            dropped = identifiers.loc[dropped_positions].reset_index(drop=True)
            for model_name in selected_preds:
                dropped[f"pred_{model_name}"] = _NO_DATA_MSG
            result = pd.concat([result, dropped], ignore_index=True)

        # -- Return -------------------------------------------------------
        fmt = output_format or self.output_format
        if fmt == "dicts":
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