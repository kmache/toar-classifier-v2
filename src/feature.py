"""
feature.py
----------
Feature engineering pipeline for TOAR station classification.

The TOARFeature class is a stateful sklearn-compatible transformer:
  - fit()           : learns encoders, scalers, and transformers on training data
  - transform()     : applies the learned transformations to new data
  - fit_transform() : fit then transform in one call
  - save(path)      : persist fitted engineer to models/feature_engineer.pkl
  - load(path)      : (classmethod) reload a saved engineer for inference

Typical training workflow:
    feature_eng = TOARFeature(scaling='standard', cat_encoder='ohe')
    X_train     = feature_eng.fit_transform(df_train)
    feature_eng.save('models/feature_engineer.pkl')

Typical inference workflow:
    feature_eng = TOARFeature.load('models/feature_engineer.pkl')
    X_new       = feature_eng.transform(new_df)
"""

import warnings
from pathlib import Path
import sys
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

warnings.filterwarnings("ignore")

# Re-use the column schema from processing to stay DRY
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from src.processing import NUM_VARS


class TOARFeature(BaseEstimator, TransformerMixin):
    """Feature engineering pipeline for TOAR station metadata.

    Applies the following steps in order (all learned on training data only):
        1.  Column selection.
        2.  IQR-based outlier capping (optional).
        3.  Categorical encoding: OHE or Label Encoding (optional).
        4.  Numeric conversion of specified categorical columns (optional).
        5.  Skewness correction via Yeo-Johnson PowerTransformer (optional).
        6.  Feature scaling: standard, minmax, or robust (optional).

    Every stateful step (IQR bounds, encoders, scaler, power transformer) is
    learned exclusively during fit() and then applied consistently in
    transform(), preventing data leakage between train and test/inference.

    Args:
        selected_columns: Columns to keep. None keeps all columns.
        scaling: Scaler to apply: 'standard', 'minmax', 'robust', or None.
        scale_only: 'numeric' (default) scales only numeric columns;
                    'all' scales all columns except 'area_code'.
        cat_encoder: Encoding for categorical columns: 'ohe', 'le', or None.
        encode_vars: Explicit list of columns to encode. None encodes nothing.
        convert_vars: Columns to convert from str to numeric.
        cap_outliers: If True, cap numeric features to IQR bounds at fit time.
        handle_skewness: If True, apply Yeo-Johnson to highly skewed features
                         (|skewness| > 1) at fit time.

    Attributes:
        _scaler: Fitted scaler instance (or None).
        _power_transformer: Fitted PowerTransformer (or None).
        _label_encoders: Dict of {column: fitted LabelEncoder}.
        _ohe_columns: Column list after OHE (used to align inference data).
        _iqr_bounds: Dict of {column: (lower, upper)} learned IQR bounds.
        _skewed_cols: List of columns selected for Yeo-Johnson transform.
        _numeric_cols: Numeric columns selected during fit.
        _cols_to_scale: Columns to scale (resolved during fit).
        _is_fitted: Whether fit() has been called.

    Example:
        >>> fe = TOARFeature(scaling='standard', cat_encoder='ohe',
        ...                  encode_vars=['climatic_zone_year2016'],
        ...                  cap_outliers=True)
        >>> X_train = fe.fit_transform(df_train)
        >>> fe.save()
        >>> fe = TOARFeature.load()
        >>> X_test = fe.transform(df_test)
    """

    def __init__(
        self,
        selected_columns: list[str] | None = None,
        scaling: str | None = None,
        scale_only: str = "numeric",
        cat_encoder: str | None = None,
        encode_vars: list[str] | None = None,
        convert_vars: list[str] | None = None,
        cap_outliers: bool = False,
        handle_skewness: bool = True,
    ) -> None:
        self.selected_columns = selected_columns
        self.scaling = scaling
        self.scale_only = scale_only
        self.cat_encoder = cat_encoder
        self.encode_vars = encode_vars
        self.convert_vars = convert_vars
        self.cap_outliers = cap_outliers
        self.handle_skewness = handle_skewness

        # Learned state
        self._scaler: StandardScaler | MinMaxScaler | RobustScaler | None = None
        self._power_transformer: PowerTransformer | None = None
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._ohe_columns: list[str] | None = None
        self._iqr_bounds: dict[str, tuple[float, float]] = {}
        self._skewed_cols: list[str] = []
        self._numeric_cols: list[str] = []
        self._cols_to_scale: list[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 1: column selection."""
        cols = self.selected_columns if self.selected_columns is not None else df.columns.tolist()
        # Only keep columns that actually exist (safe for inference data)
        cols = [c for c in cols if c in df.columns]
        return df[cols].copy()

    def _resolve_numeric_cols(self, df: pd.DataFrame) -> list[str]:
        """Return the numeric columns that are also in NUM_VARS."""
        return [c for c in df.select_dtypes(include="number").columns if c in NUM_VARS]

    def _build_scaler(self) -> StandardScaler | MinMaxScaler | RobustScaler:
        if self.scaling == "standard":
            return StandardScaler()
        elif self.scaling == "minmax":
            return MinMaxScaler()
        elif self.scaling == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"scaling must be 'standard', 'minmax', or 'robust', got '{self.scaling}'.")

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "TOARFeature":
        """Learn all stateful transformations from training data.

        Args:
            X: Preprocessed (post-TOARProcessing) DataFrame.
            y: Ignored.

        Returns:
            self
        """
        df = self._select_columns(X)

        # Resolve encode/convert lists — filter to present columns
        encode_vars = [v for v in (self.encode_vars or []) if v in df.columns]
        convert_vars = [v for v in (self.convert_vars or []) if v in df.columns]

        # Step 4: convert categorical columns to numeric before resolving numeric_cols
        for var in convert_vars:
            df[var] = pd.to_numeric(df[var], errors="coerce")

        self._numeric_cols = self._resolve_numeric_cols(df)

        # Step 2: learn IQR bounds
        if self.cap_outliers and self._numeric_cols:
            q1 = df[self._numeric_cols].quantile(0.25)
            q3 = df[self._numeric_cols].quantile(0.75)
            iqr = q3 - q1
            for col in self._numeric_cols:
                lower = float(q1[col] - 1.5 * iqr[col])
                upper = float(q3[col] + 1.5 * iqr[col])
                self._iqr_bounds[col] = (lower, upper)
            # Apply capping to train data for downstream fitting
            for col, (lo, hi) in self._iqr_bounds.items():
                df[col] = df[col].clip(lower=lo, upper=hi)

        # Step 3: fit encoders
        if encode_vars and self.cat_encoder:
            if self.cat_encoder == "ohe":
                # Fit = just record OHE output column names using training data
                df = pd.get_dummies(df, columns=encode_vars, drop_first=True, dtype=float)
                self._ohe_columns = df.columns.tolist()
            elif self.cat_encoder == "le":
                for var in encode_vars:
                    le = LabelEncoder()
                    le.fit(df[var].astype(str))
                    self._label_encoders[var] = le
                    df[var] = le.transform(df[var].astype(str))
            else:
                raise ValueError("cat_encoder must be 'ohe' or 'le'.")

        # Refresh numeric cols after encoding (OHE may add columns)
        self._numeric_cols = self._resolve_numeric_cols(df)

        # Step 5: learn PowerTransformer on skewed features
        if self.handle_skewness and self._numeric_cols:
            skewness = df[self._numeric_cols].skew().abs()
            self._skewed_cols = skewness[skewness > 1].index.tolist()
            if self._skewed_cols:
                self._power_transformer = PowerTransformer(method="yeo-johnson")
                self._power_transformer.fit(df[self._skewed_cols])

        # Step 6: resolve and fit scaler
        if self.scaling:
            if self.scale_only == "numeric":
                self._cols_to_scale = self._numeric_cols
            elif self.scale_only == "all":
                self._cols_to_scale = [c for c in df.columns if c != "area_code"]
            else:
                raise ValueError(f"scale_only must be 'numeric' or 'all', got '{self.scale_only}'.")
            self._scaler = self._build_scaler()
            self._scaler.fit(df[self._cols_to_scale])

        self._is_fitted = True
        print("✅ TOARFeature fitted successfully!")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Apply learned feature engineering to new data.

        Args:
            X: Preprocessed (post-TOARProcessing) DataFrame.
            y: Ignored.

        Returns:
            Transformed feature DataFrame.

        Raises:
            RuntimeError: If transform() is called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        df = self._select_columns(X)

        encode_vars = [v for v in (self.encode_vars or []) if v in df.columns]
        convert_vars = [v for v in (self.convert_vars or []) if v in df.columns]

        # Step 4: convert to numeric
        for var in convert_vars:
            df[var] = pd.to_numeric(df[var], errors="coerce")

        # Step 2: apply IQR capping using learned bounds
        for col, (lo, hi) in self._iqr_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lo, upper=hi)

        # Step 3: apply encoders
        if encode_vars and self.cat_encoder:
            if self.cat_encoder == "ohe":
                df = pd.get_dummies(df, columns=encode_vars, drop_first=True, dtype=float)
                # Align columns with training — add missing cols as 0, drop extra cols
                if self._ohe_columns is not None:
                    for c in self._ohe_columns:
                        if c not in df.columns:
                            df[c] = 0.0
                    df = df[[c for c in self._ohe_columns if c in df.columns]]
            elif self.cat_encoder == "le":
                for var, le in self._label_encoders.items():
                    if var in df.columns:
                        # Handle unseen labels gracefully
                        known = set(le.classes_)
                        df[var] = df[var].astype(str).apply(
                            lambda x: x if x in known else le.classes_[0]
                        )
                        df[var] = le.transform(df[var])

        # Step 5: apply PowerTransformer to skewed columns
        if self._power_transformer is not None and self._skewed_cols:
            present_skewed = [c for c in self._skewed_cols if c in df.columns]
            if present_skewed:
                df[present_skewed] = self._power_transformer.transform(df[present_skewed])

        # Step 6: apply scaling
        if self._scaler is not None and self._cols_to_scale:
            present_scale_cols = [c for c in self._cols_to_scale if c in df.columns]
            if present_scale_cols:
                df[present_scale_cols] = self._scaler.transform(df[present_scale_cols])

        print(f"✅ Feature engineering complete! Shape: {df.shape}")
        return df

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        """Fit on X then transform X in a single efficient pass.

        Args:
            X: Preprocessed (post-TOARProcessing) DataFrame.
            y: Ignored.

        Returns:
            Transformed feature DataFrame.
        """
        df = self._select_columns(X)
        encode_vars = [v for v in (self.encode_vars or []) if v in df.columns]
        convert_vars = [v for v in (self.convert_vars or []) if v in df.columns]

        # Step 4: convert to numeric
        for var in convert_vars:
            df[var] = pd.to_numeric(df[var], errors="coerce")
        self._numeric_cols = self._resolve_numeric_cols(df)

        # Step 2: learn + apply IQR capping
        if self.cap_outliers and self._numeric_cols:
            q1 = df[self._numeric_cols].quantile(0.25)
            q3 = df[self._numeric_cols].quantile(0.75)
            iqr = q3 - q1
            for col in self._numeric_cols:
                lo = float(q1[col] - 1.5 * iqr[col])
                hi = float(q3[col] + 1.5 * iqr[col])
                self._iqr_bounds[col] = (lo, hi)
                df[col] = df[col].clip(lower=lo, upper=hi)
            print(f"✅ Outlier capping applied to {len(self._iqr_bounds)} columns!")

        # Step 3: fit + apply encoders
        if encode_vars and self.cat_encoder:
            if self.cat_encoder == "ohe":
                df = pd.get_dummies(df, columns=encode_vars, drop_first=True, dtype=float)
                self._ohe_columns = df.columns.tolist()
                print(f"✅ One-hot encoded {len(encode_vars)} variables: {encode_vars}")
            elif self.cat_encoder == "le":
                for var in encode_vars:
                    le = LabelEncoder()
                    df[var] = le.fit_transform(df[var].astype(str))
                    self._label_encoders[var] = le
                print(f"✅ Label encoded {len(encode_vars)} variables: {encode_vars}")
            else:
                raise ValueError("cat_encoder must be 'ohe' or 'le'.")

        # Refresh numeric cols after encoding
        self._numeric_cols = self._resolve_numeric_cols(df)

        # Step 5: fit + apply PowerTransformer on skewed columns
        if self.handle_skewness and self._numeric_cols:
            skewness = df[self._numeric_cols].skew().abs()
            self._skewed_cols = skewness[skewness > 1].index.tolist()
            if self._skewed_cols:
                self._power_transformer = PowerTransformer(method="yeo-johnson")
                df[self._skewed_cols] = self._power_transformer.fit_transform(df[self._skewed_cols])
                print(f"✅ Yeo-Johnson applied to {len(self._skewed_cols)} skewed columns!")

        # Step 6: fit + apply scaler
        if self.scaling:
            if self.scale_only == "numeric":
                self._cols_to_scale = self._numeric_cols
            elif self.scale_only == "all":
                self._cols_to_scale = [c for c in df.columns if c != "area_code"]
            else:
                raise ValueError(f"scale_only must be 'numeric' or 'all', got '{self.scale_only}'.")
            self._scaler = self._build_scaler()
            df[self._cols_to_scale] = self._scaler.fit_transform(df[self._cols_to_scale])
            print(f"✅ {self.scaling} scaling applied to {self.scale_only} features!")

        self._is_fitted = True
        print(f"✅ Feature engineering complete! Shape: {df.shape}")
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path = "models/feature_engineer.pkl") -> None:
        """Persist the fitted feature engineer to disk using joblib.

        Args:
            path: Destination file path. Defaults to 'models/feature_engineer.pkl'.

        Raises:
            RuntimeError: If called before fit().

        Example:
            >>> fe = TOARFeature(scaling='standard')
            >>> fe.fit_transform(df_train)
            >>> fe.save()
            >>> fe.save('models/fe_v2.pkl')
        """
        if not self._is_fitted:
            raise RuntimeError("Feature engineer must be fitted before saving. Call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"✅ Feature engineer saved to '{path}'")

    @classmethod
    def load(cls, path: str | Path = "models/feature_engineer.pkl") -> "TOARFeature":
        """Load a previously fitted feature engineer from disk.

        Args:
            path: Path to the saved .pkl file.

        Returns:
            A fitted TOARFeature instance.

        Raises:
            FileNotFoundError: If the file does not exist.

        Example:
            >>> fe = TOARFeature.load('models/feature_engineer.pkl')
            >>> X_new = fe.transform(new_df)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No feature engineer found at '{path}'. Train and save one first."
            )
        fe = joblib.load(path)
        print(f"✅ Feature engineer loaded from '{path}'")
        return fe

if __name__ == "__main__":
    print("This module defines the TOARFeature class for feature engineering. It is not meant to be run directly. Use it as part of a training or inference pipeline.")
    