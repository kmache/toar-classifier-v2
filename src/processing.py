"""
processing.py
-------------
Data cleaning and preprocessing for TOAR station classification.

The TOARProcessing class is a stateful sklearn-compatible transformer:
  - fit()           : learns imputers on training data
  - transform()     : applies the learned transformations
  - fit_transform() : fit then transform in one call
  - save(path)      : persist fitted processor to models/processor.pkl
  - load(path)      : (classmethod) reload a saved processor for inference

Typical training workflow:
    processor = TOARProcessing(filter_nan='any')
    df_clean  = processor.fit_transform(raw_train_df)
    processor.save('models/processor.pkl')

Typical training workflow (tree-based, no imputation):
    processor = TOARProcessing(filter_nan='any', impute_nan=False)
    df_clean  = processor.fit_transform(raw_train_df)

Typical inference workflow:
    processor = TOARProcessing.load('models/processor.pkl')
    df_clean  = processor.transform(new_raw_df)

Standalone helpers (parse_vars, timezone_mapping, group_landcover_cat,
iterative_imputer, read_stat_data, prepare_data_for_stat_eval) are kept
public so they can be used independently or in custom sklearn Pipelines.
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Column schema constants
# ---------------------------------------------------------------------------

CAT_VARS: list[str] = [
    "area_code",
    "timezone",
    "type_of_area",
    "climatic_zone_year2016",
    "htap_region_tier1_year2010",
    "dominant_landcover_year2012",
    "landcover_description_25km_year2012",
    "dominant_ecoregion_year2017",
    "ecoregion_description_25km_year2017",
]

NUM_VARS: list[str] = [
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

COLS_TO_PARSE: list[str] = [
    "climatic_zone_year2016",
    "htap_region_tier1_year2010",
    "dominant_landcover_year2012",
    "landcover_description_25km_year2012",
    "dominant_ecoregion_year2017",
    "ecoregion_description_25km_year2017",
]

NAN_SUBSET: list[str] = [
    "mean_stable_nightlights_5km_year2013",
    "max_stable_nightlights_25km_year2013",
    "max_stable_nightlights_25km_year1992",
    "mean_population_density_250m_year2015",
    "mean_population_density_5km_year2015",
    "max_population_density_25km_year2015",
    "mean_population_density_250m_year1990",
    "mean_population_density_5km_year1990",
    "max_population_density_25km_year1990",
]

TIMEZONE_MAP: dict[str, str] = {
    "Africa": "Africa",
    "America": "America",
    "Europe": "Europe",
    "Asia": "Asia",
    "Antarctica": "Antarctica",
    "Australia": "Australia",
    "Pacific": "Pacific",
    "N": "N",
    "UTC": "UTC",
    "Indian": "Indian",
    "Atlantic": "Atlantic",
    "Canada": "America",
    "CET": "Europe",
    "Arctic": "Europe",
    "EST": "America",
    "Turkey": "Europe",
    "Etc": "UTC",
}

# ---------------------------------------------------------------------------
# Standalone helper functions
# ---------------------------------------------------------------------------

def parse_vars(df: pd.DataFrame, vars_to_parse: list[str] = COLS_TO_PARSE) -> pd.DataFrame:
    """Parse coded categorical columns (e.g. '10 Cropland') to their numeric code.

    Args:
        df: Input DataFrame.
        vars_to_parse: Columns to parse. Defaults to COLS_TO_PARSE.

    Returns:
        DataFrame with parsed columns converted to numeric dtype.
    """
    def _parse_code(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series.astype("category").astype(str).str.partition(" ")[0])

    df = df.copy()
    for col in vars_to_parse:
        if col in df.columns:
            df[col] = _parse_code(df[col])
    return df


def timezone_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the 'timezone' column to broad continental regions.

    Args:
        df: Input DataFrame (must contain a 'timezone' column).

    Returns:
        DataFrame with normalised 'timezone' values.
    """
    if "timezone" not in df.columns:
        return df
    df = df.copy()
    df["timezone"] = df["timezone"].str.split("/").str[0].str.strip().map(TIMEZONE_MAP)
    return df


def group_landcover_cat(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse fine-grained landcover codes to their 10-unit parent category.

    E.g. codes 11, 12, 14 → 10 (all in the same broad class).

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with grouped 'dominant_landcover_year2012' values.
    """
    col = "dominant_landcover_year2012"
    if col not in df.columns:
        return df
    df = df.copy()
    mapping = {k: (k // 10) * 10 for k in df[col].dropna().unique()}
    df[col] = df[col].replace(mapping)
    return df


def iterative_imputer(
    df_: pd.DataFrame,
    subset_to_impute: list[str],
    estimator: str | None = None,
) -> pd.DataFrame:
    """Impute missing numeric values using sklearn's IterativeImputer.

    Args:
        df_: Input DataFrame.
        subset_to_impute: Columns to impute.
        estimator: Regression estimator to use inside the imputer.
            None → BayesianRidge (sklearn default)
            'rf' → RandomForestRegressor
            'lgbm' → LGBMRegressor
            'cboost' → CatBoostRegressor

    Returns:
        DataFrame with imputed values in the specified columns.

    Raises:
        NotImplementedError: If an unsupported estimator name is given.
    """
    from fancyimpute import IterativeImputer

    df = df_.copy()

    if estimator is None:
        imputer = IterativeImputer()
    elif estimator == "rf":
        imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100))
    elif estimator == "lgbm":
        from lightgbm import LGBMRegressor
        imputer = IterativeImputer(estimator=LGBMRegressor(verbose=0))
    elif estimator == "cboost":
        from catboost import CatBoostRegressor
        imputer = IterativeImputer(CatBoostRegressor(verbose=0))
    else:
        raise NotImplementedError(
            f"Unknown estimator: '{estimator}'. Choose None, 'rf', 'lgbm', or 'cboost'."
        )

    imputed = pd.DataFrame(
        imputer.fit_transform(df[subset_to_impute]),
        columns=subset_to_impute,
        index=df.index,
    )
    df[subset_to_impute] = imputed
    return df


# ---------------------------------------------------------------------------
# Sklearn-compatible TOARProcessing class
# ---------------------------------------------------------------------------
class TOARProcessing(BaseEstimator, TransformerMixin):
    """Full preprocessing pipeline for TOAR station metadata.

    Wraps all cleaning and imputation steps in a single sklearn-compatible
    transformer so it can be embedded in a Pipeline and applied consistently
    to train and test/inference data.

    Steps performed in order:
        1. Set (lat, lon) as MultiIndex (if columns exist).
        2. Remove duplicate (lat, lon) pairs.
        3. Normalize timezone strings.
        4. Parse coded categorical columns to numeric.
        5. Group landcover codes to 10-unit categories.
        6. Replace sentinel values (-999, -9999, 9999) with NaN.
        7. Drop rows where all/any values in nan_subset are NaN.
        8. Cross-fill altitude <-> topography_srtm_alt_90m.
        9. Impute remaining numeric NaNs (IterativeImputer or mean) — skipped if impute_nan=False.
        10. Impute remaining categorical NaNs (most-frequent) — skipped if impute_nan=False.
        11. Remove rows with negative road_distance (data quality filter).
        12. Cast categorical columns to str dtype.

    Args:
        nan_subset: Columns used to detect and filter fully-missing rows.
        filter_nan: 'any' drops rows missing any nan_subset value;
                    'all' drops only rows missing all of them.
        num_imputer: Estimator for numeric imputation.
                     None -> BayesianRidge, 'rf', 'lgbm', 'cboost', or 'mean'.
        impute_nan: If True (default), impute numeric and categorical NaNs
                    (steps 9-10). If False, skip imputation entirely — useful
                    for tree-based models (XGBoost, LightGBM, CatBoost) that
                    handle NaN values natively.
        display_duplicate: If True, prints duplicate rows found during fit.

    Attributes:
        _cat_imputer: Fitted SimpleImputer for categorical columns.
        _num_simple_imputer: Fitted SimpleImputer (only when num_imputer='mean').
        _is_fitted: Whether fit() has been called.

    Example:
        >>> processor = TOARProcessing(filter_nan='any', num_imputer=None)
        >>> df_clean = processor.fit_transform(raw_df)
        >>> df_new_clean = processor.transform(new_raw_df)

        # Skip imputation for tree-based models:
        >>> processor = TOARProcessing(impute_nan=False)
        >>> df_clean = processor.fit_transform(raw_df)
    """

    def __init__(
        self,
        nan_subset: list[str] = NAN_SUBSET,
        filter_nan: str = "any",
        num_imputer: str | None = None,
        impute_nan: bool = True,
        display_duplicate: bool = False,
    ) -> None:
        self.nan_subset = nan_subset
        self.filter_nan = filter_nan
        self.num_imputer = num_imputer
        self.impute_nan = impute_nan
        self.display_duplicate = display_duplicate

        # Snapshot column schemas so they survive pickling
        self._num_vars: list[str] = list(NUM_VARS)
        self._cat_vars: list[str] = list(CAT_VARS)

        # State learned during fit
        self._cat_imputer: SimpleImputer | None = None
        self._num_simple_imputer: SimpleImputer | None = None
        self._is_fitted: bool = False

    # Bind module-level helpers so they survive pickling / %autoreload
    _timezone_mapping = staticmethod(timezone_mapping)
    _parse_vars = staticmethod(parse_vars)
    _group_landcover_cat = staticmethod(group_landcover_cat)
    _iterative_imputer = staticmethod(iterative_imputer)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_structural_cleaning(self, dataset: pd.DataFrame, is_fit: bool) -> pd.DataFrame:
        """Steps 1-8: deterministic cleaning that requires no learned state."""
        init_shape = dataset.shape

        # Step 1 & 2: set index and drop duplicates
        if "lat" in dataset.columns and "lon" in dataset.columns:
            dataset = dataset.set_index(["lat", "lon"])
            print("✅ Set (lat, lon) as index!")

        if self.display_duplicate and is_fit:
            duplicates = dataset[dataset.index.duplicated(keep=False)]
            if not duplicates.empty:
                print("Duplicate data points:")
                print(duplicates)

        n_before = dataset.shape[0]
        dataset = dataset[~dataset.index.duplicated(keep="first")]
        print(f"✅ {n_before - dataset.shape[0]} duplicate rows removed!")

        # Steps 3-5: structural encodings
        dataset = self._timezone_mapping(dataset)
        dataset = self._parse_vars(dataset)
        dataset = self._group_landcover_cat(dataset)
        print("✅ Parsed timezone and categorical variables!")

        # Step 6: replace sentinel missing-value codes
        dataset.replace(-999.0, np.nan, inplace=True)
        dataset.replace(-9999.0, np.nan, inplace=True)
        if "altitude" in dataset.columns:
            dataset["altitude"].replace(9999.0, np.nan, inplace=True)
        print("✅ Sentinel values (-999, -9999, 9999) replaced with NaN!")

        # Step 7: filter rows with too many NaN in key columns
        present_nan_subset = [c for c in self.nan_subset if c in dataset.columns]
        print(f"Initial data size: {init_shape}")
        n_before = dataset.shape[0]
        if self.filter_nan == "all":
            dataset = dataset[~dataset[present_nan_subset].isna().all(axis=1)]
        else:
            dataset = dataset[~dataset[present_nan_subset].isna().any(axis=1)]
        n_dropped_nan = n_before - dataset.shape[0]
        print(f"✅ NaN filter (filter_nan={self.filter_nan!r}): {n_dropped_nan} rows dropped ({dataset.shape[0]} remaining).")

        # Step 8: cross-fill altitude <-> topography srtm
        if "altitude" in dataset.columns and "mean_topography_srtm_alt_90m_year1994" in dataset.columns:
            dataset["altitude"] = dataset["altitude"].fillna(
                dataset["mean_topography_srtm_alt_90m_year1994"]
            )
            dataset["mean_topography_srtm_alt_90m_year1994"] = dataset[
                "mean_topography_srtm_alt_90m_year1994"
            ].fillna(dataset["altitude"])

        return dataset

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "TOARProcessing":
        """Learn imputers from training data.

        Args:
            X: Raw station metadata DataFrame.
            y: Ignored (sklearn Pipeline compatibility).

        Returns:
            self
        """
        dataset = X.copy()
        dataset = self._apply_structural_cleaning(dataset, is_fit=True)

        present_num_vars = [c for c in self._num_vars if c in dataset.columns]
        cat_cols = [c for c in self._cat_vars if c in dataset.columns]
        cat_cols_to_impute = [
            c for c in cat_cols if c not in ("area_code", "timezone", "type_of_area")
        ]

        # Step 9: fit numeric imputer
        if self.impute_nan:
            if self.num_imputer == "mean":
                self._num_simple_imputer = SimpleImputer(strategy="mean")
                self._num_simple_imputer.fit(dataset[present_num_vars])

            # Step 10: fit categorical imputer
            if cat_cols_to_impute:
                self._cat_imputer = SimpleImputer(strategy="most_frequent")
                self._cat_imputer.fit(dataset[cat_cols_to_impute])
        else:
            print("⏭️  Imputation skipped (impute_nan=False).")

        self._is_fitted = True
        print("✅ TOARProcessing fitted successfully!")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned preprocessing to new data.

        Args:
            X: Raw station metadata DataFrame.

        Returns:
            Preprocessed DataFrame.

        Raises:
            RuntimeError: If transform() is called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        dataset = X.copy()
        dataset = self._apply_structural_cleaning(dataset, is_fit=False)

        present_num_vars = [c for c in self._num_vars if c in dataset.columns]
        cat_cols = [c for c in self._cat_vars if c in dataset.columns]
        cat_cols_to_impute = [
            c for c in cat_cols if c not in ("area_code", "timezone", "type_of_area")
        ]

        # Step 9: apply numeric imputation
        if self.impute_nan:
            if self.num_imputer == "mean" and self._num_simple_imputer is not None:
                dataset[present_num_vars] = self._num_simple_imputer.transform(
                    dataset[present_num_vars]
                )
            else:
                dataset = self._iterative_imputer(
                    dataset, subset_to_impute=present_num_vars, estimator=self.num_imputer
                )
            print("✅ Numeric missing values imputed!")

            # Step 10: apply categorical imputation
            if cat_cols_to_impute and self._cat_imputer is not None:
                dataset[cat_cols_to_impute] = self._cat_imputer.transform(
                    dataset[cat_cols_to_impute]
                )
            print("✅ Categorical missing values imputed!")
        else:
            print("⏭️  Imputation skipped (impute_nan=False).")

        # Step 11: remove rows with negative road distance (data quality)
        if "distance_to_major_road_year2020" in dataset.columns:
            n_before = dataset.shape[0]
            dataset = dataset[dataset["distance_to_major_road_year2020"] >= 0]
            n_dropped_road = n_before - dataset.shape[0]
            print(f"✅ Negative road distance filter: {n_dropped_road} rows dropped ({dataset.shape[0]} remaining).")

        # Step 12: cast categorical columns to str
        for col in cat_cols:
            dataset[col] = dataset[col].astype(str)
        print("✅ Categorical columns cast to str dtype!")

        print(f"✅ Preprocessing complete! Shape: {dataset.shape}")
        return dataset

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        """Fit on X then transform X in a single pass (avoids running cleaning twice).

        Args:
            X: Raw station metadata DataFrame.
            y: Ignored (sklearn Pipeline compatibility).

        Returns:
            Preprocessed DataFrame.
        """
        dataset = X.copy()
        dataset = self._apply_structural_cleaning(dataset, is_fit=True)

        present_num_vars = [c for c in self._num_vars if c in dataset.columns]
        cat_cols = [c for c in self._cat_vars if c in dataset.columns]
        cat_cols_to_impute = [
            c for c in cat_cols if c not in ("area_code", "timezone", "type_of_area")
        ]

        # Numeric imputation (fit + transform)
        if self.impute_nan:
            if self.num_imputer == "mean":
                self._num_simple_imputer = SimpleImputer(strategy="mean")
                dataset[present_num_vars] = self._num_simple_imputer.fit_transform(
                    dataset[present_num_vars]
                )
            else:
                dataset = self._iterative_imputer(
                    dataset, subset_to_impute=present_num_vars, estimator=self.num_imputer
                )
            print("✅ Numeric missing values imputed!")

            # Categorical imputation (fit + transform)
            if cat_cols_to_impute:
                self._cat_imputer = SimpleImputer(strategy="most_frequent")
                dataset[cat_cols_to_impute] = self._cat_imputer.fit_transform(
                    dataset[cat_cols_to_impute]
                )
            print("✅ Categorical missing values imputed!")
        else:
            print("⏭️  Imputation skipped (impute_nan=False).")

        # Road distance quality filter
        if "distance_to_major_road_year2020" in dataset.columns:
            n_before = dataset.shape[0]
            dataset = dataset[dataset["distance_to_major_road_year2020"] >= 0]
            n_dropped_road = n_before - dataset.shape[0]
            print(f"✅ Negative road distance filter: {n_dropped_road} rows dropped ({dataset.shape[0]} remaining).")

        # Cast categoricals to str
        for col in cat_cols:
            dataset[col] = dataset[col].astype(str)
        print("✅ Categorical columns cast to str dtype!")

        self._is_fitted = True
        print(f"✅ Preprocessing complete! Shape: {dataset.shape}")
        return dataset

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path = "models/processor.pkl") -> None:
        """Persist the fitted processor to disk using joblib.

        Should be called after fit() or fit_transform() so that the learned
        imputers are serialised and can be reused at inference time without
        re-fitting on training data.

        Args:
            path: Destination file path. Defaults to 'models/processor.pkl'.

        Raises:
            RuntimeError: If the processor has not been fitted yet.

        Example:
            >>> processor = TOARProcessing()
            >>> processor.fit_transform(train_df)
            >>> processor.save()                        # saves to models/processor.pkl
            >>> processor.save('models/proc_v2.pkl')   # custom path
        """
        if not self._is_fitted:
            raise RuntimeError("Processor must be fitted before saving. Call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"✅ Processor saved to '{path}'")

    @classmethod
    def load(cls, path: str | Path = "models/processor.pkl") -> "TOARProcessing":
        """Load a previously fitted processor from disk.

        Use this at inference or test time to apply the exact same
        transformations that were learned on the training data.

        Args:
            path: Path to the saved .pkl file. Defaults to 'models/processor.pkl'.

        Returns:
            A fitted TOARProcessing instance.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.

        Example:
            >>> processor = TOARProcessing.load('models/processor.pkl')
            >>> df_clean = processor.transform(new_raw_df)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No processor found at '{path}'. Train and save one first.")
        processor = joblib.load(path)
        print(f"✅ Processor loaded from '{path}'")
        return processor


# ---------------------------------------------------------------------------
# Train / test split helper
# ---------------------------------------------------------------------------
def prepare_train_test_data(
    df: pd.DataFrame,
    train_on_labeled_data: bool = False,
    hand_label_station: pd.DataFrame | None = None,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Split preprocessed data into training and test sets.

    Separates labeled from unlabeled stations. Optionally reserves
    hand-labeled stations as a dedicated test set before the random split.

    Args:
        df: Preprocessed DataFrame with (lat, lon) as MultiIndex and a
            'type_of_area' column.
        train_on_labeled_data: If True, train only on labeled stations.
            If False (default), combine labeled and unlabeled for training.
        hand_label_station: DataFrame of manually-labeled stations to use
            as an additional held-out test set (optional).
        test_size: Fraction of labeled data reserved for the random test split.

    Returns:
        df_train: Training DataFrame.
        df_test: Random-split test DataFrame (labeled only).
        hand_labeled_test_data: Held-out hand-labeled test DataFrame, or
            None if hand_label_station was not provided.

    Raises:
        ValueError: If no labeled data remains after extracting hand-labeled
            stations.
    """
    labelled_data = df[df["type_of_area"] != "unknown"].copy()
    unlabelled_data = df[df["type_of_area"] == "unknown"].copy()

    print(f"Total stations      : {df.shape[0]}")
    print(f"Labeled stations    : {labelled_data.shape[0]}")
    print(f"Unlabeled stations  : {unlabelled_data.shape[0]}")

    hand_labeled_test_data: pd.DataFrame | None = None

    if hand_label_station is not None:
        # Ensure hand_label_station shares the same (lat, lon) MultiIndex
        if not (
            isinstance(hand_label_station.index, pd.MultiIndex)
            and hand_label_station.index.names == ["lat", "lon"]
        ):
            if "lat" in hand_label_station.columns and "lon" in hand_label_station.columns:
                hand_label_station = hand_label_station.set_index(["lat", "lon"])
            else:
                print("Warning: hand_label_station has no lat/lon columns — skipping.")
                hand_label_station = None

        if hand_label_station is not None:
            shared_idx = hand_label_station.index.intersection(labelled_data.index)
            if not shared_idx.empty:
                hand_labeled_test_data = labelled_data.loc[shared_idx].copy()
                labelled_data = labelled_data.drop(shared_idx)
                print(f"Reserved {len(shared_idx)} hand-labeled stations for testing.")
            else:
                print("Warning: No matching stations found in hand_label_station — skipping.")

    if labelled_data.empty:
        raise ValueError("No labeled data remaining after extracting hand-labeled stations.")

    df_train_labeled, df_test = train_test_split(
        labelled_data,
        test_size=test_size,
        stratify=labelled_data["type_of_area"],
        random_state=42,
    )

    if not train_on_labeled_data:
        df_train = pd.concat([df_train_labeled, unlabelled_data])
        print("Training on labeled + unlabeled data.")
    else:
        df_train = df_train_labeled
        print("Training on labeled data only.")

    print(f"Train shape : {df_train.shape}")
    print(f"Test shape  : {df_test.shape}")
    if hand_labeled_test_data is not None:
        print(f"Hand-labeled test shape : {hand_labeled_test_data.shape}")

    return df_train, df_test, hand_labeled_test_data


# ---------------------------------------------------------------------------
# Statistical-evaluation data helpers
# ---------------------------------------------------------------------------

def read_stat_data(
    data_dir: str | Path = "data",
    percentile: str = "p75",
    species: str | list[str] | None = None,
) -> pd.DataFrame:
    """Load per-species percentile CSVs and merge them into a single DataFrame.

    Each CSV is expected to live at ``<data_dir>/<species>/<percentile>.csv``
    and to contain at least ``lat``, ``lon``, and ``value`` columns.
    Duplicate (lat, lon) pairs are removed (both entries dropped) to avoid
    ambiguous merges, which matches the original notebook logic.

    Args:
        data_dir: Root data directory (absolute or relative to the working
            directory).  Defaults to ``'data'``.
        percentile: Filename stem to load, e.g. ``'median'``, ``'p75'``,
            ``'p90'``, ``'p95'``.  Defaults to ``'p75'``.
        species: Species sub-directory (or list of sub-directories) to load.
            Accepts a single string (e.g. ``'no2'``) or a list of strings
            (e.g. ``['no2', 'nox']``).  Defaults to ``['no2', 'nox', 'pm2p5']``
            when *None*.

    Returns:
        DataFrame indexed by ``(lat, lon)`` with one column per species.
        When multiple species are requested the result contains only rows
        present in **all** DataFrames (inner join).  When a single species
        string is passed the result is a one-column DataFrame (no join needed).

    Examples:
        >>> df_stat = read_stat_data(data_dir='data', percentile='p75')
        >>> df_stat.columns
        Index(['no2', 'nox', 'pm2p5'], dtype='object')

        >>> df_no2 = read_stat_data(data_dir='data', species='no2')
        >>> df_no2.columns
        Index(['no2'], dtype='object')
    """
    if species is None:
        species = ["no2", "nox", "pm2p5"]
    elif isinstance(species, str):
        species = [species]

    data_dir = Path(data_dir)

    def _load_species(name: str) -> pd.DataFrame:
        path = data_dir / name / f"{percentile}.csv"
        df = pd.read_csv(path, comment="#")
        df.set_index(["lat", "lon"], inplace=True)
        # Drop both entries when duplicates exist (keep=False)
        df = df.loc[~df.index.duplicated(keep=False)]
        df = df.rename(columns={"value": name})
        df = df[[name]]          # keep only the species column
        df = df.dropna()
        return df

    frames = [_load_species(s) for s in species]
    df_merged = frames[0].join(frames[1:], how="inner")
    return df_merged

def prepare_data_for_stat_eval(
    df_clean: pd.DataFrame,
    df_spice: pd.DataFrame,
    know_station_only: bool = False,
    n_sample: int | None = 1000,
) -> pd.DataFrame:
    """Merge station metadata with species statistics and optionally sub-sample.

    Joins *df_clean* (station features + ``'type_of_area'``) with *df_spice*
    (species percentile values) on their shared ``(lat, lon)`` index, then
    optionally restricts to known station types and draws a balanced sample.

    Args:
        df_clean: Pre-processed station metadata DataFrame indexed by
            ``(lat, lon)``.  Must contain a ``'type_of_area'`` column.
        df_spice: Species statistics DataFrame indexed by ``(lat, lon)``.
            Typically the output of :func:`read_stat_data`.
        know_station_only: If True, rows labelled ``'unknown'`` are removed
            before sampling.  Defaults to False.
        n_sample: Total number of rows to return when *know_station_only* is
            True.  The sample is balanced across the three station types:
            urban, suburban, and rural.  Set to ``None`` to return all rows
            without sub-sampling.  Defaults to 1000.

    Returns:
        Merged (and optionally sampled) DataFrame.

    Example:
        >>> df_stat = read_stat_data()
        >>> df_eval = prepare_data_for_stat_eval(df_clean, df_stat,
        ...                                      know_station_only=True,
        ...                                      n_sample=900)
    """
    df = pd.merge(df_clean, df_spice, left_index=True, right_index=True)
    print(df["type_of_area"].value_counts())
    print("Total data points:", df.shape)

    if know_station_only:
        df = df[df["type_of_area"] != "unknown"]

        if n_sample is not None:
            n_indiv = int(n_sample / 3)
            n_rural    = min(len(df[df["type_of_area"] == "rural"]),    n_indiv)
            n_suburban = min(len(df[df["type_of_area"] == "suburban"]), n_indiv)
            n_urban    = n_sample - n_rural - n_suburban
            print(f"Sampling – urban: {n_urban}, suburban: {n_suburban}, rural: {n_rural}")
            df = pd.concat([
                df[df["type_of_area"] == "urban"].sample(n=n_urban,    random_state=42),
                df[df["type_of_area"] == "suburban"].sample(n=n_suburban, random_state=42),
                df[df["type_of_area"] == "rural"].sample(n=n_rural,    random_state=42),
            ])

    return df


if __name__ == "__main__":
    print("This module defines the TOARProcessing class and helper functions for data cleaning and preprocessing. It is not meant to be run directly.")

