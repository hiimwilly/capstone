"""
preprocessing.py
================
Reusable data preprocessing pipeline for the capstone e-commerce regression project.

Target variable : ``estimated_lifetime_value`` (right-skewed → log1p-transformed).
Categorical     : ``loyalty_program_membership``  (binary / OHE).
Numeric features: all remaining columns.

Public API
----------
split_X_y(df)
    Separate features from the target column.

transform_target(y)
    Apply log1p to the target Series.

inverse_transform_target(y_transformed)
    Invert log1p (expm1) to recover original scale predictions.

build_numeric_pipeline(scaler)
    Return a sklearn Pipeline for numeric features (impute + optional scale).

build_categorical_pipeline()
    Return a sklearn Pipeline for the loyalty membership column (impute + OHE).

build_preprocessor(scaler, categorical_cols, numeric_cols)
    Return a ColumnTransformer combining numeric and categorical pipelines.

fit_transform_train(X_train, scaler, categorical_cols, numeric_cols)
    Fit preprocessor on training data and transform it; return (preprocessor, X_train_transformed).

transform_new(preprocessor, X)
    Apply an already-fitted preprocessor to new data (validation / test).

preprocess_dataset(df, scaler, test_size, random_state, transform_y)
    End-to-end convenience function: split, fit on train, transform all splits.
"""

from __future__ import annotations

import glob
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL: str = "estimated_lifetime_value"
LOYALTY_COL: str = "loyalty_program_membership"
DEFAULT_DATA_FILENAME: str = "synthetic_data_86 (ecommerce data set).csv"

_SCALER_MAP: dict[str, object] = {
    "standard": StandardScaler(),
    "robust": RobustScaler(),
    "minmax": MinMaxScaler(),
    "none": None,
}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load the capstone dataset into a DataFrame.

    Parameters
    ----------
    path : str, optional
        Explicit path to the CSV file.  If *None*, the function first looks for
        ``DEFAULT_DATA_FILENAME`` in the current working directory, then falls
        back to the first ``*.csv`` it finds there.

    Returns
    -------
    pd.DataFrame
    """
    if path is None:
        candidate = os.path.join(os.getcwd(), DEFAULT_DATA_FILENAME)
        if os.path.exists(candidate):
            path = candidate
        else:
            csvs = glob.glob(os.path.join(os.getcwd(), "*.csv"))
            if not csvs:
                raise FileNotFoundError(
                    f"No CSV file found in {os.getcwd()!r}. "
                    "Pass an explicit 'path' argument to load_data()."
                )
            path = csvs[0]

    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Target helpers
# ---------------------------------------------------------------------------


def split_X_y(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix *X* and target vector *y*.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including the target column.
    target_col : str
        Name of the target column.  Defaults to ``estimated_lifetime_value``.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not found in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def transform_target(y: pd.Series) -> pd.Series:
    """Apply ``log1p`` transformation to the target variable.

    Using ``log1p`` (i.e., ``log(1 + y)``) handles the right-skewed distribution
    of ``estimated_lifetime_value`` and is safe for non-negative values.

    Parameters
    ----------
    y : pd.Series
        Raw target values (must be ≥ 0).

    Returns
    -------
    pd.Series
        Log1p-transformed target.
    """
    return np.log1p(y)


def inverse_transform_target(y_transformed: pd.Series | np.ndarray) -> np.ndarray:
    """Invert the ``log1p`` transformation to recover original-scale predictions.

    Parameters
    ----------
    y_transformed : array-like
        Log1p-transformed predictions.

    Returns
    -------
    np.ndarray
        Predictions in the original scale.
    """
    return np.expm1(y_transformed)


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def _resolve_scaler(scaler: str | object | None) -> object | None:
    """Return a scikit-learn scaler instance from a string alias or pass-through."""
    if isinstance(scaler, str):
        key = scaler.lower()
        if key not in _SCALER_MAP:
            raise ValueError(
                f"Unknown scaler {scaler!r}. "
                f"Choose from: {list(_SCALER_MAP.keys())}."
            )
        return _SCALER_MAP[key]
    return scaler  # already an instance or None


def build_numeric_pipeline(
    scaler: str | object | None = "robust",
) -> Pipeline:
    """Build a Pipeline for numeric features.

    Steps
    -----
    1. ``SimpleImputer`` (median strategy) – handles missing values.
    2. Scaler (optional) – ``RobustScaler`` by default, which is resistant to
       outliers (preferred per EDA findings).

    Parameters
    ----------
    scaler : str or sklearn scaler or None
        * ``"standard"`` → :class:`~sklearn.preprocessing.StandardScaler`
        * ``"robust"``   → :class:`~sklearn.preprocessing.RobustScaler` *(default)*
        * ``"minmax"``   → :class:`~sklearn.preprocessing.MinMaxScaler`
        * ``"none"`` / ``None`` → no scaling

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    scaler_instance = _resolve_scaler(scaler)
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scaler_instance is not None:
        steps.append(("scaler", scaler_instance))
    return Pipeline(steps)


def build_categorical_pipeline() -> Pipeline:
    """Build a Pipeline for the ``loyalty_program_membership`` column.

    Steps
    -----
    1. ``SimpleImputer`` (most_frequent strategy).
    2. ``OneHotEncoder`` – drops the first category to avoid multicollinearity
       (``drop="first"``), yielding a single binary column.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
            ),
        ]
    )


def build_preprocessor(
    scaler: str | object | None = "robust",
    categorical_cols: Optional[list[str]] = None,
    numeric_cols: Optional[list[str]] = None,
) -> ColumnTransformer:
    """Compose a :class:`~sklearn.compose.ColumnTransformer` for the full feature set.

    Parameters
    ----------
    scaler : str or sklearn scaler or None
        Forwarded to :func:`build_numeric_pipeline`.
    categorical_cols : list[str], optional
        Categorical column names.  Defaults to ``[LOYALTY_COL]``.
    numeric_cols : list[str], optional
        Numeric column names.  If *None*, they are inferred at fit-time using
        ``"passthrough"`` for anything not in *categorical_cols*.

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """
    if categorical_cols is None:
        categorical_cols = [LOYALTY_COL]

    transformers: list[tuple] = [
        ("categorical", build_categorical_pipeline(), categorical_cols),
    ]

    if numeric_cols is not None:
        transformers.insert(
            0, ("numeric", build_numeric_pipeline(scaler), numeric_cols)
        )
    else:
        # Let ColumnTransformer handle remaining columns automatically.
        transformers.insert(
            0, ("numeric", build_numeric_pipeline(scaler), _make_selector(categorical_cols))
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _make_selector(exclude_cols: list[str]):
    """Return a sklearn ``make_column_selector``-style callable that excludes given columns."""
    from sklearn.compose import make_column_selector

    # We cannot directly use make_column_selector to exclude names, so we use a
    # lambda that is evaluated at fit-time by ColumnTransformer when the
    # transformers list is built with a callable selector.
    def selector(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in exclude_cols]

    return selector


# ---------------------------------------------------------------------------
# Fit / transform helpers
# ---------------------------------------------------------------------------


def fit_transform_train(
    X_train: pd.DataFrame,
    scaler: str | object | None = "robust",
    categorical_cols: Optional[list[str]] = None,
    numeric_cols: Optional[list[str]] = None,
) -> Tuple[ColumnTransformer, np.ndarray]:
    """Fit the preprocessor on training data and transform it.

    This is the only function that should call ``fit`` or ``fit_transform``.
    Use :func:`transform_new` for validation and test splits to avoid data leakage.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (no target column).
    scaler : str or sklearn scaler or None
        Forwarded to :func:`build_preprocessor`.
    categorical_cols : list[str], optional
        Forwarded to :func:`build_preprocessor`.
    numeric_cols : list[str], optional
        Forwarded to :func:`build_preprocessor`.

    Returns
    -------
    preprocessor : ColumnTransformer
        Fitted preprocessor.
    X_train_transformed : np.ndarray
        Transformed training features.
    """
    preprocessor = build_preprocessor(
        scaler=scaler,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )
    X_train_transformed = preprocessor.fit_transform(X_train)
    return preprocessor, X_train_transformed


def transform_new(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> np.ndarray:
    """Apply an already-fitted preprocessor to new (validation / test) data.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        A preprocessor returned by :func:`fit_transform_train`.
    X : pd.DataFrame
        New feature data (must have the same columns as the training data).

    Returns
    -------
    np.ndarray
        Transformed features.
    """
    return preprocessor.transform(X)


# ---------------------------------------------------------------------------
# End-to-end convenience function
# ---------------------------------------------------------------------------


def preprocess_dataset(
    df: pd.DataFrame,
    scaler: str | object | None = "robust",
    test_size: float = 0.2,
    random_state: int = 42,
    transform_y: bool = True,
    categorical_cols: Optional[list[str]] = None,
    numeric_cols: Optional[list[str]] = None,
    target_col: str = TARGET_COL,
) -> dict:
    """End-to-end preprocessing: split data, fit on train, transform all splits.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    scaler : str or sklearn scaler or None
        Numeric scaler.  Defaults to ``"robust"``.
    test_size : float
        Fraction of data reserved for the test set.
    random_state : int
        Random seed for reproducible splits.
    transform_y : bool
        If *True* (default), apply ``log1p`` to the target.
    categorical_cols : list[str], optional
        Categorical columns (defaults to ``[LOYALTY_COL]``).
    numeric_cols : list[str], optional
        Numeric columns (auto-inferred if *None*).
    target_col : str
        Target column name.

    Returns
    -------
    dict with keys:
        ``preprocessor``       – fitted ColumnTransformer
        ``X_train``            – transformed training features (np.ndarray)
        ``X_test``             – transformed test features (np.ndarray)
        ``y_train``            – training target (log1p if transform_y)
        ``y_test``             – test target (log1p if transform_y)
        ``y_train_raw``        – untransformed training target
        ``y_test_raw``         – untransformed test target
    """
    from sklearn.model_selection import train_test_split

    X, y = split_X_y(df, target_col=target_col)

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor, X_train_transformed = fit_transform_train(
        X_train,
        scaler=scaler,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )
    X_test_transformed = transform_new(preprocessor, X_test)

    y_train = transform_target(y_train_raw) if transform_y else y_train_raw.values
    y_test = transform_target(y_test_raw) if transform_y else y_test_raw.values

    return {
        "preprocessor": preprocessor,
        "X_train": X_train_transformed,
        "X_test": X_test_transformed,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_raw": y_train_raw.values,
        "y_test_raw": y_test_raw.values,
    }
