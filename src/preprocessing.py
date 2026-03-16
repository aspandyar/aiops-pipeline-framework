"""
Data processing and feature engineering for TravisTorrent CI build data.
Robust to missing columns across different TravisTorrent file versions.
Used by Notebook 2 and by the dashboard for live prediction encoding.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Default paths (can be overridden)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def _has_col(df, name):
    return name in df.columns


def build_duration_seconds(df):
    """Build duration in seconds, clipped at 99th percentile. Uses tr_duration or tr_log_buildduration."""
    if _has_col(df, "tr_duration"):
        s = df["tr_duration"].copy()
    elif _has_col(df, "tr_log_buildduration"):
        s = df["tr_log_buildduration"].copy()
    else:
        warnings.warn("Neither tr_duration nor tr_log_buildduration found; using 0.")
        return pd.Series(0, index=df.index)
    cap = s.quantile(0.99)
    return s.clip(upper=cap).fillna(0)


def commits_per_build(df):
    """Number of commits per build. Prefer git_num_all_built_commits, else gh_num_commits_in_push."""
    if _has_col(df, "git_num_all_built_commits"):
        return df["git_num_all_built_commits"].fillna(0).astype(int)
    if _has_col(df, "gh_num_commits_in_push"):
        return df["gh_num_commits_in_push"].fillna(0).astype(int)
    warnings.warn("No commits-per-build column found; using 0.")
    return pd.Series(0, index=df.index)


def lines_of_code_changed(df):
    """Lines of code changed = git_diff_src_churn + git_diff_test_churn."""
    out = pd.Series(0, index=df.index)
    if _has_col(df, "git_diff_src_churn"):
        out = out + df["git_diff_src_churn"].fillna(0)
    else:
        warnings.warn("git_diff_src_churn not found.")
    if _has_col(df, "git_diff_test_churn"):
        out = out + df["git_diff_test_churn"].fillna(0)
    else:
        warnings.warn("git_diff_test_churn not found.")
    return out.astype(int)


def temporal_features(df):
    """From gh_build_started_at: hour_of_day, day_of_week, is_weekend."""
    if not _has_col(df, "gh_build_started_at"):
        warnings.warn("gh_build_started_at not found; temporal features set to 0/False.")
        return (
            pd.Series(0, index=df.index),
            pd.Series(0, index=df.index),
            pd.Series(False, index=df.index),
        )
    ts = pd.to_datetime(df["gh_build_started_at"], errors="coerce")
    hour = ts.dt.hour.fillna(0).astype(int)
    dow = ts.dt.dayofweek.fillna(0).astype(int)
    weekend = dow >= 5
    return hour, dow, weekend


def test_pass_rate(df):
    """Test pass rate = passing / total, with safe fallback when no tests run."""
    ok = df["tr_log_num_tests_ok"].fillna(0) if _has_col(df, "tr_log_num_tests_ok") else 0
    total = df["tr_log_num_tests_run"].fillna(0) if _has_col(df, "tr_log_num_tests_run") else 0
    if isinstance(ok, int):
        ok = pd.Series(ok, index=df.index)
    if isinstance(total, int):
        total = pd.Series(total, index=df.index)
    rate = np.where(total > 0, ok / total, 1.0)
    return pd.Series(rate, index=df.index)


def previous_build_failed(df):
    """Per-project lag: whether the immediately preceding build (by tr_prev_build) also failed."""
    if not _has_col(df, "tr_prev_build") or not _has_col(df, "tr_status"):
        warnings.warn("tr_prev_build or tr_status not found; previous_build_failed = False.")
        return pd.Series(False, index=df.index)
    status = df["tr_status"].astype(str).str.strip().str.lower()
    failed = status.isin(["failed", "errored"])
    build_id = df["tr_build_id"] if _has_col(df, "tr_build_id") else df.index
    id_to_failed = dict(zip(build_id, failed))
    prev_ids = df["tr_prev_build"]
    prev_failed = [id_to_failed.get(int(pid), False) if pd.notna(pid) else False for pid in prev_ids]
    return pd.Series(prev_failed, index=df.index)


def process_raw_to_features(df, target_col="tr_status"):
    """
    Build the engineered feature DataFrame and target Series.
    Handles missing columns; prints messages for missing sources.
    """
    out = pd.DataFrame(index=df.index)

    out["build_duration_sec"] = build_duration_seconds(df)
    out["commits_per_build"] = commits_per_build(df)
    out["loc_changed"] = lines_of_code_changed(df)

    hour, dow, weekend = temporal_features(df)
    out["hour_of_day"] = hour
    out["day_of_week"] = dow
    out["is_weekend"] = weekend.astype(int)

    if _has_col(df, "tr_log_num_tests_ok") and _has_col(df, "tr_log_num_tests_run"):
        out["test_pass_rate"] = test_pass_rate(df)
    else:
        warnings.warn("Test columns not found; test_pass_rate = 1.0")
        out["test_pass_rate"] = 1.0

    out["previous_build_failed"] = previous_build_failed(df).astype(int)

    # Optional: repo/team size and codebase size (predictive in CI literature)
    if _has_col(df, "gh_team_size"):
        out["gh_team_size"] = df["gh_team_size"].fillna(0).astype(int)
    if _has_col(df, "gh_sloc"):
        sloc = df["gh_sloc"].replace(0, np.nan)
        out["gh_sloc_log"] = np.log1p(sloc.fillna(0)).replace(0, 0)  # log scale for heavy tail
    if _has_col(df, "gh_repo_age"):
        out["gh_repo_age"] = df["gh_repo_age"].fillna(0)

    # Optional: test volume (number of tests run) — predictive of outcome
    if _has_col(df, "tr_log_num_tests_run"):
        n_run = df["tr_log_num_tests_run"].fillna(0).astype(float)
        out["tests_run_log"] = np.log1p(n_run.clip(lower=0))

    if target_col in df.columns:
        y = df[target_col].astype(str).str.strip().str.lower()
        # Normalize: canceled/cancelled
        y = y.replace("cancelled", "canceled")
    else:
        y = None

    return out, y


def fill_and_encode(X, y, cat_columns=None, target_encoder=None, lang_encoder=None):
    """
    Fill numeric nulls with median, label-encode categoricals and target.
    Returns (X_final, y_encoded, target_encoder, lang_encoder), with encoders created if not provided.
    """
    X = X.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        X[c] = X[c].fillna(X[c].median())

    if cat_columns is None:
        cat_columns = []
    if lang_encoder is None and "gh_lang" in X.columns:
        cat_columns = [c for c in cat_columns if c in X.columns]
    elif "gh_lang" in X.columns:
        cat_columns = ["gh_lang"] + [c for c in cat_columns if c in X.columns and c != "gh_lang"]

    if "gh_lang" in X.columns:
        if lang_encoder is None:
            lang_encoder = LabelEncoder()
            X["gh_lang"] = X["gh_lang"].fillna("unknown").astype(str)
            X["gh_lang_enc"] = lang_encoder.fit_transform(X["gh_lang"])
        else:
            X["gh_lang"] = X["gh_lang"].fillna("unknown").astype(str)
            X["gh_lang_enc"] = lang_encoder.transform(X["gh_lang"])
        X = X.drop(columns=["gh_lang"], errors="ignore")

    if y is not None and target_encoder is None:
        target_encoder = LabelEncoder()
        y_enc = target_encoder.fit_transform(y.astype(str))
    elif y is not None:
        y_enc = target_encoder.transform(y.astype(str))
    else:
        y_enc = None

    return X, y_enc, target_encoder, lang_encoder


def get_feature_columns():
    """Ordered list of feature names used for modeling (no target, no ids). Optional columns included if present."""
    return [
        "build_duration_sec",
        "commits_per_build",
        "loc_changed",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "test_pass_rate",
        "previous_build_failed",
        "tests_run_log",
        "gh_team_size",
        "gh_sloc_log",
        "gh_repo_age",
        "gh_lang_enc",
    ]


def prepare_for_training():
    """
    Load processed train/val/test splits and encoders from disk (saved by Notebook 2).
    Returns (X_train, X_val, X_test, y_train, y_val, y_test, feature_list, class_names, encoders_dict).
    """
    import joblib

    data_dir = DATA_PROCESSED
    models_dir = MODELS_DIR

    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_val = pd.read_csv(data_dir / "X_val.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = np.load(data_dir / "y_train.npy")
    y_val = np.load(data_dir / "y_val.npy")
    y_test = np.load(data_dir / "y_test.npy")

    encoders = joblib.load(models_dir / "encoders.joblib")
    feature_list = encoders.get("feature_list", list(X_train.columns))
    class_names = encoders.get("class_names", [])

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_list,
        class_names,
        encoders,
    )
