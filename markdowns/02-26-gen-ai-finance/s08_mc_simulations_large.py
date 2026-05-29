import re
import ast
import json
import copy
import typing
import hashlib
import argparse
import pathlib
import pickle
import warnings
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from scipy.special import expit

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
)
from sklearn.model_selection import ParameterGrid
from sklearn.exceptions import ConvergenceWarning


DEFAULT_TFIDF_GRID = {
    "pre__tfidf__ngram_range": [(1, 2)],
    "pre__tfidf__min_df": [3],
    "pre__tfidf__max_df": [0.8],
    "pre__tfidf__max_features": [10000],
    "pre__tfidf__sublinear_tf": [False],
    "pre__tfidf__stop_words": ["english"],
    "pre__tfidf__binary": [False],
}


# ------------------
# CLI
# ------------------
ALPHA_GRID = [1.0, 10.0, 50.0, 100.0, 150.0, 200.0, 350.0, 500.0, 750.0, 1000.0]  # larger is higher prior sample size
L2_GRID = [1e-3, 0.0005, 1e-2, 0.05, 0.1, 0.5, 1, 10, 100]  # smaller is larger regularization
def parse_args(verbose: bool = False) -> typing.Dict[str, typing.Any]:
    parser = argparse.ArgumentParser("Monte Carlo TF-IDF + Bayesian bootstrap / L2 benchmark")
    parser.add_argument(
        "--model_output_dir",
        default=pathlib.Path.home().joinpath(pathlib.Path("EL", "models", "gen-ai-finance")),
    )
    parser.add_argument("--b", type=int, default=2)
    parser.add_argument("--m", type=int, default=3000)
    parser.add_argument("--n_sims", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=12345)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--y_col", type=str, default="overnight_sign")
    parser.add_argument("--id_col", type=str, default="rp_entity_id")
    parser.add_argument("--save_prefix", type=str, default="large_tfidf_genel_v1")
    parser.set_defaults(rescale_alpha_on_refit=True)
    parser.add_argument(
        "--no_rescale_alpha_on_refit",
        action="store_false",
        dest="rescale_alpha_on_refit",
        help="Use the selected alpha directly on train+val instead of rescaling by sample size.",
    )
    arguments = vars(parser.parse_args())
    return arguments


# ------------------
# Text / source feature prep
# ------------------
def _join_headlines(x):
    if isinstance(x, list):
        return " . ".join([str(s) for s in x if isinstance(s, str) and len(s) > 0])
    if pd.isna(x):
        return ""
    return str(x)


def _normalize_source_token(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return f"src_{s}" if s else "src_unknown"


def _sources_to_doc(x):
    if isinstance(x, list):
        toks = [_normalize_source_token(s) for s in x if not pd.isna(s)]
        return " ".join(toks)
    if pd.isna(x):
        return ""
    return _normalize_source_token(str(x))


def prepare_baseline1_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text_doc"] = out["headlines"].apply(_join_headlines)
    out["source_doc"] = out["sources"].apply(_sources_to_doc)
    return out


# ------------------
# Utils
# ------------------
def _load_firm_date_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["headlines", "sources"]:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)
    return df


def _maybe_normalize_aug_df(aug_df: pd.DataFrame) -> pd.DataFrame:
    out = aug_df.copy()
    if "n_headlines_generated" in out.columns and "n_headlines_unique" not in out.columns:
        out = out.rename(columns={"n_headlines_generated": "n_headlines_unique"})
    return out


def _stable_seed_from_params(random_state: int, params: Mapping[str, Any]) -> int:
    s = "|".join(f"{k}={repr(v)}" for k, v in sorted(params.items()))
    h = int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)
    return (int(random_state) + h) % (2**32)


def _config_to_json(params: Mapping[str, Any]) -> str:
    return json.dumps(dict(params), sort_keys=True, default=str)


def _flatten_tfidf_params(tfidf_params: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in tfidf_params.items()}


def _selection_score(x: float) -> float:
    return -np.inf if pd.isna(x) else float(x)


def _ensure_csr(X: Any) -> csr_matrix:
    return X.tocsr() if hasattr(X, "tocsr") else csr_matrix(X)


def _normalize_alpha_grid(alpha_grid: Optional[List[float]] = None) -> List[float]:
    grid = list(ALPHA_GRID if alpha_grid is None else alpha_grid)
    out = sorted(set(float(a) for a in ([0.0] + grid)))
    if any(a < 0 for a in out):
        raise ValueError(f"alpha_grid must be nonnegative. Got: {out}")
    return out


def _infer_synthetic_label_cols(synthetic_label_df: pd.DataFrame) -> List[str]:
    sign_cols = [
        c for c in synthetic_label_df.columns
        if re.fullmatch(r"synthetic_overnight_sign_\d{3}", str(c))
    ]
    if len(sign_cols) > 0:
        return sorted(sign_cols)

    z_cols = [
        c for c in synthetic_label_df.columns
        if re.fullmatch(r"synthetic_z_\d{3}", str(c))
    ]
    if len(z_cols) > 0:
        return sorted(z_cols)

    raise ValueError(
        "Could not find synthetic label columns. Expected columns like "
        "synthetic_overnight_sign_001 or synthetic_z_001."
    )


def _align_synthetic_label_df(
    train_df: pd.DataFrame,
    synthetic_label_df: pd.DataFrame,
    id_col: str = "rp_entity_id",
) -> pd.DataFrame:
    base = train_df.reset_index(drop=True).copy()
    synth = synthetic_label_df.reset_index(drop=True).copy()

    key_cols = [c for c in [id_col, "trade_date"] if c in base.columns and c in synth.columns]

    if len(base) == len(synth):
        if not key_cols:
            return synth
        keys_match_in_order = all(base[c].astype(str).equals(synth[c].astype(str)) for c in key_cols)
        if keys_match_in_order:
            return synth

    if not key_cols:
        raise ValueError(
            "Synthetic-label dataframe could not be aligned to train_df by row order, "
            "and alignment keys were unavailable. Expected matching columns such as "
            f"{id_col!r} and 'trade_date'."
        )

    left = base[key_cols].copy()
    left["_row_order"] = np.arange(len(base))

    try:
        merged = left.merge(synth, on=key_cols, how="left", sort=False, validate="one_to_one")
    except Exception as exc:
        raise ValueError(
            "Failed to align synthetic_label_df to train_df using keys "
            f"{key_cols}."
        ) from exc

    merged = merged.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)

    if len(merged) != len(base):
        raise ValueError(
            "Aligned synthetic-label dataframe has wrong number of rows: "
            f"expected {len(base)}, got {len(merged)}."
        )

    return merged


def _extract_synthetic_label_matrix(
    train_df: pd.DataFrame,
    synthetic_label_df: pd.DataFrame,
    id_col: str = "rp_entity_id",
) -> Tuple[List[str], np.ndarray]:
    aligned = _align_synthetic_label_df(
        train_df=train_df,
        synthetic_label_df=synthetic_label_df,
        id_col=id_col,
    )
    label_cols = _infer_synthetic_label_cols(aligned)
    raw = aligned[label_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if len(label_cols) == 0:
        raise ValueError("No synthetic label columns were found after alignment.")

    if label_cols[0].startswith("synthetic_overnight_sign_"):
        non_missing = raw[~np.isnan(raw)]
        if non_missing.size > 0:
            bad_values = np.setdiff1d(np.unique(non_missing), np.array([0.0, 1.0]))
            if bad_values.size > 0:
                raise ValueError(
                    "synthetic_overnight_sign_* columns must contain only {0, 1, NaN}. "
                    f"Found values like {bad_values[:10].tolist()}."
                )
        label_matrix = raw
    else:
        warnings.warn(
            "synthetic_overnight_sign_* columns were not found; deriving binary labels from "
            "synthetic_z_* using the threshold z > 0.",
            RuntimeWarning,
        )
        label_matrix = np.where(np.isnan(raw), np.nan, (raw > 0.0).astype(float))

    if label_matrix.shape[0] != len(train_df):
        raise ValueError(
            "Synthetic label matrix row count does not match train_df: "
            f"expected {len(train_df)}, got {label_matrix.shape[0]}."
        )
    if np.all(np.isnan(label_matrix)):
        raise ValueError("Synthetic label matrix is entirely missing.")

    n_missing = int(np.isnan(label_matrix).sum())
    n_all_missing_rows = int(np.isnan(label_matrix).all(axis=1).sum())
    if n_missing > 0:
        warnings.warn(
            "Synthetic-label dataframe contains missing values. "
            f"There are {n_missing} missing entries and {n_all_missing_rows} rows with all labels missing. "
            "The synthetic-label augmentation method will sample only from rows with at least one available label.",
            RuntimeWarning,
        )

    return label_cols, label_matrix


def _build_synthetic_label_candidate_lists(
    synthetic_label_matrix: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray]:
    mat = np.asarray(synthetic_label_matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError(
            "synthetic_label_matrix must be 2-dimensional. "
            f"Got shape {mat.shape}."
        )

    label_candidates: List[np.ndarray] = []
    counts = np.zeros(mat.shape[0], dtype=np.int64)
    for i in range(mat.shape[0]):
        vals = mat[i]
        avail = vals[~np.isnan(vals)].astype(int, copy=False)
        label_candidates.append(avail)
        counts[i] = int(len(avail))

    return label_candidates, counts


def _sample_empirical_synthetic_label_draws(
    label_candidates: List[np.ndarray],
    eligible_row_idx: np.ndarray,
    b: int,
    m: int,
    rng_rows: np.random.Generator,
    rng_labels: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if b <= 0:
        raise ValueError(f"b must be positive. Got {b}.")
    if m <= 0:
        raise ValueError(f"m must be positive. Got {m}.")
    if len(eligible_row_idx) == 0:
        raise ValueError("No rows have at least one available synthetic label.")

    sampled_row_idx = rng_rows.choice(eligible_row_idx, size=(b, m), replace=True)
    sampled_y = np.empty((b, m), dtype=np.int8)

    for k in range(b):
        rows_k = sampled_row_idx[k]
        sampled_y[k] = np.fromiter(
            (
                int(label_candidates[row_idx][rng_labels.integers(len(label_candidates[row_idx]))])
                for row_idx in rows_k
            ),
            dtype=np.int8,
            count=m,
        )

    return sampled_row_idx, sampled_y


def _safe_fit_logistic(
    clf: LogisticRegression,
    X: csr_matrix,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> LogisticRegression:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        clf.fit(X, y, sample_weight=sample_weight)
    return clf


def _metrics_from_proba(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)
    proba = np.clip(proba, 1e-12, 1 - 1e-12)
    y_pred = (proba >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, proba))
    else:
        auc = float("nan")

    out = {
        "n_obs": int(len(y_true)),
        "pos_rate": float(y_true.mean()) if len(y_true) > 0 else float("nan"),
        "auc": auc,
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    return out


def _expand_tfidf_configs(
    tfidf_grid: Optional[Dict[str, List[Any]]] = None,
) -> List[Dict[str, Any]]:
    merged = dict(DEFAULT_TFIDF_GRID)
    if tfidf_grid is not None:
        clean_grid = dict(tfidf_grid)
        for bad_key in ["alpha", "c_value", "C", "l2_grid", "l2_c"]:
            clean_grid.pop(bad_key, None)
        merged.update(clean_grid)
    return list(ParameterGrid(merged))


# ------------------
# TF-IDF component
# ------------------
def _build_preprocessor(
    tfidf_params: Dict[str, Any],
    train_m: pd.DataFrame,
    use_numeric_features: bool = True,
) -> Tuple[ColumnTransformer, List[str]]:
    candidate_numeric = ["n_headlines_unique", "n_sources", "avg_relevance", "pr_proportion"]
    numeric_cols = [c for c in candidate_numeric if (use_numeric_features and c in train_m.columns)]

    transformers = []

    tfidf = TfidfVectorizer(
        ngram_range=tfidf_params["pre__tfidf__ngram_range"],
        min_df=tfidf_params["pre__tfidf__min_df"],
        max_df=tfidf_params["pre__tfidf__max_df"],
        max_features=tfidf_params["pre__tfidf__max_features"],
        sublinear_tf=tfidf_params["pre__tfidf__sublinear_tf"],
        stop_words=tfidf_params["pre__tfidf__stop_words"],
        binary=tfidf_params["pre__tfidf__binary"],
    )
    transformers.append(("tfidf", tfidf, "text_doc"))
    transformers.append(("src", CountVectorizer(token_pattern=r"(?u)\b\w+\b"), "source_doc"))

    if len(numeric_cols) > 0:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("num", num_pipe, numeric_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    pre.fit(train_m)
    return pre, numeric_cols


# ------------------
# Bayesian-bootstrap / ETEL component
# ------------------
def _fit_augmented_betas(
    X_train: csr_matrix,
    y_train: np.ndarray,
    X_aug_pool: csr_matrix,
    y_aug_pool: np.ndarray,
    aug_pos_draws: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
    random_state: int,
    max_iter: int = 2000,
    tol: float = 1e-3,
) -> np.ndarray:
    if float(alpha) <= 0:
        raise ValueError("_fit_augmented_betas requires alpha > 0. Use _fit_bayes_bootstrap_betas for alpha=0.")

    n = X_train.shape[0]
    b = aug_pos_draws.shape[0]
    m = aug_pos_draws.shape[1]
    p = X_train.shape[1]

    dir_a = np.concatenate([
        np.ones(n, dtype=float),
        np.full(m, float(alpha) / float(m), dtype=float),
    ])
    scale = float(n + alpha)
    betas = np.zeros((b, p + 1), dtype=np.float32)

    clf = LogisticRegression(
        penalty=None,
        solver="saga",
        fit_intercept=True,
        warm_start=True,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        n_jobs=1,
    )

    try:
        _safe_fit_logistic(clf, X_train, y_train)
    except Exception:
        pass

    for k in range(b):
        pos = aug_pos_draws[k]
        X_aug_k = X_aug_pool[pos]
        y_aug_k = y_aug_pool[pos]

        X_k = vstack([X_train, X_aug_k], format="csr")
        y_k = np.concatenate([y_train, y_aug_k])

        w = rng.dirichlet(dir_a) * scale
        _safe_fit_logistic(clf, X_k, y_k, sample_weight=w)

        betas[k, 0] = clf.intercept_[0]
        betas[k, 1:] = clf.coef_[0].astype(np.float32, copy=False)

    return betas


def _fit_bayes_bootstrap_betas(
    X_train: csr_matrix,
    y_train: np.ndarray,
    b: int,
    rng: np.random.Generator,
    random_state: int,
    max_iter: int = 2000,
    tol: float = 1e-3,
) -> np.ndarray:
    n = X_train.shape[0]
    p = X_train.shape[1]
    dir_a = np.ones(n, dtype=float)
    scale = float(n)
    betas = np.zeros((b, p + 1), dtype=np.float32)

    clf = LogisticRegression(
        penalty=None,
        solver="saga",
        fit_intercept=True,
        warm_start=True,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        n_jobs=1,
    )

    try:
        _safe_fit_logistic(clf, X_train, y_train)
    except Exception:
        pass

    for k in range(b):
        w = rng.dirichlet(dir_a) * scale
        _safe_fit_logistic(clf, X_train, y_train, sample_weight=w)
        betas[k, 0] = clf.intercept_[0]
        betas[k, 1:] = clf.coef_[0].astype(np.float32, copy=False)

    return betas


def _fit_empirical_label_augmented_betas(
    X_train: csr_matrix,
    y_train: np.ndarray,
    X_synth_pool: csr_matrix,
    synth_pos_draws: np.ndarray,
    y_synthetic_draws: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
    random_state: int,
    max_iter: int = 2000,
    tol: float = 1e-3,
) -> np.ndarray:
    if float(alpha) <= 0:
        raise ValueError("_fit_empirical_label_augmented_betas requires alpha > 0.")

    n = X_train.shape[0]
    b = synth_pos_draws.shape[0]
    m = synth_pos_draws.shape[1]
    p = X_train.shape[1]

    if X_synth_pool.shape[1] != p:
        raise ValueError(
            "X_train and X_synth_pool must have the same number of columns. "
            f"Got {p} and {X_synth_pool.shape[1]}."
        )
    if y_synthetic_draws.shape != (b, m):
        raise ValueError(
            "y_synthetic_draws has the wrong shape. "
            f"Expected {(b, m)}, got {y_synthetic_draws.shape}."
        )

    dir_a = np.concatenate([
        np.ones(n, dtype=float),
        np.full(m, float(alpha) / float(m), dtype=float),
    ])
    scale = float(n + alpha)
    betas = np.zeros((b, p + 1), dtype=np.float32)

    clf = LogisticRegression(
        penalty=None,
        solver="saga",
        fit_intercept=True,
        warm_start=True,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        n_jobs=1,
    )

    try:
        _safe_fit_logistic(clf, X_train, y_train)
    except Exception:
        pass

    for k in range(b):
        pos = synth_pos_draws[k]
        X_synth_k = X_synth_pool[pos]
        y_synth_k = y_synthetic_draws[k].astype(int, copy=False)

        X_joint = vstack([X_train, X_synth_k], format="csr")
        y_joint = np.concatenate([y_train, y_synth_k])

        w = rng.dirichlet(dir_a) * scale
        _safe_fit_logistic(clf, X_joint, y_joint, sample_weight=w)

        betas[k, 0] = clf.intercept_[0]
        betas[k, 1:] = clf.coef_[0].astype(np.float32, copy=False)

    return betas


def _fit_l2_logistic_model(
    X_train: csr_matrix,
    y_train: np.ndarray,
    c_value: float,
    random_state: int,
    max_iter: int = 2000,
) -> LogisticRegression:
    clf = LogisticRegression(
        penalty="l2",
        C=float(c_value),
        solver="liblinear",
        fit_intercept=True,
        max_iter=max_iter,
        random_state=random_state,
    )
    _safe_fit_logistic(clf, X_train, y_train)
    return clf


def _ensemble_proba_from_betas(X: csr_matrix, beta_draws: np.ndarray) -> np.ndarray:
    intercepts = beta_draws[:, 0].astype(np.float64)
    coefs = beta_draws[:, 1:].astype(np.float64).T
    scores = X.dot(coefs) + intercepts
    proba = expit(scores)
    return np.asarray(proba.mean(axis=1)).ravel()


# ------------------
# Prefit candidate models
# ------------------
def prefit_train_side_candidates(
    train_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    y_col: str = "overnight_sign",
    use_numeric_features: bool = True,
    b: int = 100,
    m: int = 300,
    alpha_grid: Optional[List[float]] = None,
    l2_grid: Optional[List[float]] = None,
    random_state: int = 0,
    tfidf_grid: Optional[Dict[str, List[Any]]] = None,
    verbose: int = 1,
) -> List[Dict[str, Any]]:
    alpha_grid = _normalize_alpha_grid(alpha_grid)
    l2_grid = list(L2_GRID if l2_grid is None else l2_grid)
    tfidf_configs = _expand_tfidf_configs(tfidf_grid)

    train_m = prepare_baseline1_frame(train_df)
    y_train = train_m[y_col].astype(int).to_numpy()

    if b <= 0:
        raise ValueError(f"b must be positive. Got b={b}")

    has_positive_alpha = any(float(alpha) > 0 for alpha in alpha_grid)
    n_aug = len(aug_df)

    if has_positive_alpha and not (1 <= m < n_aug):
        raise ValueError(f"Need 1 <= m < len(aug_df) for positive alpha. Got m={m}, len(aug_df)={n_aug}")

    artifacts: List[Dict[str, Any]] = []

    for cfg_id, tfidf_params in enumerate(tfidf_configs):
        if verbose:
            print(
                f"Prefitting train-side candidates for TF-IDF config "
                f"{cfg_id + 1}/{len(tfidf_configs)}"
            )

        pre, _ = _build_preprocessor(
            tfidf_params,
            train_m,
            use_numeric_features=use_numeric_features,
        )
        X_train = _ensure_csr(pre.transform(train_m))

        X_aug_pool = None
        y_aug_pool = None
        aug_pos_draws = None

        if has_positive_alpha:
            aug_seed = _stable_seed_from_params(
                random_state,
                {"stage": "prefit_aug_draws", "tfidf_config_id": cfg_id, "b": int(b), "m": int(m), **tfidf_params},
            )
            rng_aug = np.random.default_rng(aug_seed)
            aug_draw_idxs = np.empty((b, m), dtype=np.int64)
            for k in range(b):
                aug_draw_idxs[k] = rng_aug.choice(n_aug, size=m, replace=False)

            uniq_idx = np.unique(aug_draw_idxs)
            aug_sub = prepare_baseline1_frame(aug_df.iloc[uniq_idx].copy())
            y_aug_pool = aug_sub[y_col].astype(int).to_numpy()
            X_aug_pool = _ensure_csr(pre.transform(aug_sub))
            aug_pos_draws = np.searchsorted(uniq_idx, aug_draw_idxs)

        beta_draws_by_alpha: Dict[float, np.ndarray] = {}
        for alpha in alpha_grid:
            alpha = float(alpha)
            alpha_seed = _stable_seed_from_params(
                random_state,
                {
                    "stage": "prefit_alpha_grid",
                    "tfidf_config_id": cfg_id,
                    "alpha": alpha,
                    "b": int(b),
                    "m": int(m),
                    **tfidf_params,
                },
            )
            rng_alpha = np.random.default_rng(alpha_seed)

            if alpha == 0.0:
                beta_draws = _fit_bayes_bootstrap_betas(
                    X_train=X_train,
                    y_train=y_train,
                    b=b,
                    rng=rng_alpha,
                    random_state=alpha_seed,
                )
            else:
                if X_aug_pool is None or y_aug_pool is None or aug_pos_draws is None:
                    raise RuntimeError("Positive alpha requested but augmented pool artifacts were not prepared.")
                beta_draws = _fit_augmented_betas(
                    X_train=X_train,
                    y_train=y_train,
                    X_aug_pool=X_aug_pool,
                    y_aug_pool=y_aug_pool,
                    aug_pos_draws=aug_pos_draws,
                    alpha=alpha,
                    rng=rng_alpha,
                    random_state=alpha_seed,
                )

            beta_draws_by_alpha[alpha] = beta_draws

        l2_models_by_c: Dict[float, LogisticRegression] = {}
        for c_value in l2_grid:
            c_value = float(c_value)
            c_seed = _stable_seed_from_params(
                random_state,
                {"stage": "prefit_l2", "tfidf_config_id": cfg_id, "c_value": c_value, **tfidf_params},
            )
            l2_models_by_c[c_value] = _fit_l2_logistic_model(
                X_train=X_train,
                y_train=y_train,
                c_value=c_value,
                random_state=c_seed,
            )

        artifacts.append(
            {
                "tfidf_config_id": int(cfg_id),
                "tfidf_params": dict(tfidf_params),
                "tfidf_params_json": _config_to_json(tfidf_params),
                "preprocessor": pre,
                "beta_draws_by_alpha": beta_draws_by_alpha,
                "l2_models_by_c": l2_models_by_c,
            }
        )

    return artifacts


def prefit_synthetic_only_candidates(
    aug_df: pd.DataFrame,
    y_col: str = "overnight_sign",
    use_numeric_features: bool = True,
    b: int = 100,
    random_state: int = 0,
    tfidf_grid: Optional[Dict[str, List[Any]]] = None,
    verbose: int = 1,
) -> List[Dict[str, Any]]:
    tfidf_configs = _expand_tfidf_configs(tfidf_grid)

    aug_m = prepare_baseline1_frame(aug_df)
    y_aug = aug_m[y_col].astype(int).to_numpy()

    artifacts: List[Dict[str, Any]] = []
    for cfg_id, tfidf_params in enumerate(tfidf_configs):
        if verbose:
            print(
                f"Prefitting synthetic-only alpha=0 candidates for TF-IDF config "
                f"{cfg_id + 1}/{len(tfidf_configs)}"
            )

        pre, _ = _build_preprocessor(
            tfidf_params,
            aug_m,
            use_numeric_features=use_numeric_features,
        )
        X_aug = _ensure_csr(pre.transform(aug_m))

        alpha0_seed = _stable_seed_from_params(
            random_state,
            {"stage": "prefit_alpha0_synthetic", "tfidf_config_id": cfg_id, "b": int(b), **tfidf_params},
        )
        rng_alpha0 = np.random.default_rng(alpha0_seed)
        beta_draws = _fit_bayes_bootstrap_betas(
            X_train=X_aug,
            y_train=y_aug,
            b=b,
            rng=rng_alpha0,
            random_state=alpha0_seed,
        )

        artifacts.append(
            {
                "tfidf_config_id": int(cfg_id),
                "tfidf_params": dict(tfidf_params),
                "tfidf_params_json": _config_to_json(tfidf_params),
                "preprocessor": pre,
                "beta_draws": beta_draws,
            }
        )

    return artifacts



def prefit_synthetic_label_candidates(
    train_df: pd.DataFrame,
    synthetic_label_matrix: np.ndarray,
    synthetic_label_cols: List[str],
    y_col: str = "overnight_sign",
    use_numeric_features: bool = True,
    b: int = 100,
    m: int = 300,
    alpha_grid: Optional[List[float]] = None,
    random_state: int = 0,
    tfidf_grid: Optional[Dict[str, List[Any]]] = None,
    verbose: int = 1,
) -> List[Dict[str, Any]]:
    alpha_grid = _normalize_alpha_grid(alpha_grid)
    tfidf_configs = _expand_tfidf_configs(tfidf_grid)

    train_m = prepare_baseline1_frame(train_df)
    y_train = train_m[y_col].astype(int).to_numpy()
    label_matrix = np.asarray(synthetic_label_matrix, dtype=float)

    if label_matrix.shape[0] != len(train_df):
        raise ValueError(
            "synthetic_label_matrix row count must equal len(train_df). "
            f"Expected {len(train_df)}, got {label_matrix.shape[0]}."
        )
    if b <= 0:
        raise ValueError(f"b must be positive. Got b={b}.")
    if m <= 0:
        raise ValueError(f"m must be positive. Got m={m}.")

    label_candidates, available_counts = _build_synthetic_label_candidate_lists(label_matrix)
    eligible_row_idx = np.flatnonzero(available_counts > 0)
    if len(eligible_row_idx) == 0:
        raise ValueError("No training rows have at least one available synthetic label.")

    artifacts: List[Dict[str, Any]] = []
    has_positive_alpha = any(float(alpha) > 0 for alpha in alpha_grid)

    for cfg_id, tfidf_params in enumerate(tfidf_configs):
        if verbose:
            print(
                f"Prefitting synthetic-label augmentation candidates for TF-IDF config "
                f"{cfg_id + 1}/{len(tfidf_configs)}"
            )

        pre, _ = _build_preprocessor(
            tfidf_params,
            train_m,
            use_numeric_features=use_numeric_features,
        )
        X_train = _ensure_csr(pre.transform(train_m))

        X_synth_pool = None
        synth_pos_draws = None
        y_synthetic_sampled = None

        if has_positive_alpha:
            row_seed = _stable_seed_from_params(
                random_state,
                {
                    "stage": "prefit_synthetic_label_empirical_rows",
                    "tfidf_config_id": cfg_id,
                    "b": int(b),
                    "m": int(m),
                    **tfidf_params,
                },
            )
            label_seed = _stable_seed_from_params(
                random_state,
                {
                    "stage": "prefit_synthetic_label_row_labels",
                    "tfidf_config_id": cfg_id,
                    "b": int(b),
                    "m": int(m),
                    **tfidf_params,
                },
            )
            rng_rows = np.random.default_rng(row_seed)
            rng_labels = np.random.default_rng(label_seed)
            sampled_row_draws, y_synthetic_sampled = _sample_empirical_synthetic_label_draws(
                label_candidates=label_candidates,
                eligible_row_idx=eligible_row_idx,
                b=b,
                m=m,
                rng_rows=rng_rows,
                rng_labels=rng_labels,
            )
            uniq_idx = np.unique(sampled_row_draws)
            X_synth_pool = X_train[uniq_idx]
            synth_pos_draws = np.searchsorted(uniq_idx, sampled_row_draws)

        beta_draws_by_alpha: Dict[float, np.ndarray] = {}
        for alpha in alpha_grid:
            alpha = float(alpha)
            alpha_seed = _stable_seed_from_params(
                random_state,
                {
                    "stage": "prefit_synthetic_label_alpha_grid",
                    "tfidf_config_id": cfg_id,
                    "alpha": alpha,
                    "b": int(b),
                    "m": int(m),
                    **tfidf_params,
                },
            )
            rng_alpha = np.random.default_rng(alpha_seed)

            if alpha == 0.0:
                beta_draws = _fit_bayes_bootstrap_betas(
                    X_train=X_train,
                    y_train=y_train,
                    b=b,
                    rng=rng_alpha,
                    random_state=alpha_seed,
                )
            else:
                if X_synth_pool is None or synth_pos_draws is None or y_synthetic_sampled is None:
                    raise RuntimeError("Positive alpha requested but synthetic-label sampling artifacts were not prepared.")
                beta_draws = _fit_empirical_label_augmented_betas(
                    X_train=X_train,
                    y_train=y_train,
                    X_synth_pool=X_synth_pool,
                    synth_pos_draws=synth_pos_draws,
                    y_synthetic_draws=y_synthetic_sampled,
                    alpha=alpha,
                    rng=rng_alpha,
                    random_state=alpha_seed,
                )

            beta_draws_by_alpha[alpha] = beta_draws

        artifacts.append(
            {
                "tfidf_config_id": int(cfg_id),
                "tfidf_params": dict(tfidf_params),
                "tfidf_params_json": _config_to_json(tfidf_params),
                "preprocessor": pre,
                "beta_draws_by_alpha": beta_draws_by_alpha,
                "synthetic_label_cols": list(synthetic_label_cols),
                "n_synthetic_label_draws": int(len(synthetic_label_cols)),
                "n_synthetic_label_rows_eligible": int(len(eligible_row_idx)),
                "b_bootstrap_draws": int(b),
                "m_synthetic_sample_size": int(m),
            }
        )

    return artifacts


# ------------------
# Refit helpers
# ------------------
def refit_alpha0_real_model_on_trainval(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tfidf_params: Dict[str, Any],
    y_col: str = "overnight_sign",
    use_numeric_features: bool = True,
    b: int = 100,
    random_state: int = 0,
    sim_id: int = 0,
) -> Dict[str, Any]:
    trainval_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    trainval_m = prepare_baseline1_frame(trainval_df)
    y_trainval = trainval_m[y_col].astype(int).to_numpy()

    pre, _ = _build_preprocessor(
        tfidf_params,
        trainval_m,
        use_numeric_features=use_numeric_features,
    )
    X_trainval = _ensure_csr(pre.transform(trainval_m))

    fit_seed = _stable_seed_from_params(
        random_state,
        {"stage": "final_refit_alpha0_real", "sim_id": int(sim_id), "b": int(b), **tfidf_params},
    )
    rng_fit = np.random.default_rng(fit_seed)
    beta_draws = _fit_bayes_bootstrap_betas(
        X_train=X_trainval,
        y_train=y_trainval,
        b=b,
        rng=rng_fit,
        random_state=fit_seed,
    )

    return {
        "preprocessor": pre,
        "beta_draws": beta_draws,
        "tfidf_params": dict(tfidf_params),
        "tfidf_params_json": _config_to_json(tfidf_params),
        "selected_alpha": 0.0,
        "refit_alpha": 0.0,
    }


def refit_alpha_grid_model_on_trainval(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    tfidf_params: Dict[str, Any],
    selected_alpha: float,
    y_col: str = "overnight_sign",
    use_numeric_features: bool = True,
    b: int = 100,
    m: int = 300,
    random_state: int = 0,
    sim_id: int = 0,
    rescale_alpha_on_refit: bool = True,
) -> Dict[str, Any]:
    selected_alpha = float(selected_alpha)

    if selected_alpha == 0.0:
        return refit_alpha0_real_model_on_trainval(
            train_df=train_df,
            val_df=val_df,
            tfidf_params=tfidf_params,
            y_col=y_col,
            use_numeric_features=use_numeric_features,
            b=b,
            random_state=random_state,
            sim_id=sim_id,
        )

    trainval_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    trainval_m = prepare_baseline1_frame(trainval_df)
    y_trainval = trainval_m[y_col].astype(int).to_numpy()

    pre, _ = _build_preprocessor(
        tfidf_params,
        trainval_m,
        use_numeric_features=use_numeric_features,
    )
    X_trainval = _ensure_csr(pre.transform(trainval_m))

    refit_alpha = float(selected_alpha)
    if rescale_alpha_on_refit:
        refit_alpha = float(selected_alpha) / float(len(train_df)) * float(len(trainval_df))

    aug_seed = _stable_seed_from_params(
        random_state,
        {
            "stage": "final_refit_aug_draws",
            "sim_id": int(sim_id),
            "selected_alpha": float(selected_alpha),
            "refit_alpha": float(refit_alpha),
            "b": int(b),
            "m": int(m),
            **tfidf_params,
        },
    )
    dirichlet_seed = _stable_seed_from_params(
        random_state,
        {
            "stage": "final_refit_aug_dirichlet",
            "sim_id": int(sim_id),
            "selected_alpha": float(selected_alpha),
            "refit_alpha": float(refit_alpha),
            "b": int(b),
            "m": int(m),
            **tfidf_params,
        },
    )

    rng_aug = np.random.default_rng(aug_seed)
    rng_dirichlet = np.random.default_rng(dirichlet_seed)
    n_aug = len(aug_df)

    if not (1 <= m < n_aug):
        raise ValueError(f"Need 1 <= m < len(aug_df) for positive alpha refit. Got m={m}, len(aug_df)={n_aug}")

    aug_draw_idxs = np.empty((b, m), dtype=np.int64)
    for k in range(b):
        aug_draw_idxs[k] = rng_aug.choice(n_aug, size=m, replace=False)

    uniq_idx = np.unique(aug_draw_idxs)
    aug_sub = prepare_baseline1_frame(aug_df.iloc[uniq_idx].copy())
    y_aug_pool = aug_sub[y_col].astype(int).to_numpy()
    X_aug_pool = _ensure_csr(pre.transform(aug_sub))
    aug_pos_draws = np.searchsorted(uniq_idx, aug_draw_idxs)

    beta_draws = _fit_augmented_betas(
        X_train=X_trainval,
        y_train=y_trainval,
        X_aug_pool=X_aug_pool,
        y_aug_pool=y_aug_pool,
        aug_pos_draws=aug_pos_draws,
        alpha=refit_alpha,
        rng=rng_dirichlet,
        random_state=dirichlet_seed,
    )

    return {
        "preprocessor": pre,
        "beta_draws": beta_draws,
        "selected_alpha": float(selected_alpha),
        "refit_alpha": float(refit_alpha),
        "tfidf_params": dict(tfidf_params),
        "tfidf_params_json": _config_to_json(tfidf_params),
    }


def refit_synthetic_label_alpha_grid_model_on_trainval(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    synthetic_label_matrix: np.ndarray,
    synthetic_label_cols: List[str],
    tfidf_params: Dict[str, Any],
    selected_alpha: float,
    y_col: str = "overnight_sign",
    use_numeric_features: bool = True,
    b: int = 100,
    m: int = 300,
    random_state: int = 0,
    sim_id: int = 0,
    rescale_alpha_on_refit: bool = True,
) -> Dict[str, Any]:
    selected_alpha = float(selected_alpha)
    label_matrix = np.asarray(synthetic_label_matrix, dtype=float)
    if label_matrix.shape[0] != len(train_df):
        raise ValueError(
            "synthetic_label_matrix row count must equal len(train_df). "
            f"Expected {len(train_df)}, got {label_matrix.shape[0]}."
        )
    if b <= 0:
        raise ValueError(f"b must be positive. Got b={b}.")
    if m <= 0:
        raise ValueError(f"m must be positive. Got m={m}.")

    label_candidates, available_counts = _build_synthetic_label_candidate_lists(label_matrix)
    eligible_row_idx = np.flatnonzero(available_counts > 0)
    if len(eligible_row_idx) == 0:
        raise ValueError("No training rows have at least one available synthetic label.")

    train_m = prepare_baseline1_frame(train_df)
    trainval_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    trainval_m = prepare_baseline1_frame(trainval_df)
    y_trainval = trainval_m[y_col].astype(int).to_numpy()

    pre, _ = _build_preprocessor(
        tfidf_params,
        trainval_m,
        use_numeric_features=use_numeric_features,
    )
    X_trainval = _ensure_csr(pre.transform(trainval_m))
    X_train_only = _ensure_csr(pre.transform(train_m))

    if selected_alpha == 0.0:
        fit_seed = _stable_seed_from_params(
            random_state,
            {"stage": "final_refit_synthetic_label_alpha0", "sim_id": int(sim_id), "b": int(b), **tfidf_params},
        )
        rng_fit = np.random.default_rng(fit_seed)
        beta_draws = _fit_bayes_bootstrap_betas(
            X_train=X_trainval,
            y_train=y_trainval,
            b=b,
            rng=rng_fit,
            random_state=fit_seed,
        )
        return {
            "preprocessor": pre,
            "beta_draws": beta_draws,
            "selected_alpha": 0.0,
            "refit_alpha": 0.0,
            "tfidf_params": dict(tfidf_params),
            "tfidf_params_json": _config_to_json(tfidf_params),
            "synthetic_label_cols": list(synthetic_label_cols),
            "n_synthetic_label_draws": int(len(synthetic_label_cols)),
            "n_synthetic_label_rows_eligible": int(len(eligible_row_idx)),
            "b_bootstrap_draws": int(b),
            "m_synthetic_sample_size": int(m),
        }

    refit_alpha = float(selected_alpha)
    if rescale_alpha_on_refit:
        refit_alpha = float(selected_alpha) / float(len(train_df)) * float(len(trainval_df))

    row_seed = _stable_seed_from_params(
        random_state,
        {
            "stage": "final_refit_synthetic_label_empirical_rows",
            "sim_id": int(sim_id),
            "selected_alpha": float(selected_alpha),
            "refit_alpha": float(refit_alpha),
            "b": int(b),
            "m": int(m),
            **tfidf_params,
        },
    )
    label_seed = _stable_seed_from_params(
        random_state,
        {
            "stage": "final_refit_synthetic_label_row_labels",
            "sim_id": int(sim_id),
            "selected_alpha": float(selected_alpha),
            "refit_alpha": float(refit_alpha),
            "b": int(b),
            "m": int(m),
            **tfidf_params,
        },
    )
    fit_seed = _stable_seed_from_params(
        random_state,
        {
            "stage": "final_refit_synthetic_label_aug",
            "sim_id": int(sim_id),
            "selected_alpha": float(selected_alpha),
            "refit_alpha": float(refit_alpha),
            "b": int(b),
            "m": int(m),
            **tfidf_params,
        },
    )

    rng_rows = np.random.default_rng(row_seed)
    rng_labels = np.random.default_rng(label_seed)
    rng_fit = np.random.default_rng(fit_seed)

    sampled_row_draws, y_synthetic_sampled = _sample_empirical_synthetic_label_draws(
        label_candidates=label_candidates,
        eligible_row_idx=eligible_row_idx,
        b=b,
        m=m,
        rng_rows=rng_rows,
        rng_labels=rng_labels,
    )
    uniq_idx = np.unique(sampled_row_draws)
    X_synth_pool = X_train_only[uniq_idx]
    synth_pos_draws = np.searchsorted(uniq_idx, sampled_row_draws)

    beta_draws = _fit_empirical_label_augmented_betas(
        X_train=X_trainval,
        y_train=y_trainval,
        X_synth_pool=X_synth_pool,
        synth_pos_draws=synth_pos_draws,
        y_synthetic_draws=y_synthetic_sampled,
        alpha=refit_alpha,
        rng=rng_fit,
        random_state=fit_seed,
    )

    return {
        "preprocessor": pre,
        "beta_draws": beta_draws,
        "selected_alpha": float(selected_alpha),
        "refit_alpha": float(refit_alpha),
        "tfidf_params": dict(tfidf_params),
        "tfidf_params_json": _config_to_json(tfidf_params),
        "synthetic_label_cols": list(synthetic_label_cols),
        "n_synthetic_label_draws": int(len(synthetic_label_cols)),
        "n_synthetic_label_rows_eligible": int(len(eligible_row_idx)),
        "b_bootstrap_draws": int(b),
        "m_synthetic_sample_size": int(m),
    }


def refit_l2_model_on_trainval(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tfidf_params: Dict[str, Any],
    c_value: float,
    y_col: str = "overnight_sign",
    use_numeric_features: bool = True,
    random_state: int = 0,
    sim_id: int = 0,
) -> Dict[str, Any]:
    trainval_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    trainval_m = prepare_baseline1_frame(trainval_df)
    y_trainval = trainval_m[y_col].astype(int).to_numpy()

    pre, _ = _build_preprocessor(
        tfidf_params,
        trainval_m,
        use_numeric_features=use_numeric_features,
    )
    X_trainval = _ensure_csr(pre.transform(trainval_m))

    fit_seed = _stable_seed_from_params(
        random_state,
        {"stage": "final_refit_l2", "sim_id": int(sim_id), "c_value": float(c_value), **tfidf_params},
    )
    clf = _fit_l2_logistic_model(
        X_train=X_trainval,
        y_train=y_trainval,
        c_value=float(c_value),
        random_state=fit_seed,
    )

    return {
        "preprocessor": pre,
        "model": clf,
        "c_value": float(c_value),
        "tfidf_params": dict(tfidf_params),
        "tfidf_params_json": _config_to_json(tfidf_params),
    }


# ------------------
# Prediction / evaluation helpers
# ------------------
def evaluate_bootstrap_model(
    model: Dict[str, Any],
    df: pd.DataFrame,
    y_col: str = "overnight_sign",
    threshold: float = 0.5,
) -> Dict[str, float]:
    df_m = prepare_baseline1_frame(df)
    X = _ensure_csr(model["preprocessor"].transform(df_m))
    proba = _ensemble_proba_from_betas(X, model["beta_draws"])
    return _metrics_from_proba(df[y_col].astype(int).to_numpy(), proba, threshold=threshold)


def evaluate_l2_model(
    model: Dict[str, Any],
    df: pd.DataFrame,
    y_col: str = "overnight_sign",
    threshold: float = 0.5,
) -> Dict[str, float]:
    df_m = prepare_baseline1_frame(df)
    X = _ensure_csr(model["preprocessor"].transform(df_m))
    proba = model["model"].predict_proba(X)[:, 1]
    return _metrics_from_proba(df[y_col].astype(int).to_numpy(), proba, threshold=threshold)


# ------------------
# Monte Carlo benchmark
# ------------------
def run_monte_carlo_benchmark(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    synthetic_label_df: pd.DataFrame,
    y_col: str = "overnight_sign",
    id_col: str = "rp_entity_id",
    use_numeric_features: bool = True,
    b: int = 100,
    m: int = 300,
    n_sims: int = 100,
    alpha_grid: Optional[List[float]] = None,
    l2_grid: Optional[List[float]] = None,
    tfidf_grid: Optional[Dict[str, List[Any]]] = None,
    verbose: int = 1,
    random_state: int = 0,
    rescale_alpha_on_refit: bool = True,
) -> Dict[str, Any]:
    alpha_grid = _normalize_alpha_grid(alpha_grid)
    l2_grid = list(L2_GRID if l2_grid is None else l2_grid)
    tfidf_configs = _expand_tfidf_configs(tfidf_grid)

    aug_df = _maybe_normalize_aug_df(aug_df)
    synthetic_label_cols, synthetic_label_matrix = _extract_synthetic_label_matrix(
        train_df=train_df,
        synthetic_label_df=synthetic_label_df,
        id_col=id_col,
    )

    eval_df = pd.concat([val_df, test_df], axis=0, ignore_index=True)
    eval_idx = np.arange(len(eval_df))
    n_half = len(eval_df) // 2

    if verbose:
        print(
            f"Prefit phase: {len(tfidf_configs)} TF-IDF configs, "
            f"{len(alpha_grid)} alpha values (including 0), {len(l2_grid)} L2 C values, "
            f"and {len(synthetic_label_cols)} synthetic-label columns"
        )

    train_artifacts = prefit_train_side_candidates(
        train_df=train_df,
        aug_df=aug_df,
        y_col=y_col,
        use_numeric_features=use_numeric_features,
        b=b,
        m=m,
        alpha_grid=alpha_grid,
        l2_grid=l2_grid,
        random_state=random_state,
        tfidf_grid=tfidf_grid,
        verbose=verbose,
    )
    synthetic_artifacts = prefit_synthetic_only_candidates(
        aug_df=aug_df,
        y_col=y_col,
        use_numeric_features=use_numeric_features,
        b=b,
        random_state=random_state,
        tfidf_grid=tfidf_grid,
        verbose=verbose,
    )
    synthetic_label_artifacts = prefit_synthetic_label_candidates(
        train_df=train_df,
        synthetic_label_matrix=synthetic_label_matrix,
        synthetic_label_cols=synthetic_label_cols,
        y_col=y_col,
        use_numeric_features=use_numeric_features,
        b=b,
        m=m,
        alpha_grid=alpha_grid,
        random_state=random_state,
        tfidf_grid=tfidf_grid,
        verbose=verbose,
    )

    val_aug_rows: List[Dict[str, Any]] = []
    val_l2_rows: List[Dict[str, Any]] = []
    val_alpha0_synth_rows: List[Dict[str, Any]] = []
    val_synthetic_label_aug_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []

    for sim_id in range(n_sims):
        if verbose:
            print(f"Monte Carlo simulation {sim_id + 1}/{n_sims}")

        split_seed = _stable_seed_from_params(
            random_state,
            {"stage": "monte_carlo_split", "sim_id": int(sim_id)},
        )
        rng_split = np.random.default_rng(split_seed)
        perm = rng_split.permutation(eval_idx)
        val_idx = perm[:n_half]
        test_idx = perm[n_half:]
        val_df_sim = eval_df.iloc[val_idx].copy()
        test_df_sim = eval_df.iloc[test_idx].copy()

        split_rows.append(
            {
                "sim_id": int(sim_id),
                "split_seed": int(split_seed),
                "val_n": int(len(val_df_sim)),
                "test_n": int(len(test_df_sim)),
                "val_pos_rate": float(val_df_sim[y_col].astype(int).mean()),
                "test_pos_rate": float(test_df_sim[y_col].astype(int).mean()),
            }
        )

        val_m = prepare_baseline1_frame(val_df_sim)
        y_val = val_df_sim[y_col].astype(int).to_numpy()

        best_aug: Optional[Dict[str, Any]] = None
        best_l2: Optional[Dict[str, Any]] = None
        best_alpha0_synth: Optional[Dict[str, Any]] = None
        best_synthetic_label_aug: Optional[Dict[str, Any]] = None

        for art in train_artifacts:
            X_val = _ensure_csr(art["preprocessor"].transform(val_m))
            tfidf_meta = {
                "tfidf_config_id": art["tfidf_config_id"],
                "tfidf_params_json": art["tfidf_params_json"],
                **_flatten_tfidf_params(art["tfidf_params"]),
            }

            # synthetic-news alpha-grid benchmark, including alpha = 0
            for alpha, beta_draws in art["beta_draws_by_alpha"].items():
                proba = _ensemble_proba_from_betas(X_val, beta_draws)
                metrics = _metrics_from_proba(y_val, proba)
                row = {
                    "sim_id": int(sim_id),
                    "method": "augmented_alpha_grid",
                    "alpha": float(alpha),
                    **tfidf_meta,
                    **{f"val_{k}": v for k, v in metrics.items()},
                }
                val_aug_rows.append(row)
                if (best_aug is None) or (
                    _selection_score(metrics["auc"]) > _selection_score(best_aug["selected_val_auc"])
                ):
                    best_aug = {
                        "tfidf_config_id": art["tfidf_config_id"],
                        "tfidf_params": art["tfidf_params"],
                        "tfidf_params_json": art["tfidf_params_json"],
                        "selected_alpha": float(alpha),
                        "selected_val_auc": metrics["auc"],
                        "selected_val_log_loss": metrics["log_loss"],
                        "selected_val_accuracy": metrics["accuracy"],
                        "selected_val_balanced_accuracy": metrics["balanced_accuracy"],
                    }

            # no-augmentation L2 logistic baseline
            for c_value, clf in art["l2_models_by_c"].items():
                proba = clf.predict_proba(X_val)[:, 1]
                metrics = _metrics_from_proba(y_val, proba)
                row = {
                    "sim_id": int(sim_id),
                    "method": "l2_logistic",
                    "c_value": float(c_value),
                    **tfidf_meta,
                    **{f"val_{k}": v for k, v in metrics.items()},
                }
                val_l2_rows.append(row)
                if (best_l2 is None) or (
                    _selection_score(metrics["auc"]) > _selection_score(best_l2["selected_val_auc"])
                ):
                    best_l2 = {
                        "tfidf_config_id": art["tfidf_config_id"],
                        "tfidf_params": art["tfidf_params"],
                        "tfidf_params_json": art["tfidf_params_json"],
                        "selected_c_value": float(c_value),
                        "selected_val_auc": metrics["auc"],
                        "selected_val_log_loss": metrics["log_loss"],
                        "selected_val_accuracy": metrics["accuracy"],
                        "selected_val_balanced_accuracy": metrics["balanced_accuracy"],
                    }

        for art in synthetic_artifacts:
            X_val = _ensure_csr(art["preprocessor"].transform(val_m))
            tfidf_meta = {
                "tfidf_config_id": art["tfidf_config_id"],
                "tfidf_params_json": art["tfidf_params_json"],
                **_flatten_tfidf_params(art["tfidf_params"]),
            }
            proba = _ensemble_proba_from_betas(X_val, art["beta_draws"])
            metrics = _metrics_from_proba(y_val, proba)
            row = {
                "sim_id": int(sim_id),
                "method": "alpha0_synthetic_only",
                **tfidf_meta,
                **{f"val_{k}": v for k, v in metrics.items()},
            }
            val_alpha0_synth_rows.append(row)
            if (best_alpha0_synth is None) or (
                _selection_score(metrics["auc"]) > _selection_score(best_alpha0_synth["selected_val_auc"])
            ):
                best_alpha0_synth = {
                    "tfidf_config_id": art["tfidf_config_id"],
                    "tfidf_params": art["tfidf_params"],
                    "tfidf_params_json": art["tfidf_params_json"],
                    "selected_val_auc": metrics["auc"],
                    "selected_val_log_loss": metrics["log_loss"],
                    "selected_val_accuracy": metrics["accuracy"],
                    "selected_val_balanced_accuracy": metrics["balanced_accuracy"],
                    "prefit_model": art,
                }

        for art in synthetic_label_artifacts:
            X_val = _ensure_csr(art["preprocessor"].transform(val_m))
            tfidf_meta = {
                "tfidf_config_id": art["tfidf_config_id"],
                "tfidf_params_json": art["tfidf_params_json"],
                **_flatten_tfidf_params(art["tfidf_params"]),
            }

            for alpha, beta_draws in art["beta_draws_by_alpha"].items():
                proba = _ensemble_proba_from_betas(X_val, beta_draws)
                metrics = _metrics_from_proba(y_val, proba)
                row = {
                    "sim_id": int(sim_id),
                    "method": "synthetic_label_alpha_grid",
                    "alpha": float(alpha),
                    **tfidf_meta,
                    **{f"val_{k}": v for k, v in metrics.items()},
                }
                val_synthetic_label_aug_rows.append(row)
                if (best_synthetic_label_aug is None) or (
                    _selection_score(metrics["auc"]) > _selection_score(best_synthetic_label_aug["selected_val_auc"])
                ):
                    best_synthetic_label_aug = {
                        "tfidf_config_id": art["tfidf_config_id"],
                        "tfidf_params": art["tfidf_params"],
                        "tfidf_params_json": art["tfidf_params_json"],
                        "selected_alpha": float(alpha),
                        "selected_val_auc": metrics["auc"],
                        "selected_val_log_loss": metrics["log_loss"],
                        "selected_val_accuracy": metrics["accuracy"],
                        "selected_val_balanced_accuracy": metrics["balanced_accuracy"],
                        "n_synthetic_label_draws": int(art["n_synthetic_label_draws"]),
                        "n_synthetic_label_rows_eligible": int(art["n_synthetic_label_rows_eligible"]),
                    }

        if best_aug is None or best_l2 is None or best_alpha0_synth is None or best_synthetic_label_aug is None:
            raise RuntimeError("At least one method failed to produce a validation candidate.")

        # Final synthetic-news alpha-grid model on train + validation split; alpha=0 is handled inside
        final_aug_model = refit_alpha_grid_model_on_trainval(
            train_df=train_df,
            val_df=val_df_sim,
            aug_df=aug_df,
            tfidf_params=best_aug["tfidf_params"],
            selected_alpha=best_aug["selected_alpha"],
            y_col=y_col,
            use_numeric_features=use_numeric_features,
            b=b,
            m=m,
            random_state=random_state,
            sim_id=sim_id,
            rescale_alpha_on_refit=rescale_alpha_on_refit,
        )
        aug_test_metrics = evaluate_bootstrap_model(final_aug_model, test_df_sim, y_col=y_col)
        test_rows.append(
            {
                "sim_id": int(sim_id),
                "method": "augmented_alpha_grid",
                "selected_tfidf_config_id": int(best_aug["tfidf_config_id"]),
                "selected_tfidf_params_json": best_aug["tfidf_params_json"],
                **_flatten_tfidf_params(best_aug["tfidf_params"], prefix="selected_"),
                "selected_alpha": float(best_aug["selected_alpha"]),
                "refit_alpha": float(final_aug_model["refit_alpha"]),
                "selected_val_auc": float(best_aug["selected_val_auc"]),
                "selected_val_log_loss": float(best_aug["selected_val_log_loss"]),
                "selected_val_accuracy": float(best_aug["selected_val_accuracy"]),
                "selected_val_balanced_accuracy": float(best_aug["selected_val_balanced_accuracy"]),
                **{f"test_{k}": v for k, v in aug_test_metrics.items()},
            }
        )

        # Final synthetic-label alpha-grid model on train + validation split; synthetic labels are available for train rows only
        final_synthetic_label_model = refit_synthetic_label_alpha_grid_model_on_trainval(
            train_df=train_df,
            val_df=val_df_sim,
            synthetic_label_matrix=synthetic_label_matrix,
            synthetic_label_cols=synthetic_label_cols,
            tfidf_params=best_synthetic_label_aug["tfidf_params"],
            selected_alpha=best_synthetic_label_aug["selected_alpha"],
            y_col=y_col,
            use_numeric_features=use_numeric_features,
            b=b,
            m=m,
            random_state=random_state,
            sim_id=sim_id,
            rescale_alpha_on_refit=rescale_alpha_on_refit,
        )
        synthetic_label_test_metrics = evaluate_bootstrap_model(
            final_synthetic_label_model,
            test_df_sim,
            y_col=y_col,
        )
        test_rows.append(
            {
                "sim_id": int(sim_id),
                "method": "synthetic_label_alpha_grid",
                "selected_tfidf_config_id": int(best_synthetic_label_aug["tfidf_config_id"]),
                "selected_tfidf_params_json": best_synthetic_label_aug["tfidf_params_json"],
                **_flatten_tfidf_params(best_synthetic_label_aug["tfidf_params"], prefix="selected_"),
                "selected_alpha": float(best_synthetic_label_aug["selected_alpha"]),
                "refit_alpha": float(final_synthetic_label_model["refit_alpha"]),
                "n_synthetic_label_draws": int(final_synthetic_label_model["n_synthetic_label_draws"]),
                "n_synthetic_label_rows_eligible": int(final_synthetic_label_model["n_synthetic_label_rows_eligible"]),
                "b_bootstrap_draws": int(final_synthetic_label_model["b_bootstrap_draws"]),
                "m_synthetic_sample_size": int(final_synthetic_label_model["m_synthetic_sample_size"]),
                "selected_val_auc": float(best_synthetic_label_aug["selected_val_auc"]),
                "selected_val_log_loss": float(best_synthetic_label_aug["selected_val_log_loss"]),
                "selected_val_accuracy": float(best_synthetic_label_aug["selected_val_accuracy"]),
                "selected_val_balanced_accuracy": float(best_synthetic_label_aug["selected_val_balanced_accuracy"]),
                **{f"test_{k}": v for k, v in synthetic_label_test_metrics.items()},
            }
        )

        # Final L2 model on train + validation split
        final_l2_model = refit_l2_model_on_trainval(
            train_df=train_df,
            val_df=val_df_sim,
            tfidf_params=best_l2["tfidf_params"],
            c_value=best_l2["selected_c_value"],
            y_col=y_col,
            use_numeric_features=use_numeric_features,
            random_state=random_state,
            sim_id=sim_id,
        )
        l2_test_metrics = evaluate_l2_model(final_l2_model, test_df_sim, y_col=y_col)
        test_rows.append(
            {
                "sim_id": int(sim_id),
                "method": "l2_logistic",
                "selected_tfidf_config_id": int(best_l2["tfidf_config_id"]),
                "selected_tfidf_params_json": best_l2["tfidf_params_json"],
                **_flatten_tfidf_params(best_l2["tfidf_params"], prefix="selected_"),
                "selected_c_value": float(best_l2["selected_c_value"]),
                "selected_val_auc": float(best_l2["selected_val_auc"]),
                "selected_val_log_loss": float(best_l2["selected_val_log_loss"]),
                "selected_val_accuracy": float(best_l2["selected_val_accuracy"]),
                "selected_val_balanced_accuracy": float(best_l2["selected_val_balanced_accuracy"]),
                **{f"test_{k}": v for k, v in l2_test_metrics.items()},
            }
        )

        # alpha=0 synthetic-only model: no refit, just evaluate the selected prefit model on test
        alpha0_synth_test_metrics = evaluate_bootstrap_model(
            best_alpha0_synth["prefit_model"],
            test_df_sim,
            y_col=y_col,
        )
        test_rows.append(
            {
                "sim_id": int(sim_id),
                "method": "alpha0_synthetic_only",
                "selected_tfidf_config_id": int(best_alpha0_synth["tfidf_config_id"]),
                "selected_tfidf_params_json": best_alpha0_synth["tfidf_params_json"],
                **_flatten_tfidf_params(best_alpha0_synth["tfidf_params"], prefix="selected_"),
                "selected_val_auc": float(best_alpha0_synth["selected_val_auc"]),
                "selected_val_log_loss": float(best_alpha0_synth["selected_val_log_loss"]),
                "selected_val_accuracy": float(best_alpha0_synth["selected_val_accuracy"]),
                "selected_val_balanced_accuracy": float(best_alpha0_synth["selected_val_balanced_accuracy"]),
                **{f"test_{k}": v for k, v in alpha0_synth_test_metrics.items()},
            }
        )

    validation_aug_df = pd.DataFrame(val_aug_rows)
    validation_l2_df = pd.DataFrame(val_l2_rows)
    validation_alpha0_synth_df = pd.DataFrame(val_alpha0_synth_rows)
    validation_synthetic_label_aug_df = pd.DataFrame(val_synthetic_label_aug_rows)
    test_results_df = pd.DataFrame(test_rows)
    split_df = pd.DataFrame(split_rows)

    summary_by_method = (
        test_results_df.groupby("method", dropna=False)
        .agg(
            n_sims=("sim_id", "nunique"),
            mean_selected_val_auc=("selected_val_auc", "mean"),
            sd_selected_val_auc=("selected_val_auc", "std"),
            mean_test_auc=("test_auc", "mean"),
            sd_test_auc=("test_auc", "std"),
            median_test_auc=("test_auc", "median"),
            min_test_auc=("test_auc", "min"),
            max_test_auc=("test_auc", "max"),
        )
        .reset_index()
        .sort_values("mean_test_auc", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "validation_results_aug": validation_aug_df,
        "validation_results_l2": validation_l2_df,
        "validation_results_alpha0_synthetic": validation_alpha0_synth_df,
        "validation_results_synthetic_label_aug": validation_synthetic_label_aug_df,
        "test_results": test_results_df,
        "split_results": split_df,
        "summary_by_method": summary_by_method,
        "config": {
            "b": int(b),
            "m": int(m),
            "n_sims": int(n_sims),
            "alpha_grid": list(alpha_grid),
            "l2_grid": list(l2_grid),
            "tfidf_grid": copy.deepcopy(DEFAULT_TFIDF_GRID if tfidf_grid is None else tfidf_grid),
            "n_tfidf_configs": int(len(tfidf_configs)),
            "n_synthetic_label_draws": int(len(synthetic_label_cols)),
            "synthetic_label_cols": list(synthetic_label_cols),
            "n_synthetic_label_rows_eligible": int(np.sum(~np.isnan(synthetic_label_matrix).all(axis=1))),
            "use_numeric_features": bool(use_numeric_features),
            "random_state": int(random_state),
            "rescale_alpha_on_refit": bool(rescale_alpha_on_refit),
        },
    }


# ------------------
# Saving helpers
# ------------------
def save_benchmark_outputs(
    results: Dict[str, Any],
    output_dir: pathlib.Path,
    save_prefix: str,
) -> Dict[str, pathlib.Path]:
    """Save *all* benchmark outputs into a single pickle file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = output_dir.joinpath(f"{save_prefix}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return {"pickle": pkl_path}


# ------------------
# Main
# ------------------
if __name__ == "__main__":
    arguments = parse_args()
    output_dir = pathlib.Path(arguments["model_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    random_state = int(arguments["random_state"])
    m = int(arguments["m"])
    b_draws = int(arguments["b"])
    n_sims = int(arguments["n_sims"])
    verbose = int(arguments["verbose"])
    y_col = arguments["y_col"]
    id_col = arguments["id_col"]
    save_prefix = str(arguments["save_prefix"])
    rescale_alpha_on_refit = bool(arguments["rescale_alpha_on_refit"])

    train_df = _load_firm_date_csv("firm_date_train_r50.csv")
    val_df = _load_firm_date_csv("firm_date_val_r50.csv")
    test_df = _load_firm_date_csv("firm_date_test_r50.csv")
    here = pathlib.Path(__file__).resolve().parent
    aug_df = _load_firm_date_csv(here / "news_generation_code" / "out_test_600" / "synthetic_firmdate.csv")
    aug_df = _maybe_normalize_aug_df(aug_df)
    synthetic_label_csv = here / "news_generation_code" / "synthetic_label_out" / "synthetic_label_wide.csv"
    synthetic_label_df = _load_firm_date_csv(synthetic_label_csv)

    results = run_monte_carlo_benchmark(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        aug_df=aug_df,
        synthetic_label_df=synthetic_label_df,
        y_col=y_col,
        id_col=id_col,
        use_numeric_features=True,
        b=b_draws,
        m=m,
        n_sims=n_sims,
        tfidf_grid=None,
        verbose=verbose,
        random_state=random_state,
        rescale_alpha_on_refit=rescale_alpha_on_refit,
    )

    paths = save_benchmark_outputs(
        results=results,
        output_dir=output_dir,
        save_prefix=save_prefix,
    )

    print("\nSummary by method:")
    print(results["summary_by_method"])
