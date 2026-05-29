#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1: Calibrate a scalar Engel-curve Monte Carlo design from public Engel95 data,
        generate Monte Carlo train/test samples, and prepare OpenAI Batch API
        requests for conditional-y augmentation.

Main design
-----------
We follow the scalar no-children setup of Blundell, Chen, and Kristensen (2007):
- y  : food budget share
- x  : log total expenditure (endogenous regressor)
- z  : log gross earnings (instrument)

The joint law of (x,z) is calibrated from the public Engel95 data set. We mimic the
paper's Simulation 1 by drawing (x,z) from a smoothed empirical density and then
constructing the latent share

    y* = h0(x) + eps,
    eps = m0(z) - h0(x) + v,

where v is mean-zero Gaussian noise and m0(z) = E[h0(X)|Z=z]. Before support
correction this guarantees

    E[y* - h0(x) | z] = 0.

To keep the simulated outcome semantically consistent with a budget share, the stored
outcome is then clipped to [y_eps, 1-y_eps]:

    y = clip(y*, y_eps, 1-y_eps).

When sigma_v is small, this keeps the simulated data close to the BCK-style design while
ensuring GPT, the saved bundle, and the estimators all see the same bounded share.

Relative to the exact BCK nonlinear benchmark h0(x) = Phi((x-c)/s), we use a
GPT-friendly decreasing probit Engel curve fitted to the public Engel95 food-share
relationship:

    h0(x) = a - b * Phi((x - c)/d),

which is a decreasing S-shape consistent with Engel's law for food share.

OpenAI usage
------------
For each Monte Carlo training set, we create one batch request. GPT is asked to keep
(log_total_expenditure, log_gross_earnings) fixed and generate K plausible alternative
food shares for every row. Rows are sorted by (x,z) in the prompt only to make the
pattern smoother; original row ids are preserved.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from patsy import dmatrix, build_design_matrices
from scipy.optimize import minimize
from scipy.stats import norm

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------------------------------------------------------
# Public calibration data
# -----------------------------------------------------------------------------


def load_engel95() -> pd.DataFrame:
    """Load the public Engel95 data.

    The Rdatasets mirror includes the exact variables needed for the scalar Engel-curve
    design: food, logexp, logwages, nkids.
    """
    df = pd.read_csv("Engel95.csv")
    if "rownames" in df.columns:
        df = df.drop(columns=["rownames"])
    required = {"food", "logexp", "logwages", "nkids"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Engel95 data are missing required columns: {sorted(missing)}")
    return df


# -----------------------------------------------------------------------------
# Calibration helpers
# -----------------------------------------------------------------------------


def subset_no_children(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[df["nkids"].to_numpy(dtype=float) == 0].copy()
    out = out[["food", "logexp", "logwages", "nkids"]].reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("No no-children observations found in Engel95.")
    return out


@dataclass
class ProbitH0Params:
    upper: float
    span: float
    center: float
    scale: float


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def h0_decreasing_probit(x: np.ndarray, pars: ProbitH0Params) -> np.ndarray:
    x = np.asarray(x, float)
    return pars.upper - pars.span * norm.cdf((x - pars.center) / pars.scale)



def fit_decreasing_probit_h0(x: np.ndarray, y: np.ndarray) -> ProbitH0Params:
    """Fit y ≈ a - b * Phi((x-c)/d) by nonlinear least squares.

    This keeps the same CDF family as BCK's nonlinear design but enforces a decreasing
    Engel curve suitable for food share.
    """
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)

    y_lo = float(np.quantile(y, 0.05))
    y_hi = float(np.quantile(y, 0.95))
    x_med = float(np.median(x))
    x_sd = float(np.std(x, ddof=1))
    if x_sd <= 1e-8:
        x_sd = 1.0

    a0 = min(max(y_hi, 1e-3), 0.999)
    span0 = min(max(y_hi - y_lo, 1e-3), a0 - 1e-3)
    p0 = np.array(
        [
            math.log(a0 / (1.0 - a0)),
            math.log((span0 / a0) / (1.0 - span0 / a0)),
            x_med,
            math.log(max(0.5 * x_sd, 1e-2)),
        ],
        dtype=float,
    )

    def unpack(p: np.ndarray) -> ProbitH0Params:
        upper = float(np.clip(_sigmoid(p[0]), 1e-4, 0.9999))
        frac = float(np.clip(_sigmoid(p[1]), 1e-4, 0.9999))
        span = float(np.clip(frac * upper, 1e-4, upper - 1e-4))
        center = float(p[2])
        scale = float(np.exp(p[3]))
        return ProbitH0Params(upper=upper, span=span, center=center, scale=scale)

    def obj(p: np.ndarray) -> float:
        pars = unpack(p)
        fit = h0_decreasing_probit(x, pars)
        mse = float(np.mean((y - fit) ** 2))
        mse += 1e-4 * float((pars.center - x_med) ** 2 / (x_sd ** 2))
        mse += 1e-4 * float((math.log(pars.scale) - math.log(max(0.5 * x_sd, 1e-2))) ** 2)
        return mse

    res = minimize(obj, p0, method="L-BFGS-B")
    pars = unpack(np.asarray(res.x if res.success else p0, float))
    return pars


@dataclass
class KDE2DConfig:
    bandwidth: str = "scott"
    ridge: float = 1e-8


class GaussianKDEResampler2D:
    """Simple Gaussian-kernel resampler for a 2D empirical distribution."""

    def __init__(self, data: np.ndarray, *, bandwidth: str = "scott", ridge: float = 1e-8):
        data = np.asarray(data, float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("data must be (n, 2)")
        self.data = data
        self.n = int(data.shape[0])
        self.d = 2
        self.ridge = float(ridge)

        cov = np.cov(self.data, rowvar=False)
        cov = 0.5 * (cov + cov.T) + self.ridge * np.eye(2)
        if bandwidth == "scott":
            factor = self.n ** (-1.0 / (self.d + 4.0))
        elif bandwidth == "silverman":
            factor = ((self.n * (self.d + 2.0)) / 4.0) ** (-1.0 / (self.d + 4.0))
        else:
            try:
                factor = float(bandwidth)
            except Exception as e:
                raise ValueError("bandwidth must be 'scott', 'silverman', or a positive float") from e
            if factor <= 0.0:
                raise ValueError("bandwidth factor must be positive")

        self.bandwidth = bandwidth
        self.factor = float(factor)
        self.kernel_cov = cov * (self.factor ** 2)
        self.kernel_cov = 0.5 * (self.kernel_cov + self.kernel_cov.T) + self.ridge * np.eye(2)

    def sample(self, size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        idx = rng.integers(0, self.n, size=int(size))
        centers = self.data[idx]
        noise = rng.multivariate_normal(np.zeros(2, dtype=float), self.kernel_cov, size=int(size))
        out = centers + noise
        return out[:, 0].astype(float), out[:, 1].astype(float)


@dataclass
class SplineMeanMeta:
    design_info: Any
    coef: np.ndarray


def fit_spline_mean(z: np.ndarray, target: np.ndarray, *, df: int = 7) -> SplineMeanMeta:
    z = np.asarray(z, float).reshape(-1)
    target = np.asarray(target, float).reshape(-1)
    dm = dmatrix(f"0 + cr(z, df={int(df)})", {"z": z})
    B = np.asarray(dm, float)
    Q = np.column_stack([np.ones_like(z), B])
    coef = np.linalg.lstsq(Q, target, rcond=None)[0]
    return SplineMeanMeta(design_info=dm.design_info, coef=np.asarray(coef, float))


def predict_spline_mean(meta: SplineMeanMeta, z_new: np.ndarray) -> np.ndarray:
    z_new = np.asarray(z_new, float).reshape(-1)
    B = np.asarray(build_design_matrices([meta.design_info], {"z": z_new})[0], float)
    Q = np.column_stack([np.ones_like(z_new), B])
    return (Q @ meta.coef).astype(float)


@dataclass
class CalibrationMeta:
    subset: str
    n_source: int
    kde_bandwidth: str
    df_m: int
    sigma_v: float
    sigma_v_mode: str
    clip_y: bool
    y_eps: float
    h0_params: Dict[str, float]
    x_quantiles: Dict[str, float]
    z_quantiles: Dict[str, float]
    y_quantiles: Dict[str, float]


@dataclass
class MonteCarloRecord:
    N: int
    rep: int
    train_seed: int
    test_seed: int
    custom_id: str
    k_per_row: int
    y_train: np.ndarray
    x_train: np.ndarray
    z_train: np.ndarray
    y_test: np.ndarray
    x_test: np.ndarray
    z_test: np.ndarray
    train_clip_rate: float = 0.0
    test_clip_rate: float = 0.0
    gpt_ok: bool = False
    gpt_error: Optional[str] = None
    gpt_yalt_matrix: Optional[np.ndarray] = None
    gpt_conditional_df: Optional[pd.DataFrame] = None


# -----------------------------------------------------------------------------
# DGP calibration and simulation
# -----------------------------------------------------------------------------


def calibrate_engel95_design(
    *,
    bandwidth: str = "scott",
    df_m: int = 7,
    sigma_v_mode: str = "fixed",
    sigma_v_fixed: float = 0.01,
    sigma_v_cap: float = 0.02,
    aux_size: int = 100_000,
    random_state: int = 123,
    clip_y: bool = True,
    y_eps: float = 0.005,
) -> Tuple[GaussianKDEResampler2D, ProbitH0Params, SplineMeanMeta, CalibrationMeta]:
    df = load_engel95()
    df0 = subset_no_children(df)

    y_real = np.clip(df0["food"].to_numpy(dtype=float), y_eps, 1.0 - y_eps) if clip_y else df0["food"].to_numpy(dtype=float)
    x_real = df0["logexp"].to_numpy(dtype=float)
    z_real = df0["logwages"].to_numpy(dtype=float)

    h0_params = fit_decreasing_probit_h0(x_real, y_real)

    kde = GaussianKDEResampler2D(
        np.column_stack([x_real, z_real]),
        bandwidth=bandwidth,
        ridge=1e-8,
    )

    rng = np.random.default_rng(int(random_state))
    x_aux, z_aux = kde.sample(int(aux_size), rng)
    m_meta = fit_spline_mean(z_aux, h0_decreasing_probit(x_aux, h0_params), df=int(df_m))

    if sigma_v_mode == "fixed":
        sigma_v = float(sigma_v_fixed)
    else:
        m_real = predict_spline_mean(m_meta, z_real)
        var_real = float(np.var(y_real, ddof=1))
        var_m = float(np.var(m_real, ddof=1))
        sigma_v = math.sqrt(max(1e-8, var_real - min(var_real, var_m)))
        if sigma_v_mode == "matchvar_cap":
            sigma_v = min(float(sigma_v), float(sigma_v_cap))
        elif sigma_v_mode != "matchvar":
            raise ValueError("sigma_v_mode must be one of: fixed, matchvar, matchvar_cap")

    meta = CalibrationMeta(
        subset="Engel95 no-children (nkids == 0)",
        n_source=int(len(df0)),
        kde_bandwidth=str(bandwidth),
        df_m=int(df_m),
        sigma_v=float(sigma_v),
        sigma_v_mode=str(sigma_v_mode),
        clip_y=bool(clip_y),
        y_eps=float(y_eps),
        h0_params=asdict(h0_params),
        x_quantiles={
            "q05": float(np.quantile(x_real, 0.05)),
            "q50": float(np.quantile(x_real, 0.50)),
            "q95": float(np.quantile(x_real, 0.95)),
        },
        z_quantiles={
            "q05": float(np.quantile(z_real, 0.05)),
            "q50": float(np.quantile(z_real, 0.50)),
            "q95": float(np.quantile(z_real, 0.95)),
        },
        y_quantiles={
            "q05": float(np.quantile(y_real, 0.05)),
            "q50": float(np.quantile(y_real, 0.50)),
            "q95": float(np.quantile(y_real, 0.95)),
        },
    )
    return kde, h0_params, m_meta, meta



def simulate_scalar_engel_dgp(
    n: int,
    *,
    kde: GaussianKDEResampler2D,
    h0_params: ProbitH0Params,
    m_meta: SplineMeanMeta,
    sigma_v: float,
    rng: np.random.Generator,
    clip_y: bool = True,
    y_eps: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x, z = kde.sample(int(n), rng)
    h = h0_decreasing_probit(x, h0_params)
    m = predict_spline_mean(m_meta, z)
    v = rng.normal(loc=0.0, scale=float(sigma_v), size=int(n))
    y_latent = m + v
    if clip_y:
        y = np.clip(y_latent, y_eps, 1.0 - y_eps)
        clip_rate = float(np.mean((y_latent <= y_eps) | (y_latent >= 1.0 - y_eps)))
    else:
        y = y_latent
        clip_rate = 0.0
    return y.astype(float), x.astype(float), z.astype(float), clip_rate


# -----------------------------------------------------------------------------
# Prompt helpers
# -----------------------------------------------------------------------------

ENGEL_SYSTEM_PROMPT = """You are a conservative conditional outcome generator for a scalar Engel-curve simulation.

You will receive simulated household rows with columns:
[id, log_total_expenditure, log_gross_earnings, food_share]

Each row represents a working-age couple without children from an expenditure survey.
Rows are sorted by log_total_expenditure and then log_gross_earnings only to make the broad pattern easier to see.

For each row, KEEP log_total_expenditure and log_gross_earnings FIXED and generate K alternative plausible food_share values.
Your main goal is to infer a smoothed local conditional distribution, not to reproduce the exact observed value for the same row.

Main objective:
- Produce conservative, smoothed alternatives of plausible food_share values.
- Underfit rather than overfit if uncertain.
- Shrink toward the local median pattern of nearby rows if uncertain.

Economic guidance:
- Engel's law implies that, on average, food_share tends to FALL as total expenditure rises.
- log_total_expenditure is the main predictor.
- log_gross_earnings may matter, but its direct effect beyond expenditure should be weaker, smoother, and secondary.
- Nearby rows in (log_total_expenditure, log_gross_earnings) should have similar central food_share values.
- Keep every alternative food_share between 0 and 1.
- Avoid extreme tails unless strongly supported by the overall pattern in the data.

Important:
- Use the observed food_share values only to learn the broad and local conditional pattern.
- Treat the observed food_share in the same row as noisy; do NOT simply copy it.
- Do NOT reproduce row-specific noise.
- The K alternatives should be ORDERED from low to high and should represent a conservative spread around the same conditional distribution.
- The middle value should be close to the conditional center.
- The outer values should be mild deviations around that center, not extreme outliers.

Output STRICT JSON only with schema:
{"rows": [[id, food_share_alt_1, food_share_alt_2, ..., food_share_alt_K], ...]}

Rules:
- Return exactly one output row for each input id.
- Keep ids unchanged.
- Every alternative must be either a finite number in [0,1].
- Do not return null.
- No text, no explanations, no markdown.
"""


def matrix_to_text(df: pd.DataFrame, fmt: str = "csv", decimals: int = 4) -> str:
    if fmt == "csv":
        return df.to_csv(index=False, float_format=f"%.{decimals}f").strip()
    if fmt == "array":
        arr = df.to_numpy()
        return np.array2string(
            arr,
            precision=decimals,
            separator=", ",
            suppress_small=False,
            threshold=arr.size + 10,
            max_line_width=200,
        )
    raise ValueError("fmt must be 'csv' or 'array'")



def build_user_prompt(
    *,
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    k_per_row: int,
    matrix_format: str = "csv",
    decimals: int = 4,
) -> str:
    n = int(len(y))
    df = pd.DataFrame(
        {
            "id": np.arange(n, dtype=int),
            "log_total_expenditure": np.asarray(x, float),
            "log_gross_earnings": np.asarray(z, float),
            "food_share": np.asarray(y, float),
        }
    )
    df_sorted = df.sort_values(["log_total_expenditure", "log_gross_earnings"], kind="mergesort").reset_index(drop=True)

    summary = {
        "food_share_q10": float(df["food_share"].quantile(0.10)),
        "food_share_q50": float(df["food_share"].quantile(0.50)),
        "food_share_q90": float(df["food_share"].quantile(0.90)),
        "log_total_expenditure_q10": float(df["log_total_expenditure"].quantile(0.10)),
        "log_total_expenditure_q50": float(df["log_total_expenditure"].quantile(0.50)),
        "log_total_expenditure_q90": float(df["log_total_expenditure"].quantile(0.90)),
        "log_gross_earnings_q10": float(df["log_gross_earnings"].quantile(0.10)),
        "log_gross_earnings_q50": float(df["log_gross_earnings"].quantile(0.50)),
        "log_gross_earnings_q90": float(df["log_gross_earnings"].quantile(0.90)),
    }

    return (
        f"Observed training sample:\n"
        f"- number of rows: {n}\n"
        f"- requested number of alternative food shares per row: {int(k_per_row)}\n"
        f"- summary quantiles: {json.dumps(summary)}\n\n"
        f"Below is the full observed data matrix in {matrix_format.upper()} format.\n"
        f"IMPORTANT: rows are sorted only for readability. You must still keep the original id unchanged in the output.\n"
        f"IMPORTANT: estimate a smoothed conditional center for each row rather than copying that row's exact observed food_share.\n\n"
        f"{matrix_to_text(df_sorted, fmt=matrix_format, decimals=decimals)}\n\n"
        f"Return STRICT JSON only with schema:\n"
        f'{{"rows": [[id, food_share_alt_1, ..., food_share_alt_{int(k_per_row)}], ...]}}\n\n'
        f"Requirements:\n"
        f"- return exactly {n} rows in 'rows',\n"
        f"- each row must begin with the integer id,\n"
        f"- each row must contain exactly {int(k_per_row)} alternative food shares after the id,\n"
        f"- each row's alternatives must be ordered from low to high,\n"
        f"- each alternative must be a number in [0,1],\n"
        f"- do not return null,\n"
        f"- do not simply repeat the row's observed food_share."
    )



def make_batch_request(
    *,
    custom_id: str,
    model: str,
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    k_per_row: int,
    matrix_format: str = "csv",
    decimals: int = 4,
    temperature: float = 0.10,
    max_completion_tokens: int = 12000,
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(
        x=x,
        z=z,
        y=y,
        k_per_row=int(k_per_row),
        matrix_format=matrix_format,
        decimals=decimals,
    )
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": float(temperature),
            "response_format": {"type": "json_object"},
            "max_completion_tokens": int(max_completion_tokens),
            "messages": [
                {"role": "system", "content": ENGEL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        },
    }


# -----------------------------------------------------------------------------
# Batch output parsing
# -----------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()



def _extract_json_object(text: str) -> Dict[str, Any]:
    t = _strip_code_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        return json.loads(t[start : end + 1])
    raise ValueError("Could not parse JSON object from model output")



def _coerce_yalt_matrix(rows: Any, *, n: int, k: int) -> np.ndarray:
    out = np.full((n, k), np.nan, dtype=float)
    if not isinstance(rows, list):
        raise ValueError("JSON 'rows' is not a list")
    for row in rows:
        if not isinstance(row, list) or len(row) < 1:
            continue
        try:
            idx = int(row[0])
        except Exception:
            continue
        if not (0 <= idx < n):
            continue
        vals = row[1:]
        vals = vals[:k] + [None] * max(0, k - len(vals))
        parsed: List[float] = []
        for v in vals:
            if v is None:
                parsed.append(np.nan)
            else:
                try:
                    val = float(v)
                    if np.isfinite(val) and 0.0 <= val <= 1.0:
                        parsed.append(val)
                    else:
                        parsed.append(np.nan)
                except Exception:
                    parsed.append(np.nan)
        out[idx, :] = np.asarray(parsed, float)
    return out



def parse_batch_chatcompletion_output_line(obj: Dict[str, Any], *, expected_n: int, expected_k: int) -> np.ndarray:
    if obj.get("error") is not None:
        raise RuntimeError(str(obj["error"]))
    body = obj["response"]["body"]
    choices = body.get("choices", [])
    if len(choices) == 0:
        raise RuntimeError("No choices found in batch response body")
    content = choices[0]["message"]["content"]
    if not isinstance(content, str):
        raise RuntimeError("Expected message.content to be a string")
    parsed = _extract_json_object(content)
    rows = parsed.get("rows", None)
    if rows is None:
        raise RuntimeError("JSON did not contain key 'rows'")
    mat = _coerce_yalt_matrix(rows, n=int(expected_n), k=int(expected_k))
    if not np.isfinite(mat).any():
        raise RuntimeError("Parsed matrix contains no usable synthetic food-share values")
    return mat


# -----------------------------------------------------------------------------
# Bundle generation
# -----------------------------------------------------------------------------

def generate_bundle(
    *,
    mc: int,
    Ns: Iterable[int],
    n_test: int,
    k_per_row: int,
    random_state: int,
    bandwidth: str,
    df_m: int,
    sigma_v_mode: str,
    sigma_v_fixed: float,
    sigma_v_cap: float,
    aux_size: int,
    clip_y: bool,
    y_eps: float,
) -> Dict[str, Any]:
    kde, h0_params, m_meta, calib_meta = calibrate_engel95_design(
        bandwidth=bandwidth,
        df_m=df_m,
        sigma_v_mode=sigma_v_mode,
        sigma_v_fixed=sigma_v_fixed,
        sigma_v_cap=sigma_v_cap,
        aux_size=aux_size,
        random_state=random_state,
        clip_y=clip_y,
        y_eps=y_eps,
    )

    seed_rng = np.random.default_rng(int(random_state))

    bundle: Dict[str, Any] = {
        "_meta": {
            "design": "Engel95 scalar food-share Monte Carlo with decreasing probit h0",
            "calibration": asdict(calib_meta),
            "k_per_row": int(k_per_row),
            "Ns": [int(v) for v in Ns],
            "mc": int(mc),
            "n_test": int(n_test),
        }
    }

    for N in [int(v) for v in Ns]:
        records: List[Dict[str, Any]] = []
        for rep in range(int(mc)):
            train_seed = int(seed_rng.integers(0, 2**32 - 1))
            test_seed = int(seed_rng.integers(0, 2**32 - 1))

            y_tr, x_tr, z_tr, clip_tr = simulate_scalar_engel_dgp(
                int(N),
                kde=kde,
                h0_params=h0_params,
                m_meta=m_meta,
                sigma_v=calib_meta.sigma_v,
                rng=np.random.default_rng(train_seed),
                clip_y=bool(clip_y),
                y_eps=float(y_eps),
            )
            y_te, x_te, z_te, clip_te = simulate_scalar_engel_dgp(
                int(n_test),
                kde=kde,
                h0_params=h0_params,
                m_meta=m_meta,
                sigma_v=calib_meta.sigma_v,
                rng=np.random.default_rng(test_seed),
                clip_y=bool(clip_y),
                y_eps=float(y_eps),
            )
            custom_id = f"engel95_foodshare_N{N}_rep{rep:04d}"
            rec = MonteCarloRecord(
                N=int(N),
                rep=int(rep),
                train_seed=train_seed,
                test_seed=test_seed,
                custom_id=custom_id,
                k_per_row=int(k_per_row),
                y_train=np.asarray(y_tr, float),
                x_train=np.asarray(x_tr, float),
                z_train=np.asarray(z_tr, float),
                y_test=np.asarray(y_te, float),
                x_test=np.asarray(x_te, float),
                z_test=np.asarray(z_te, float),
                train_clip_rate=float(clip_tr),
                test_clip_rate=float(clip_te),
            )
            records.append(asdict(rec))
        bundle[int(N)] = records
    return bundle


# -----------------------------------------------------------------------------
# Prepare / assemble CLI
# -----------------------------------------------------------------------------


def do_prepare(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    Ns = [int(v) for v in args.Ns]
    bundle = generate_bundle(
        mc=int(args.mc),
        Ns=Ns,
        n_test=int(args.n_test),
        k_per_row=int(args.k_per_row),
        random_state=int(args.seed_base),
        bandwidth=str(args.kde_bandwidth),
        df_m=int(args.df_m),
        sigma_v_mode=str(args.sigma_v_mode),
        sigma_v_fixed=float(args.sigma_v_fixed),
        sigma_v_cap=float(args.sigma_v_cap),
        aux_size=int(args.aux_size),
        clip_y=bool(args.clip_y),
        y_eps=float(args.y_eps),
    )

    batch_reqs: List[Dict[str, Any]] = []
    for N in Ns:
        for rec in bundle[int(N)]:
            req = make_batch_request(
                custom_id=str(rec["custom_id"]),
                model=str(args.model),
                x=np.asarray(rec["x_train"], float),
                z=np.asarray(rec["z_train"], float),
                y=np.asarray(rec["y_train"], float),
                k_per_row=int(rec["k_per_row"]),
                matrix_format=str(args.matrix_format),
                decimals=int(args.decimals),
                temperature=float(args.temperature),
                max_completion_tokens=int(args.max_completion_tokens),
            )
            batch_reqs.append(req)

    bundle_path = out_dir / str(args.bundle_name)
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)

    jsonl_path = out_dir / str(args.batch_jsonl_name)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for req in batch_reqs:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    print(f"[prepare] wrote Monte Carlo bundle to: {bundle_path}")
    print(f"[prepare] wrote batch JSONL to: {jsonl_path}")
    print(f"[prepare] total requests: {len(batch_reqs)}")

    if args.submit:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed. Run `pip install openai` first.")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=api_key)
        with open(jsonl_path, "rb") as f:
            up = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=up.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "experiment": "engel95_scalar_foodshare_gpt_conditional_y",
                "bundle": str(args.bundle_name),
                "matrix_format": str(args.matrix_format),
                "model": str(args.model),
                "k_per_row": str(args.k_per_row),
            },
        )
        print(f"[prepare] uploaded file_id: {up.id}")
        print(f"[prepare] created batch_id: {batch.id}")
        print("[prepare] poll the batch in the dashboard or via API, then download the output JSONL.")



def do_assemble(args: argparse.Namespace) -> None:
    bundle_in = Path(args.bundle_in).resolve()
    batch_output_jsonl = Path(args.batch_output_jsonl).resolve()
    bundle_out = Path(args.bundle_out).resolve()

    with open(bundle_in, "rb") as f:
        bundle = pickle.load(f)

    rec_map: Dict[str, Dict[str, Any]] = {}
    for k, v in bundle.items():
        if k == "_meta":
            continue
        for rec in v:
            rec_map[str(rec["custom_id"])] = rec

    n_ok = 0
    n_fail = 0

    with open(batch_output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id")
            if custom_id not in rec_map:
                continue
            rec = rec_map[custom_id]
            n = int(rec["N"])
            k = int(rec["k_per_row"])
            try:
                mat = parse_batch_chatcompletion_output_line(obj, expected_n=n, expected_k=k)
                rec["gpt_yalt_matrix"] = np.asarray(mat, float)
                rec["gpt_ok"] = True
                rec["gpt_error"] = None

                df = pd.DataFrame(
                    {
                        "log_total_expenditure": np.asarray(rec["x_train"], float),
                        "log_gross_earnings": np.asarray(rec["z_train"], float),
                        "food_share": np.asarray(rec["y_train"], float),
                    }
                )
                for j in range(k):
                    df[f"synthetic_food_share{j+1}"] = mat[:, j]
                rec["gpt_conditional_df"] = df
                n_ok += 1
            except Exception as e:
                rec["gpt_yalt_matrix"] = None
                rec["gpt_conditional_df"] = None
                rec["gpt_ok"] = False
                rec["gpt_error"] = repr(e)
                n_fail += 1

    with open(bundle_out, "wb") as f:
        pickle.dump(bundle, f)

    print(f"[assemble] wrote completed bundle to: {bundle_out}")
    print(f"[assemble] GPT conditional-label sets ok: {n_ok}")
    print(f"[assemble] GPT conditional-label sets failed: {n_fail}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        "Calibrate Engel95-based Monte Carlo datasets and prepare GPT conditional-y batch inputs"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_prep = sub.add_parser("prepare")
    ap_prep.add_argument("--Ns", type=int, nargs="+", default=[100])
    ap_prep.add_argument("--mc", type=int, default=10)
    ap_prep.add_argument("--n_test", type=int, default=100)
    ap_prep.add_argument("--k_per_row", type=int, default=5)
    ap_prep.add_argument("--model", type=str, default="gpt-5.2")
    ap_prep.add_argument("--matrix_format", type=str, choices=["csv", "array"], default="csv")
    ap_prep.add_argument("--decimals", type=int, default=4)
    ap_prep.add_argument("--temperature", type=float, default=0.10)
    ap_prep.add_argument("--max_completion_tokens", type=int, default=12000)
    ap_prep.add_argument("--seed_base", type=int, default=12345)
    ap_prep.add_argument("--out_dir", type=str, default=".")
    ap_prep.add_argument("--bundle_name", type=str, default="engel95_gpt_training_bundle.pkl")
    ap_prep.add_argument("--batch_jsonl_name", type=str, default="engel95_gpt_batch_input.jsonl")
    ap_prep.add_argument("--submit", action="store_true")

    ap_prep.add_argument("--kde_bandwidth", type=str, default="scott")
    ap_prep.add_argument("--df_m", type=int, default=7)
    ap_prep.add_argument("--aux_size", type=int, default=10000)
    ap_prep.add_argument("--sigma_v_mode", type=str, choices=["fixed", "matchvar", "matchvar_cap"], default="fixed")
    ap_prep.add_argument("--sigma_v_fixed", type=float, default=0.01)
    ap_prep.add_argument("--sigma_v_cap", type=float, default=0.02)
    ap_prep.add_argument("--clip_y", dest="clip_y", action="store_true")
    ap_prep.add_argument("--no_clip_y", dest="clip_y", action="store_false")
    ap_prep.set_defaults(clip_y=True)
    ap_prep.add_argument("--y_eps", type=float, default=0.005)

    ap_assm = sub.add_parser("assemble")
    ap_assm.add_argument("--bundle_in", type=str, required=True)
    ap_assm.add_argument("--batch_output_jsonl", type=str, required=True)
    ap_assm.add_argument("--bundle_out", type=str, default="engel95_gpt_training_bundle_complete.pkl")

    return ap


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "prepare":
        do_prepare(args)
    elif args.cmd == "assemble":
        do_assemble(args)
