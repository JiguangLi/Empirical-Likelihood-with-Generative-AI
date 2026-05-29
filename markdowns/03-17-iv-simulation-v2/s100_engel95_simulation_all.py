#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Evaluate penalized spline NPIV vs StructuralBETELV2 with alpha=0 and
        GPT conditional-y augmentation on the Engel95-calibrated scalar food-share
        Monte Carlo design.

This script is deliberately modeled on the user's existing np_simulation_all.py but
replaces the Newey–Powell DGP with the scalar Engel-curve Monte Carlo bundle created by
engel95_prepare_training_synthetic_data.py.

Methods compared
----------------
1. NPIV baseline: penalized cubic-spline sieve IV. The structural function h(x) is
   approximated with a cubic regression-spline basis in x, the first stage uses a
   cubic regression-spline basis in z, and the second stage solves a constrained
   least-squares problem with a spline roughness penalty.
2. SBETEL(alpha=0): Bayesian bootstrap only.
3. GPT-conditional-y SBETEL(alpha>0): for each bootstrap draw, sample m row ids,
   choose one GPT synthetic y-value for each selected id, augment the training data,
   and fit StructuralBETELV2 exactly as in the user's current code.

The structural target is the known h0(x) stored in the bundle metadata.
"""

from __future__ import annotations
import pathlib
import argparse
import importlib.util
import inspect
import math
import pickle
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from patsy import dmatrix, build_design_matrices
from scipy.linalg import qr
from scipy.stats import norm
from tqdm import tqdm

from feature_map import NSFeatureMap


# -----------------------------------------------------------------------------
# Dynamic local loader: prefer the user's EL package, fall back to the uploaded
# local files in the current directory.
# -----------------------------------------------------------------------------


def load_local_el() -> Any:
    try:
        import EL  # type: ignore
        return EL
    except Exception:
        pass

    root = Path(__file__).resolve().parent
    pkg_name = "_el_local_pkg"
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(root)]
    sys.modules[pkg_name] = pkg

    def _load(name: str, filename: str):
        spec = importlib.util.spec_from_file_location(f"{pkg_name}.{name}", root / filename)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {filename}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        setattr(pkg, name, mod)
        return mod

    mod_gen = _load("GenELV2", "GenELV2.py")
    mod_struct = _load("StructuralBETELV2", "StructuralBETELV2.py")

    # expose the classes at package level to mimic `import EL`
    pkg.GenELV2 = mod_gen.GenELV2
    pkg.StructuralBETELV2 = mod_struct.StructuralBETELV2
    return pkg


EL = load_local_el()


# -----------------------------------------------------------------------------
# DGP / target h0 from bundle metadata
# -----------------------------------------------------------------------------


def h0_from_meta(x: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    pars = meta["calibration"]["h0_params"]
    upper = float(pars["upper"])
    span = float(pars["span"])
    center = float(pars["center"])
    scale = float(pars["scale"])
    x = np.asarray(x, float)
    return upper - span * norm.cdf((x - center) / scale)


# -----------------------------------------------------------------------------
# Helper functions reused from the existing code base, but kept local to avoid
# import brittleness.
# -----------------------------------------------------------------------------


def make_structural_betel(**kwargs):
    sig = inspect.signature(EL.StructuralBETELV2.__init__)
    keep = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return EL.StructuralBETELV2(**keep)



def theta_init_2sls(Phi: np.ndarray, Y: np.ndarray, Z: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    Phi = np.asarray(Phi, float)
    y = np.asarray(Y, float).reshape(-1)
    Z = np.asarray(Z, float)
    N, p = Phi.shape
    if Z.shape[0] != N:
        raise ValueError("Z must have same number of rows as Phi/Y.")

    Q, _ = qr(Z, mode="economic")
    Phi_t = Q.T @ Phi
    y_t = Q.T @ y

    G = Phi_t.T @ Phi_t
    if ridge > 0.0:
        G = G + float(ridge) * np.eye(p)
    b = Phi_t.T @ y_t
    try:
        return np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(G, b, rcond=None)[0]



def _pred_mean(preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, float)
    if preds.ndim != 2:
        raise ValueError("preds must be 2D (B, n_points).")
    return preds.mean(axis=0)



def _fit_cr_design(z: np.ndarray, *, df: int) -> Tuple[Any, np.ndarray]:
    dm = dmatrix(f"0 + cr(z, df={int(df)})", {"z": np.asarray(z, float).reshape(-1)})
    B = np.asarray(dm, float)
    return dm.design_info, B



def _cr_design_transform(design_info: Any, z_new: np.ndarray) -> np.ndarray:
    mat = build_design_matrices([design_info], {"z": np.asarray(z_new, float).reshape(-1)})[0]
    return np.asarray(mat, float)


# -----------------------------------------------------------------------------
# Penalized cubic-spline sieve NPIV baseline
# -----------------------------------------------------------------------------


def compute_spline_roughness_penalty(
    design_info: Any,
    x_min: float,
    x_max: float,
    *,
    n_grid: int = 400,
    normalize: bool = True,
) -> np.ndarray:
    r"""
    Approximate the cubic-spline roughness matrix

        Lambda_{jk} = \int b_j''(x) b_k''(x) dx

    on a dense grid over the training support. For natural cubic splines, this gives
    a standard roughness penalty whose null space is (approximately) linear trends.
    """
    x_min = float(x_min)
    x_max = float(x_max)
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("x_min/x_max must be finite")
    if x_max <= x_min:
        x_max = x_min + 1.0

    n_grid = max(int(n_grid), 25)
    grid = np.linspace(x_min, x_max, n_grid)
    B = _cr_design_transform(design_info, grid)  # (G, K)
    dx = float(grid[1] - grid[0])

    # Finite-difference approximation of second derivatives of each basis function.
    D1 = np.gradient(B, dx, axis=0, edge_order=2)
    D2 = np.gradient(D1, dx, axis=0, edge_order=2)

    w = np.full(n_grid, dx, dtype=float)
    w[0] *= 0.5
    w[-1] *= 0.5

    Lambda = D2.T @ (D2 * w[:, None])
    Lambda = 0.5 * (Lambda + Lambda.T)

    if normalize:
        avg_diag = float(np.trace(Lambda) / max(Lambda.shape[0], 1))
        if avg_diag > 0.0:
            Lambda = Lambda / avg_diag
    return Lambda



def solve_constrained_ls_general(
    y: np.ndarray,
    Phat: np.ndarray,
    Lambda: np.ndarray,
    B1: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Solve

        min_beta ||y - Phat beta||^2
        s.t.     beta' Lambda beta <= B1

    by bisection over the Lagrange multiplier. The penalty matrix Lambda may be
    singular (as is typical for spline roughness penalties).
    """
    y = np.asarray(y, float).reshape(-1, 1)
    Phat = np.asarray(Phat, float)
    Lambda = np.asarray(Lambda, float)
    B1 = float(B1)

    PtP = Phat.T @ Phat
    Pty = Phat.T @ y

    beta_ols = np.linalg.pinv(PtP) @ Pty
    s_ols = float(beta_ols.T @ Lambda @ beta_ols)
    if s_ols <= B1 + 1e-12:
        return beta_ols.reshape(-1)

    def penalized_sol(lam: float) -> Tuple[float, np.ndarray]:
        A = PtP + float(lam) * Lambda
        beta = np.linalg.pinv(A) @ Pty
        s = float(beta.T @ Lambda @ beta)
        return s, beta.reshape(-1)

    lam_low, lam_high = 0.0, 1.0
    s_high, beta_high = penalized_sol(lam_high)
    while s_high > B1:
        lam_high *= 2.0
        if lam_high > 1e10:
            break
        s_high, beta_high = penalized_sol(lam_high)

    beta_best = beta_high
    for _ in range(int(max_iter)):
        lam_mid = 0.5 * (lam_low + lam_high)
        s_mid, beta_mid = penalized_sol(lam_mid)
        if abs(s_mid - B1) < tol:
            beta_best = beta_mid
            break
        if s_mid > B1:
            lam_low = lam_mid
        else:
            lam_high = lam_mid
            beta_best = beta_mid

    return beta_best



def penalized_spline_npiv_fit(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    *,
    df_x: int,
    df_z: int,
    B1: float,
    penalty_grid: int = 400,
) -> Tuple[np.ndarray, Any]:
    """
    Penalized sieve IV with cubic regression splines for both h(x) and the first-stage
    regression in z.
    """
    y = np.asarray(y, float).reshape(-1)
    x = np.asarray(x, float).reshape(-1)
    z = np.asarray(z, float).reshape(-1)

    # Structural basis in x.
    x_info, Phi = _fit_cr_design(x, df=int(df_x))
    Lambda = compute_spline_roughness_penalty(
        x_info, float(np.min(x)), float(np.max(x)), n_grid=int(penalty_grid), normalize=True
    )

    # First-stage basis in z.
    z_info, Bz_nc = _fit_cr_design(z, df=int(df_z))
    Q = np.column_stack([np.ones_like(z), Bz_nc])
    QQinv = np.linalg.pinv(Q.T @ Q)

    # Column-wise projection of the structural basis onto the instrument space.
    Phi_hat = Q @ (QQinv @ (Q.T @ Phi))

    beta_hat = solve_constrained_ls_general(y, Phi_hat, Lambda, float(B1))
    return beta_hat, x_info



def penalized_spline_npiv_predictive_mse(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    x_test: np.ndarray,
    *,
    df_x: int,
    df_z: int,
    B1: float,
    h0_eval,
    penalty_grid: int = 400,
) -> Tuple[float, float]:
    """
    Penalized cubic-spline NPIV baseline.
    """
    y = np.asarray(y, float).reshape(-1)
    x = np.asarray(x, float).reshape(-1)
    z = np.asarray(z, float).reshape(-1)
    x_test = np.asarray(x_test, float).reshape(-1)

    beta_hat, x_info = penalized_spline_npiv_fit(
        y, x, z,
        df_x=int(df_x),
        df_z=int(df_z),
        B1=float(B1),
        penalty_grid=int(penalty_grid),
    )

    Phi_tr = _cr_design_transform(x_info, x)
    Phi_te = _cr_design_transform(x_info, x_test)

    h_tr = Phi_tr @ beta_hat
    h_te = Phi_te @ beta_hat

    train_mse = float(np.mean((h_tr - h0_eval(x)) ** 2))
    test_mse = float(np.mean((h_te - h0_eval(x_test)) ** 2))
    return train_mse, test_mse


# -----------------------------------------------------------------------------
# GPT label extraction
# -----------------------------------------------------------------------------


def extract_gpt_yalt_matrix(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    if "gpt_yalt_matrix" in rec and rec["gpt_yalt_matrix"] is not None:
        mat = np.asarray(rec["gpt_yalt_matrix"], float)
        if mat.ndim == 2:
            return mat

    df = rec.get("gpt_conditional_df", None)
    if df is None:
        return None
    df = pd.DataFrame(df).copy()
    synth_cols = [c for c in df.columns if str(c).startswith("synthetic_food_share")]
    if len(synth_cols) == 0:
        synth_cols = [c for c in df.columns if str(c).startswith("synthetic_y")]
    if len(synth_cols) == 0:
        return None
    if "id" in df.columns:
        df = df.sort_values("id").reset_index(drop=True)
    return df[synth_cols].to_numpy(dtype=float)



def record_has_usable_gpt(rec: Dict[str, Any]) -> bool:
    if not bool(rec.get("gpt_ok", False)):
        return False
    mat = extract_gpt_yalt_matrix(rec)
    return bool(mat is not None and mat.ndim == 2 and np.isfinite(mat).any())


# -----------------------------------------------------------------------------
# Baselines and BETEL wrappers
# -----------------------------------------------------------------------------

def structural_betel_predictive_mse_alpha0(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    x_test: np.ndarray,
    *,
    B_boot: int,
    random_state: int,
    df_x: int,
    df_z: int,
    whiten_basis: bool,
    h0_eval,
    outer_method: str = "L-BFGS-B",
    outer_options: Optional[dict] = None,
) -> Tuple[float, float]:
    y = np.asarray(y, float).reshape(-1)
    x = np.asarray(x, float).reshape(-1)
    z = np.asarray(z, float).reshape(-1)
    x_test = np.asarray(x_test, float).reshape(-1)

    phi = NSFeatureMap(df_first=int(df_x), df_rest=int(df_x), tilde=True, center=True, scale=True, add_constant=True)
    Phi_tr = phi.fit_transform(x[:, None])
    Phi_te = phi.transform(x_test[:, None])

    Bz_nc = np.asarray(dmatrix(f"0 + cr(z, df={int(df_z)})", {"z": z}), float)
    Q = np.column_stack([np.ones_like(z), Bz_nc])
    theta0 = theta_init_2sls(Phi_tr, y, Q, ridge=1e-6)

    def g_yh(Y, H):
        return np.asarray(Y, float).reshape(-1, 1) - np.asarray(H, float).reshape(-1, 1)

    betel = make_structural_betel(
        Y=y,
        X=x,
        W=z,
        g_yh=g_yh,
        theta0=theta0,
        phi_map=lambda Xin: phi.transform(np.asarray(Xin).reshape(-1, 1)),
        d_h=1,
        instrument_basis=Q,
        whiten_basis=bool(whiten_basis),
        max_moment_ratio=0.6,
        reg_lambda=0.0,
        init_theta=None,
        outer_use_analytic_grad=False,
        per_v_gmm_start=False,
        random_state=int(random_state),
        newton_tol=1e-6,
    )

    if outer_options is None:
        outer_options = {"maxiter": 50, "ftol": 1e-6, "eps": 1e-6}

    preds_tr: List[np.ndarray] = []
    preds_te: List[np.ndarray] = []
    theta_start = betel.theta0.copy()

    for _ in range(int(B_boot)):
        v = betel.draw_dirichlet()
        out = betel.fit_per_v(v=v, method=outer_method, options=outer_options, theta0=theta_start)
        inner = out.get("inner_info", {}) or {}
        if not inner.get("feasible", True):
            continue
        th = np.asarray(out["theta"], float).reshape(-1)
        if not np.all(np.isfinite(th)):
            continue
        preds_tr.append(Phi_tr @ th)
        preds_te.append(Phi_te @ th)

    if len(preds_tr) == 0:
        h_tr = Phi_tr @ theta0
        h_te = Phi_te @ theta0
    else:
        h_tr = _pred_mean(np.vstack(preds_tr))
        h_te = _pred_mean(np.vstack(preds_te))

    train_mse = float(np.mean((h_tr - h0_eval(x)) ** 2))
    test_mse = float(np.mean((h_te - h0_eval(x_test)) ** 2))
    return train_mse, test_mse



def structural_betel_predictive_mse_gpt_condy_aug(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    x_test: np.ndarray,
    gpt_yalt_matrix: np.ndarray,
    *,
    B_boot: int,
    random_state: int,
    df_x: int,
    df_z: int,
    whiten_basis: bool,
    alpha: float,
    m_ratio: float,
    h0_eval,
    sample_with_replacement: bool = False,
    outer_method: str = "L-BFGS-B",
    outer_options: Optional[dict] = None,
) -> Tuple[float, float]:
    y = np.asarray(y, float).reshape(-1)
    x = np.asarray(x, float).reshape(-1)
    z = np.asarray(z, float).reshape(-1)
    x_test = np.asarray(x_test, float).reshape(-1)
    yalts = np.asarray(gpt_yalt_matrix, float)
    if yalts.ndim != 2:
        raise ValueError("gpt_yalt_matrix must be 2D")
    N = int(x.size)
    if yalts.shape[0] != N:
        raise ValueError("gpt_yalt_matrix and training sample have different row counts")

    m = int(max(1, round(float(m_ratio) * N)))
    alpha = float(alpha)
    if not (alpha > 0.0):
        raise ValueError("GPT augmentation requires alpha > 0")

    valid_rows = np.where(np.any(np.isfinite(yalts), axis=1))[0]
    if len(valid_rows) == 0:
        return np.nan, np.nan

    phi = NSFeatureMap(df_first=int(df_x), df_rest=int(df_x), tilde=True, center=True, scale=True, add_constant=True)
    Phi_tr = phi.fit_transform(x[:, None])
    Phi_te = phi.transform(x_test[:, None])

    z_info, Bz_obs_nc = _fit_cr_design(z, df=int(df_z))
    Q_obs = np.column_stack([np.ones_like(z), Bz_obs_nc])

    theta0 = theta_init_2sls(Phi_tr, y, Q_obs, ridge=1e-6)

    def g_yh(Y, H):
        return np.asarray(Y, float).reshape(-1, 1) - np.asarray(H, float).reshape(-1, 1)

    if outer_options is None:
        outer_options = {"maxiter": 60, "ftol": 1e-6, "eps": 1e-6}

    rng = np.random.default_rng(int(random_state))
    conc_syn = alpha / float(m)
    conc = np.concatenate([np.ones(N, float), np.full(m, conc_syn, float)])

    preds_tr: List[np.ndarray] = []
    preds_te: List[np.ndarray] = []

    for _ in range(int(B_boot)):
        replace_rows = bool(sample_with_replacement or (m > len(valid_rows)))
        row_idx = rng.choice(valid_rows, size=m, replace=replace_rows)

        y_s = np.empty(m, dtype=float)
        x_s = x[row_idx].copy()
        z_s = z[row_idx].copy()

        for j, i in enumerate(row_idx):
            good = np.where(np.isfinite(yalts[i]))[0]
            if len(good) == 0:
                y_s[j] = float(np.clip(y[i], 0.0, 1.0))
            else:
                k = int(rng.choice(good))
                y_s[j] = float(yalts[i, k])

        y_aug = np.concatenate([y, y_s])
        x_aug = np.concatenate([x, x_s])
        z_aug = np.concatenate([z, z_s])

        Phi_aug = phi.transform(x_aug[:, None])
        Bz_syn_nc = _cr_design_transform(z_info, z_s)
        Q_syn = np.column_stack([np.ones_like(z_s), Bz_syn_nc])
        Q_aug = np.vstack([Q_obs, Q_syn])

        v_aug = rng.dirichlet(conc)

        betel = make_structural_betel(
            Y=y_aug,
            X=x_aug,
            W=z_aug,
            g_yh=g_yh,
            theta0=theta0,
            phi_map=Phi_aug,
            d_h=1,
            instrument_basis=Q_aug,
            whiten_basis=bool(whiten_basis),
            max_moment_ratio=0.6,
            reg_lambda=0.0,
            init_theta=None,
            outer_use_analytic_grad=False,
            per_v_gmm_start=False,
            random_state=int(rng.integers(0, 2**32 - 1)),
            newton_tol=1e-6,
        )

        out = betel.fit_per_v(v=v_aug, method=outer_method, options=outer_options, theta0=theta0)
        inner = out.get("inner_info", {}) or {}
        if not inner.get("feasible", True):
            continue
        th = np.asarray(out["theta"], float).reshape(-1)
        if not np.all(np.isfinite(th)):
            continue
        preds_tr.append(Phi_tr @ th)
        preds_te.append(Phi_te @ th)

    if len(preds_tr) == 0:
        h_tr = Phi_tr @ theta0
        h_te = Phi_te @ theta0
    else:
        h_tr = _pred_mean(np.vstack(preds_tr))
        h_te = _pred_mean(np.vstack(preds_te))

    train_mse = float(np.mean((h_tr - h0_eval(x)) ** 2))
    test_mse = float(np.mean((h_te - h0_eval(x_test)) ** 2))
    return train_mse, test_mse


# -----------------------------------------------------------------------------
# Monte Carlo evaluation
# -----------------------------------------------------------------------------


def monte_carlo_table_from_bundle(
    bundle: Dict[Any, Any],
    *,
    df_x: int,
    df_z: int,
    betel_B: int,
    m_ratio: float,
    alpha_fracs: Iterable[float],
    npiv_B1: float,
    npiv_penalty_grid: int,
    sample_with_replacement: bool,
    common_subset_only: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    alpha_fracs = [float(v) for v in alpha_fracs]
    meta = bundle["_meta"]
    h0_eval = lambda x: h0_from_meta(np.asarray(x, float), meta)

    rows: List[Dict[str, object]] = []

    for N in sorted(k for k in bundle.keys() if k != "_meta"):
        all_records = bundle[N]
        if common_subset_only:
            records = [rec for rec in all_records if record_has_usable_gpt(rec)]
        else:
            records = list(all_records)

        print(f"[N={N}] total reps = {len(all_records)}, used reps = {len(records)}")

        npiv_train: List[float] = []
        npiv_test: List[float] = []
        sb0_train: List[float] = []
        sb0_test: List[float] = []
        sbg_train: Dict[float, List[float]] = {af: [] for af in alpha_fracs}
        sbg_test: Dict[float, List[float]] = {af: [] for af in alpha_fracs}

        for rec in tqdm(records, desc=f"Engel95 MC N={N}"):
            y = np.asarray(rec["y_train"], float)
            x = np.asarray(rec["x_train"], float)
            z = np.asarray(rec["z_train"], float)
            x_te = np.asarray(rec["x_test"], float)

            tr_npiv, te_npiv = penalized_spline_npiv_predictive_mse(
                y, x, z, x_te,
                df_x=int(df_x),
                df_z=int(df_z),
                B1=float(npiv_B1),
                h0_eval=h0_eval,
                penalty_grid=int(npiv_penalty_grid),
            )
            npiv_train.append(tr_npiv)
            npiv_test.append(te_npiv)

            tr0, te0 = structural_betel_predictive_mse_alpha0(
                y, x, z, x_te,
                B_boot=int(betel_B),
                random_state=int(rec["train_seed"]) + 991,
                df_x=int(df_x),
                df_z=int(df_z),
                whiten_basis=False,
                h0_eval=h0_eval,
            )
            sb0_train.append(tr0)
            sb0_test.append(te0)

            yalts = extract_gpt_yalt_matrix(rec)
            if yalts is None or not np.isfinite(yalts).any():
                continue
            for af in alpha_fracs:
                alpha_val = float(af) * float(N)
                tr_g, te_g = structural_betel_predictive_mse_gpt_condy_aug(
                    y, x, z, x_te, yalts,
                    B_boot=int(betel_B),
                    random_state=int(rec["train_seed"]) + 20000 + int(1000 * af),
                    df_x=int(df_x),
                    df_z=int(df_z),
                    whiten_basis=False,
                    alpha=float(alpha_val),
                    m_ratio=float(m_ratio),
                    h0_eval=h0_eval,
                    sample_with_replacement=bool(sample_with_replacement),
                )
                sbg_train[af].append(tr_g)
                sbg_test[af].append(te_g)

        baseline_name = f"PenalizedSplineNPIV(df_x={int(df_x)},df_z={int(df_z)},B1={float(npiv_B1):g})"
        rows.append({"N": N, "method": baseline_name, "split": "train", "RMSE": math.sqrt(np.mean(npiv_train))})
        rows.append({"N": N, "method": baseline_name, "split": "test", "RMSE": math.sqrt(np.mean(npiv_test))})
        rows.append({"N": N, "method": "SBETEL(alpha=0)", "split": "train", "RMSE": math.sqrt(np.mean(sb0_train))})
        rows.append({"N": N, "method": "SBETEL(alpha=0)", "split": "test", "RMSE": math.sqrt(np.mean(sb0_test))})

        for af in alpha_fracs:
            if len(sbg_train[af]) == 0:
                continue
            rows.append({"N": N, "method": f"GPT-SBETEL(alpha={af:.2f}N)", "split": "train", "RMSE": math.sqrt(np.mean(sbg_train[af]))})
            rows.append({"N": N, "method": f"GPT-SBETEL(alpha={af:.2f}N)", "split": "test", "RMSE": math.sqrt(np.mean(sbg_test[af]))})

    df_long = pd.DataFrame(rows)
    df_wide = df_long.pivot(index="N", columns=["method", "split"], values="RMSE").sort_index()
    return df_long, df_wide


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Evaluate penalized spline NPIV vs SBETEL vs GPT-SBETEL on Engel95 Monte Carlo bundle")
    ap.add_argument("--df_x", type=int, default=3)
    ap.add_argument("--df_z", type=int, default=4)
    ap.add_argument("--betel_B", type=int, default=100)
    ap.add_argument("--m_ratio", type=float, default=0.5)
    ap.add_argument("--alpha_fracs", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20, 0.30, 0.50])
    ap.add_argument("--npiv_B1", type=float, default=5.0, help="Roughness-budget parameter B1 for the penalized spline NPIV baseline")
    ap.add_argument("--npiv_penalty_grid", type=int, default=100, help="Grid size used to approximate the spline roughness penalty matrix")
    ap.add_argument("--sample_with_replacement", action="store_true")
    ap.add_argument("--common_subset_only", action="store_true", help="If set, evaluate all methods only on replications with usable GPT labels.")
    ap.add_argument("--save_csv", type=str, default="engel95_npiv_vs_gpt_sbetel_100.csv")
    ap.add_argument("--save_long_csv", type=str, default="engel95_npiv_vs_gpt_sbetel_long_100.csv")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    here = pathlib.Path(__file__).resolve().parent
    aug_data_dir = here / "output_engel95_100" / "engel95_gpt_training_bundle_complete.pkl"
    with open(aug_data_dir, "rb") as f:
        bundle = pickle.load(f)

    df_long, df_wide = monte_carlo_table_from_bundle(
        bundle,
        df_x=int(args.df_x),
        df_z=int(args.df_z),
        betel_B=int(args.betel_B),
        m_ratio=float(args.m_ratio),
        alpha_fracs=[float(v) for v in args.alpha_fracs],
        npiv_B1=float(args.npiv_B1),
        npiv_penalty_grid=int(args.npiv_penalty_grid),
        sample_with_replacement=bool(args.sample_with_replacement),
        common_subset_only=bool(args.common_subset_only),
    )

    print(df_wide)
    df_wide.to_csv(args.save_csv)
    df_long.to_csv(args.save_long_csv, index=False)
    print(f"\nSaved wide CSV to: {args.save_csv}")
    print(f"Saved long CSV to: {args.save_long_csv}")
