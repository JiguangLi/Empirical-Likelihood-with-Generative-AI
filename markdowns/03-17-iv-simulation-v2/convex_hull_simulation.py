#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-replication convex-hull / regularization diagnostics for the Engel95 experiment.

Purpose
-------
This script does NOT run the full Monte Carlo study. Instead, it loads one training
replication from the GPT-augmented Engel95 bundle and saves detailed draw-level outputs
for:

  1. Penalized spline NPIV baseline,
  2. SBETEL(alpha=0),
  3. GPT-SBETEL(alpha>0) for one or more prior strengths.

The saved pickle is designed for later plotting. In particular it contains:
  - true structural curve h0 on a common x-grid,
  - NPIV fitted curve,
  - draw-level BETEL curves,
  - draw-level final ETEL masses p_i = v_i * w_i,
  - residual norms / feasibility flags,
  - effective sample size and mass concentration summaries,
  - for augmented runs: synthetic-row mass shares.

Interpretation of weights
-------------------------
GenELV2.fit_per_v(...) returns "weights" = w_i = p_i / v_i. The final ETEL-implied
probability masses are therefore

    p_i = v_i * w_i,

"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import math
import pathlib
import pickle
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from patsy import dmatrix, build_design_matrices
from scipy.linalg import qr
from scipy.stats import norm

from feature_map import NSFeatureMap


# -----------------------------------------------------------------------------
# Dynamic local loader
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
    pkg.GenELV2 = mod_gen.GenELV2
    pkg.StructuralBETELV2 = mod_struct.StructuralBETELV2
    return pkg


EL = load_local_el()


# -----------------------------------------------------------------------------
# Shared helpers copied from the current simulation script
# -----------------------------------------------------------------------------


def make_structural_betel(**kwargs):
    sig = inspect.signature(EL.StructuralBETELV2.__init__)
    keep = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return EL.StructuralBETELV2(**keep)



def h0_from_meta(x: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    pars = meta["calibration"]["h0_params"]
    upper = float(pars["upper"])
    span = float(pars["span"])
    center = float(pars["center"])
    scale = float(pars["scale"])
    x = np.asarray(x, float)
    return upper - span * norm.cdf((x - center) / scale)



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



def _fit_cr_design(z: np.ndarray, *, df: int):
    dm = dmatrix(f"0 + cr(z, df={int(df)})", {"z": np.asarray(z, float).reshape(-1)})
    B = np.asarray(dm, float)
    return dm.design_info, B



def _cr_design_transform(design_info: Any, z_new: np.ndarray) -> np.ndarray:
    mat = build_design_matrices([design_info], {"z": np.asarray(z_new, float).reshape(-1)})[0]
    return np.asarray(mat, float)



def compute_spline_roughness_penalty(
    design_info: Any,
    x_min: float,
    x_max: float,
    *,
    n_grid: int = 400,
    normalize: bool = True,
) -> np.ndarray:
    x_min = float(x_min)
    x_max = float(x_max)
    if x_max <= x_min:
        x_max = x_min + 1.0

    n_grid = max(int(n_grid), 25)
    grid = np.linspace(x_min, x_max, n_grid)
    B = _cr_design_transform(design_info, grid)
    dx = float(grid[1] - grid[0])

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
):
    y = np.asarray(y, float).reshape(-1)
    x = np.asarray(x, float).reshape(-1)
    z = np.asarray(z, float).reshape(-1)

    x_info, Phi = _fit_cr_design(x, df=int(df_x))
    Lambda = compute_spline_roughness_penalty(
        x_info, float(np.min(x)), float(np.max(x)), n_grid=int(penalty_grid), normalize=True
    )

    z_info, Bz_nc = _fit_cr_design(z, df=int(df_z))
    Q = np.column_stack([np.ones_like(z), Bz_nc])
    QQinv = np.linalg.pinv(Q.T @ Q)
    Phi_hat = Q @ (QQinv @ (Q.T @ Phi))

    beta_hat = solve_constrained_ls_general(y, Phi_hat, Lambda, float(B1))
    return beta_hat, x_info



def extract_gpt_yalt_matrix(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    if "gpt_yalt_matrix" in rec and rec["gpt_yalt_matrix"] is not None:
        mat = np.asarray(rec["gpt_yalt_matrix"], float)
        if mat.ndim == 2:
            return mat

    df = rec.get("gpt_conditional_df", None)
    if df is None:
        return None
    synth_cols = [c for c in list(df.keys()) if str(c).startswith("synthetic_food_share")]
    if len(synth_cols) == 0:
        synth_cols = [c for c in list(df.keys()) if str(c).startswith("synthetic_y")]
    if len(synth_cols) == 0:
        return None
    return np.asarray(np.column_stack([np.asarray(df[c], float) for c in synth_cols]), float)



def _diag_from_p(p: np.ndarray) -> Dict[str, float]:
    p = np.asarray(p, float).reshape(-1)
    p = np.clip(p, 0.0, None)
    s = float(np.sum(p))
    if s <= 0.0:
        return {
            "ess": np.nan,
            "max_mass": np.nan,
            "entropy": np.nan,
            "n90": np.nan,
        }
    p = p / s
    ess = 1.0 / float(np.sum(p ** 2))
    max_mass = float(np.max(p))
    entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    ps = np.sort(p)[::-1]
    n90 = int(np.searchsorted(np.cumsum(ps), 0.90) + 1)
    return {
        "ess": ess,
        "max_mass": max_mass,
        "entropy": entropy,
        "n90": float(n90),
    }



def _posterior_mean_or_theta0(curves: np.ndarray, theta0_curve: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    feasible = np.asarray(feasible, bool)
    if feasible.any():
        return np.nanmean(curves[feasible], axis=0)
    return np.asarray(theta0_curve, float)


# -----------------------------------------------------------------------------
# Core diagnostics
# -----------------------------------------------------------------------------


def run_alpha0_diagnostics(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    x_test: np.ndarray,
    x_grid: np.ndarray,
    *,
    B_boot: int,
    random_state: int,
    df_x: int,
    df_z: int,
    whiten_basis: bool,
    outer_method: str,
    outer_options: Dict[str, Any],
) -> Dict[str, Any]:
    y = np.asarray(y, float).reshape(-1)
    x = np.asarray(x, float).reshape(-1)
    z = np.asarray(z, float).reshape(-1)
    x_test = np.asarray(x_test, float).reshape(-1)
    x_grid = np.asarray(x_grid, float).reshape(-1)
    N = int(len(y))

    phi = NSFeatureMap(df_first=int(df_x), df_rest=int(df_x), tilde=True, center=True, scale=True, add_constant=True)
    Phi_tr = phi.fit_transform(x[:, None])
    Phi_te = phi.transform(x_test[:, None])
    Phi_grid = phi.transform(x_grid[:, None])

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

    theta0_train_curve = Phi_tr @ theta0
    theta0_test_curve = Phi_te @ theta0
    theta0_grid_curve = Phi_grid @ theta0

    thetas = np.full((B_boot, theta0.size), np.nan, dtype=float)
    feasible = np.zeros(B_boot, dtype=bool)
    residual_norm = np.full(B_boot, np.nan, dtype=float)
    losses = np.full(B_boot, np.nan, dtype=float)
    curves_train = np.full((B_boot, N), np.nan, dtype=float)
    curves_test = np.full((B_boot, len(x_test)), np.nan, dtype=float)
    curves_grid = np.full((B_boot, len(x_grid)), np.nan, dtype=float)
    v_draws = np.full((B_boot, N), np.nan, dtype=float)
    w_draws = np.full((B_boot, N), np.nan, dtype=float)
    p_draws = np.full((B_boot, N), np.nan, dtype=float)
    p_sorted = np.full((B_boot, N), np.nan, dtype=float)
    ess = np.full(B_boot, np.nan, dtype=float)
    max_mass = np.full(B_boot, np.nan, dtype=float)
    entropy = np.full(B_boot, np.nan, dtype=float)
    n90 = np.full(B_boot, np.nan, dtype=float)

    for b in range(int(B_boot)):
        if b % 10 ==0:
            print(b)
        v = betel.draw_dirichlet()
        out = betel.fit_per_v(v=v, method=outer_method, options=outer_options, theta0=theta0)
        inner = out.get("inner_info", {}) or {}
        th = np.asarray(out["theta"], float).reshape(-1)
        w = np.asarray(out["weights"], float).reshape(-1)
        p = np.asarray(v * w, float).reshape(-1)

        thetas[b] = th
        v_draws[b] = v
        w_draws[b] = w
        p_draws[b] = p
        p_sorted[b] = np.sort(p)[::-1]
        losses[b] = float(out.get("loss", np.nan))
        residual_norm[b] = float(inner.get("residual_norm", np.nan))
        feasible[b] = bool(inner.get("feasible", True)) and np.all(np.isfinite(th))

        dd = _diag_from_p(p)
        ess[b] = dd["ess"]
        max_mass[b] = dd["max_mass"]
        entropy[b] = dd["entropy"]
        n90[b] = dd["n90"]

        if np.all(np.isfinite(th)):
            curves_train[b] = Phi_tr @ th
            curves_test[b] = Phi_te @ th
            curves_grid[b] = Phi_grid @ th

    used_grid_curve = _posterior_mean_or_theta0(curves_grid, theta0_grid_curve, feasible)
    used_train_curve = _posterior_mean_or_theta0(curves_train, theta0_train_curve, feasible)
    used_test_curve = _posterior_mean_or_theta0(curves_test, theta0_test_curve, feasible)

    return {
        "label": "SBETEL(alpha=0)",
        "alpha": 0.0,
        "N_obs": N,
        "N_syn": 0,
        "theta0": theta0,
        "theta0_curve_grid": theta0_grid_curve,
        "theta0_curve_train": theta0_train_curve,
        "theta0_curve_test": theta0_test_curve,
        "theta_draws": thetas,
        "feasible": feasible,
        "residual_norm": residual_norm,
        "loss": losses,
        "curve_train_draws": curves_train,
        "curve_test_draws": curves_test,
        "curve_grid_draws": curves_grid,
        "curve_grid_mean_used": used_grid_curve,
        "curve_train_mean_used": used_train_curve,
        "curve_test_mean_used": used_test_curve,
        "v_draws": v_draws,
        "w_draws": w_draws,
        "p_draws": p_draws,
        "p_sorted_draws": p_sorted,
        "ess": ess,
        "max_mass": max_mass,
        "entropy": entropy,
        "n90": n90,
    }



def run_gpt_aug_diagnostics(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    x_test: np.ndarray,
    x_grid: np.ndarray,
    gpt_yalt_matrix: np.ndarray,
    *,
    alpha_frac: float,
    B_boot: int,
    random_state: int,
    df_x: int,
    df_z: int,
    whiten_basis: bool,
    m_ratio: float,
    sample_with_replacement: bool,
    outer_method: str,
    outer_options: Dict[str, Any],
) -> Dict[str, Any]:
    y = np.asarray(y, float).reshape(-1)
    x = np.asarray(x, float).reshape(-1)
    z = np.asarray(z, float).reshape(-1)
    x_test = np.asarray(x_test, float).reshape(-1)
    x_grid = np.asarray(x_grid, float).reshape(-1)
    yalts = np.asarray(gpt_yalt_matrix, float)
    N = int(len(y))
    m = int(max(1, round(float(m_ratio) * N)))
    alpha = float(alpha_frac) * float(N)

    valid_rows = np.where(np.any(np.isfinite(yalts), axis=1))[0]
    if len(valid_rows) == 0:
        raise ValueError("No usable GPT synthetic labels in selected replication.")

    phi = NSFeatureMap(df_first=int(df_x), df_rest=int(df_x), tilde=True, center=True, scale=True, add_constant=True)
    Phi_tr = phi.fit_transform(x[:, None])
    Phi_te = phi.transform(x_test[:, None])
    Phi_grid = phi.transform(x_grid[:, None])

    z_info, Bz_obs_nc = _fit_cr_design(z, df=int(df_z))
    Q_obs = np.column_stack([np.ones_like(z), Bz_obs_nc])
    theta0 = theta_init_2sls(Phi_tr, y, Q_obs, ridge=1e-6)

    def g_yh(Y, H):
        return np.asarray(Y, float).reshape(-1, 1) - np.asarray(H, float).reshape(-1, 1)

    theta0_train_curve = Phi_tr @ theta0
    theta0_test_curve = Phi_te @ theta0
    theta0_grid_curve = Phi_grid @ theta0

    conc_syn = alpha / float(m)
    conc = np.concatenate([np.ones(N, float), np.full(m, conc_syn, float)])
    rng = np.random.default_rng(int(random_state))

    thetas = np.full((B_boot, theta0.size), np.nan, dtype=float)
    feasible = np.zeros(B_boot, dtype=bool)
    residual_norm = np.full(B_boot, np.nan, dtype=float)
    losses = np.full(B_boot, np.nan, dtype=float)
    curves_train = np.full((B_boot, N), np.nan, dtype=float)
    curves_test = np.full((B_boot, len(x_test)), np.nan, dtype=float)
    curves_grid = np.full((B_boot, len(x_grid)), np.nan, dtype=float)
    v_draws = np.full((B_boot, N + m), np.nan, dtype=float)
    w_draws = np.full((B_boot, N + m), np.nan, dtype=float)
    p_draws = np.full((B_boot, N + m), np.nan, dtype=float)
    p_sorted = np.full((B_boot, N + m), np.nan, dtype=float)
    ess = np.full(B_boot, np.nan, dtype=float)
    max_mass = np.full(B_boot, np.nan, dtype=float)
    entropy = np.full(B_boot, np.nan, dtype=float)
    n90 = np.full(B_boot, np.nan, dtype=float)
    syn_mass = np.full(B_boot, np.nan, dtype=float)
    obs_mass = np.full(B_boot, np.nan, dtype=float)
    row_idx_draws = np.full((B_boot, m), -1, dtype=int)
    alt_idx_draws = np.full((B_boot, m), -1, dtype=int)
    y_syn_draws = np.full((B_boot, m), np.nan, dtype=float)

    for b in range(int(B_boot)):
        if b % 10 ==0:
            print(b)
        replace_rows = bool(sample_with_replacement or (m > len(valid_rows)))
        row_idx = rng.choice(valid_rows, size=m, replace=replace_rows)
        x_s = x[row_idx].copy()
        z_s = z[row_idx].copy()
        y_s = np.empty(m, dtype=float)
        alt_idx = np.empty(m, dtype=int)

        for j, i in enumerate(row_idx):
            good = np.where(np.isfinite(yalts[i]))[0]
            if len(good) == 0:
                alt_idx[j] = -1
                y_s[j] = float(np.clip(y[i], 0.0, 1.0))
            else:
                k = int(rng.choice(good))
                alt_idx[j] = k
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
        th = np.asarray(out["theta"], float).reshape(-1)
        w_aug = np.asarray(out["weights"], float).reshape(-1)
        p_aug = np.asarray(v_aug * w_aug, float).reshape(-1)

        thetas[b] = th
        v_draws[b] = v_aug
        w_draws[b] = w_aug
        p_draws[b] = p_aug
        p_sorted[b] = np.sort(p_aug)[::-1]
        losses[b] = float(out.get("loss", np.nan))
        residual_norm[b] = float(inner.get("residual_norm", np.nan))
        feasible[b] = bool(inner.get("feasible", True)) and np.all(np.isfinite(th))
        row_idx_draws[b] = row_idx
        alt_idx_draws[b] = alt_idx
        y_syn_draws[b] = y_s

        dd = _diag_from_p(p_aug)
        ess[b] = dd["ess"]
        max_mass[b] = dd["max_mass"]
        entropy[b] = dd["entropy"]
        n90[b] = dd["n90"]
        obs_mass[b] = float(np.sum(p_aug[:N]))
        syn_mass[b] = float(np.sum(p_aug[N:]))

        if np.all(np.isfinite(th)):
            curves_train[b] = Phi_tr @ th
            curves_test[b] = Phi_te @ th
            curves_grid[b] = Phi_grid @ th

    used_grid_curve = _posterior_mean_or_theta0(curves_grid, theta0_grid_curve, feasible)
    used_train_curve = _posterior_mean_or_theta0(curves_train, theta0_train_curve, feasible)
    used_test_curve = _posterior_mean_or_theta0(curves_test, theta0_test_curve, feasible)

    return {
        "label": f"GPT-SBETEL(alpha={alpha_frac:.2f}N)",
        "alpha_frac": float(alpha_frac),
        "alpha": float(alpha),
        "N_obs": N,
        "N_syn": m,
        "theta0": theta0,
        "theta0_curve_grid": theta0_grid_curve,
        "theta0_curve_train": theta0_train_curve,
        "theta0_curve_test": theta0_test_curve,
        "theta_draws": thetas,
        "feasible": feasible,
        "residual_norm": residual_norm,
        "loss": losses,
        "curve_train_draws": curves_train,
        "curve_test_draws": curves_test,
        "curve_grid_draws": curves_grid,
        "curve_grid_mean_used": used_grid_curve,
        "curve_train_mean_used": used_train_curve,
        "curve_test_mean_used": used_test_curve,
        "v_draws": v_draws,
        "w_draws": w_draws,
        "p_draws": p_draws,
        "p_sorted_draws": p_sorted,
        "ess": ess,
        "max_mass": max_mass,
        "entropy": entropy,
        "n90": n90,
        "obs_mass": obs_mass,
        "syn_mass": syn_mass,
        "row_idx_draws": row_idx_draws,
        "alt_idx_draws": alt_idx_draws,
        "y_syn_draws": y_syn_draws,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Single-replication convex-hull / regularization diagnostics for Engel95 GPT-BETEL")
    ap.add_argument("--bundle", type=str, default="output_engel95_100/engel95_gpt_training_bundle_complete.pkl")
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--rep", type=int, default=92, help= "which gpt synthetic record")
    ap.add_argument("--df_x", type=int, default=3)
    ap.add_argument("--df_z", type=int, default=4)
    ap.add_argument("--betel_B", type=int, default=100)
    ap.add_argument("--m_ratio", type=float, default=0.5)
    ap.add_argument("--alpha_fracs", type=float, nargs="+", default=[0.10, 0.50])
    ap.add_argument("--npiv_B1", type=float, default=5.0)
    ap.add_argument("--npiv_penalty_grid", type=int, default=400)
    ap.add_argument("--sample_with_replacement", action="store_true")
    ap.add_argument("--whiten_basis", action="store_true", help="Default is False to match the current fast/stable setup.")
    ap.add_argument("--outer_maxiter", type=int, default=50)
    ap.add_argument("--outer_ftol", type=float, default=1e-6)
    ap.add_argument("--outer_eps", type=float, default=1e-6)
    ap.add_argument("--grid_n", type=int, default=200)
    ap.add_argument("--save", type=str, default="convex_hull_diagnostics.pkl")
    return ap



def main() -> None:
    args = build_parser().parse_args()

    bundle_path = Path(args.bundle).resolve()
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    meta = bundle["_meta"]
    records = bundle[int(args.N)]
    if not (0 <= int(args.rep) < len(records)):
        raise IndexError(f"rep={args.rep} out of range for N={args.N}; available reps = {len(records)}")
    rec = records[int(args.rep)]

    y = np.asarray(rec["y_train"], float)
    x = np.asarray(rec["x_train"], float)
    z = np.asarray(rec["z_train"], float)
    y_test = np.asarray(rec["y_test"], float)
    x_test = np.asarray(rec["x_test"], float)
    z_test = np.asarray(rec["z_test"], float)
    gpt_mat = extract_gpt_yalt_matrix(rec)
    if gpt_mat is None:
        raise ValueError("Selected replication does not contain usable GPT synthetic labels.")

    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), int(args.grid_n))
    h0_grid = h0_from_meta(x_grid, meta)
    h0_train = h0_from_meta(x, meta)
    h0_test = h0_from_meta(x_test, meta)

    beta_npiv, x_info = penalized_spline_npiv_fit(
        y, x, z,
        df_x=int(args.df_x),
        df_z=int(args.df_z),
        B1=float(args.npiv_B1),
        penalty_grid=int(args.npiv_penalty_grid),
    )
    Phi_tr_npiv = _cr_design_transform(x_info, x)
    Phi_te_npiv = _cr_design_transform(x_info, x_test)
    Phi_grid_npiv = _cr_design_transform(x_info, x_grid)
    h_npiv_train = Phi_tr_npiv @ beta_npiv
    h_npiv_test = Phi_te_npiv @ beta_npiv
    h_npiv_grid = Phi_grid_npiv @ beta_npiv

    outer_options = {
        "maxiter": int(args.outer_maxiter),
        "ftol": float(args.outer_ftol),
        "eps": float(args.outer_eps),
    }

    alpha0 = run_alpha0_diagnostics(
        y, x, z, x_test, x_grid,
        B_boot=int(args.betel_B),
        random_state=int(rec["train_seed"]) + 991,
        df_x=int(args.df_x),
        df_z=int(args.df_z),
        whiten_basis=bool(args.whiten_basis),
        outer_method="L-BFGS-B",
        outer_options=outer_options,
    )

    alpha_runs = {}
    for af in [float(v) for v in args.alpha_fracs]:
        print(af)
        alpha_runs[f"{af:.4f}"] = run_gpt_aug_diagnostics(
            y, x, z, x_test, x_grid, gpt_mat,
            alpha_frac=float(af),
            B_boot=int(args.betel_B),
            random_state=int(rec["train_seed"]) + 20000 + int(1000 * af),
            df_x=int(args.df_x),
            df_z=int(args.df_z),
            whiten_basis=bool(args.whiten_basis),
            m_ratio=float(args.m_ratio),
            sample_with_replacement=bool(args.sample_with_replacement),
            outer_method="L-BFGS-B",
            outer_options=outer_options,
        )

    out = {
        "meta": {
            "bundle_path": str(bundle_path),
            "N": int(args.N),
            "rep": int(args.rep),
            "train_seed": int(rec["train_seed"]),
            "test_seed": int(rec["test_seed"]),
            "df_x": int(args.df_x),
            "df_z": int(args.df_z),
            "betel_B": int(args.betel_B),
            "m_ratio": float(args.m_ratio),
            "alpha_fracs": [float(v) for v in args.alpha_fracs],
            "npiv_B1": float(args.npiv_B1),
            "npiv_penalty_grid": int(args.npiv_penalty_grid),
            "whiten_basis": bool(args.whiten_basis),
            "outer_options": outer_options,
        },
        "data": {
            "x_train": x,
            "z_train": z,
            "y_train": y,
            "x_test": x_test,
            "z_test": z_test,
            "y_test": y_test,
            "gpt_yalt_matrix": gpt_mat,
        },
        "target": {
            "x_grid": x_grid,
            "h0_grid": h0_grid,
            "h0_train": h0_train,
            "h0_test": h0_test,
        },
        "npiv": {
            "label": f"PenalizedSplineNPIV(df_x={int(args.df_x)},df_z={int(args.df_z)},B1={float(args.npiv_B1):g})",
            "beta_hat": beta_npiv,
            "curve_grid": h_npiv_grid,
            "curve_train": h_npiv_train,
            "curve_test": h_npiv_test,
            "train_mse": float(np.mean((h_npiv_train - h0_train) ** 2)),
            "test_mse": float(np.mean((h_npiv_test - h0_test) ** 2)),
        },
        "alpha0": alpha0,
        "alpha_pos": alpha_runs,
    }

    save_path = Path(args.save).resolve()
    with open(save_path, "wb") as f:
        pickle.dump(out, f)

    print(f"Saved diagnostics pickle to: {save_path}")


if __name__ == "__main__":
    main()
