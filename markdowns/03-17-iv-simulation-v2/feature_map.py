from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

try:
    from patsy import dmatrix, build_design_matrices
except Exception as e:
    raise ImportError("This code requires patsy. Install with: pip install patsy") from e


@dataclass
class NSFeatureMap:
    """
    Natural-cubic-spline per-coordinate feature map for X.
    This is a *fitted* transformer:
      - fit(X): learns spline 'design_info' (i.e., knots) from training X and stores
                column-wise mean/std (if center/scale=True)
      - transform(X): applies *the same* design_info and mean/std to new X

    Parameters
    ----------
    df_first : int
        df for the first coordinate.
    df_rest : int
        df for each subsequent coordinate (before tilde).
    tilde : bool
        If True, for j>=1 use Bj[:,1:] - Bj[:,[0]] (paper-style contrasts).
    center, scale : bool
        Center and/or scale the non-constant columns (fit-time stats are reused at transform-time).
    add_constant : bool
        If True, prepend a column of ones as the first column.
    """

    df_first: int = 7
    df_rest: int = 5
    tilde: bool = True
    center: bool = True
    scale: bool = True
    add_constant: bool = True

    # learned state after fit
    design_infos_: Optional[List] = None  # one per coordinate
    mu_: Optional[np.ndarray] = None      # (1, p_no_const)
    sd_: Optional[np.ndarray] = None      # (1, p_no_const)
    p_: int = 0                           # total feature count after (optional) constant

    def fit(self, X: np.ndarray) -> "NSFeatureMap":
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        N, d = X.shape

        parts: List[np.ndarray] = []
        infos: List = []

        # j = 0 (first coordinate): keep full df_first columns
        dm0 = dmatrix(f"0 + cr(x, df={self.df_first})", {"x": X[:, 0]})
        B1 = np.asarray(dm0, float)
        parts.append(B1)
        infos.append(dm0.design_info)

        # j >= 1
        for j in range(1, d):
            dmj = dmatrix(f"0 + cr(x, df={self.df_rest})", {"x": X[:, j]})
            Bj = np.asarray(dmj, float)
            if self.tilde:
                if Bj.shape[1] < 2:
                    raise RuntimeError(f"patsy cr produced <2 columns for j={j}.")
                parts.append(Bj[:, 1:] - Bj[:, [0]])
            else:
                parts.append(Bj)
            infos.append(dmj.design_info)

        Phi_no_const = np.concatenate(parts, axis=1) if parts else np.empty((N, 0))

        # center/scale statistics learned on training
        Z = Phi_no_const.copy()
        if self.center:
            mu = Z.mean(axis=0, keepdims=True)
        else:
            mu = np.zeros((1, Z.shape[1]), dtype=float)
        if self.scale:
            sd = Z.std(axis=0, ddof=1, keepdims=True)
            sd[sd < 1e-12] = 1.0
        else:
            sd = np.ones((1, Z.shape[1]), dtype=float)

        # transformed training Φ (not strictly needed to return here)
        Z = (Z - mu) / sd
        Phi_train = np.hstack([np.ones((N, 1)) , Z]) if self.add_constant else Z

        # store learned state
        self.design_infos_ = infos
        self.mu_ = mu
        self.sd_ = sd
        self.p_ = Phi_train.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the *fitted* spline+standardization to new X."""
        if self.design_infos_ is None or self.mu_ is None or self.sd_ is None:
            raise RuntimeError("NSFeatureMap is not fitted. Call fit(X) first.")

        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        N, d = X.shape
        if len(self.design_infos_) != d:
            raise RuntimeError(f"Transformer expects X with {len(self.design_infos_)} columns, got {d}.")

        parts: List[np.ndarray] = []

        # j = 0
        B1 = np.asarray(build_design_matrices([self.design_infos_[0]], {"x": X[:, 0]})[0], float)
        parts.append(B1)

        # j >= 1
        for j in range(1, d):
            Bj = np.asarray(build_design_matrices([self.design_infos_[j]], {"x": X[:, j]})[0], float)
            if self.tilde:
                parts.append(Bj[:, 1:] - Bj[:, [0]])
            else:
                parts.append(Bj)

        Phi_no_const = np.concatenate(parts, axis=1) if parts else np.empty((N, 0))
        Z = (Phi_no_const - self.mu_) / self.sd_
        Phi = np.hstack([np.ones((N, 1)), Z]) if self.add_constant else Z
        return np.ascontiguousarray(Phi)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
