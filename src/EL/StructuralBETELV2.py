from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union, List
import numpy as np
from patsy import dmatrix
from scipy.linalg import svd

from .GenELV2 import GenELV2


# =============================================================================
# Helpers
# =============================================================================

def _as_2d(a: np.ndarray, *, name: str) -> np.ndarray:
    a = np.asarray(a, float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D; got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN/Inf.")
    return a


def _detect_constant_column(B: np.ndarray, tol: float = 1e-12) -> Optional[int]:
    """Return index of a (near-)constant column if present, else None."""
    if B.ndim != 2 or B.shape[1] == 0:
        return None
    std = B.std(axis=0)
    idx = np.where(std < tol)[0]
    return int(idx[0]) if idx.size else None


def make_basis_ns_patsy_paper(
    W: np.ndarray,
    *,
    df_first: int = 3,
    df_rest: int = 3,
    center: bool = True,
    scale: bool = True,
    add_constant: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Paper-style natural cubic regression spline basis on W.

    - For j=0: cr(w0, df=df_first) (all columns)
    - For j>=1: tilde contrast Bj[:,1:] - Bj[:,[0]] with df_rest columns in cr(...)
    - Optional centering/scaling applied to the non-constant columns only.
    - Optional constant/intercept column prepended as the first column.

    Returns
    -------
    B : (N, K) array
    meta : dict
    """
    W = np.asarray(W, float)
    if W.ndim == 1:
        W = W[:, None]
    if not np.isfinite(W).all():
        raise ValueError("W contains NaN/Inf.")
    N, d = W.shape
    if df_first < 2 or df_rest < 2:
        raise ValueError("df_first and df_rest must be >= 2.")

    parts: List[np.ndarray] = []
    const = np.ones((N, 1), dtype=float) if add_constant else None

    # j = 0
    B0 = np.asarray(dmatrix(f"0 + cr(x, df={int(df_first)})", {"x": W[:, 0]}), float)
    parts.append(B0)

    # j >= 1 (tilde)
    for j in range(1, d):
        Bj = np.asarray(dmatrix(f"0 + cr(x, df={int(df_rest)})", {"x": W[:, j]}), float)
        if Bj.shape[1] < 2:
            raise RuntimeError(f"patsy cr produced <2 columns for j={j}.")
        parts.append(Bj[:, 1:] - Bj[:, [0]])

    B_noconst = np.concatenate(parts, axis=1) if parts else np.empty((N, 0), dtype=float)

    if center or scale:
        Bc = B_noconst.copy()
        if center:
            Bc -= Bc.mean(axis=0, keepdims=True)
        if scale:
            s = Bc.std(axis=0, ddof=1, keepdims=True)
            s[s < 1e-12] = 1.0
            Bc /= s
        B_noconst = Bc

    B = np.hstack([const, B_noconst]) if add_constant else B_noconst
    meta = {
        "basis_mode": "patsy_cr_paper_tilde",
        "df_first": int(df_first),
        "df_rest": int(df_rest),
        "d_w": int(d),
        "center": bool(center),
        "scale": bool(scale),
        "add_constant": bool(add_constant),
        "K_raw": int(B.shape[1]),
    }
    return np.ascontiguousarray(B), meta


def make_basis_cr_df_1d(
    w: np.ndarray,
    *,
    df: int = 5,
    add_constant: bool = True,
    center: bool = True,
    scale: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience: 1D cubic regression spline basis with df.

    Uses patsy: 0 + cr(w, df=df). Optionally adds intercept as first col.
    """
    w = np.asarray(w, float).reshape(-1)
    B0 = np.asarray(dmatrix(f"0 + cr(z, df={int(df)})", {"z": w}), float)
    if center or scale:
        Bc = B0.copy()
        if center:
            Bc -= Bc.mean(axis=0, keepdims=True)
        if scale:
            s = Bc.std(axis=0, ddof=1, keepdims=True)
            s[s < 1e-12] = 1.0
            Bc /= s
        B0 = Bc
    if add_constant:
        B = np.column_stack([np.ones_like(w), B0])
    else:
        B = B0
    meta = {
        "basis_mode": "cr_df_1d",
        "df": int(df),
        "add_constant": bool(add_constant),
        "center": bool(center),
        "scale": bool(scale),
        "K_raw": int(B.shape[1]),
    }
    return np.ascontiguousarray(B), meta


def _whiten_and_prune_basis(
    B_raw: np.ndarray,
    *,
    q0: int,
    max_moment_ratio: float,
    whiten_basis: bool,
    svd_rel_tol: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build an instrument basis B(W) that is numerically stable:

      - Keeps (one) constant column if present.
      - Enforces K*q0 <= max_moment_ratio * N via a K-budget.
      - Optionally orthonormalizes non-constant columns via SVD (recommended).

    Returns
    -------
    B_eff : (N, K_eff)
    meta : dict
    """
    B_raw = np.asarray(B_raw, float)
    if B_raw.ndim != 2:
        raise ValueError("B_raw must be 2D.")
    if not np.isfinite(B_raw).all():
        raise ValueError("B_raw contains NaN/Inf.")
    N, K_raw = B_raw.shape
    if K_raw == 0:
        raise ValueError("B_raw must have at least 1 column.")
    q0 = int(q0)
    if q0 <= 0:
        raise ValueError("q0 must be positive.")
    max_moment_ratio = float(max_moment_ratio)
    if not (0 < max_moment_ratio <= 1.0):
        raise ValueError("max_moment_ratio must be in (0, 1].")

    # Budget (total columns, including constant if present)
    K_budget_total = max(1, int(np.floor(max_moment_ratio * N / max(1, q0))))
    const_idx = _detect_constant_column(B_raw)
    if const_idx is None:
        const_col = None
        B_nonconst = B_raw
        K_budget_nonconst = K_budget_total
    else:
        const_col = B_raw[:, [const_idx]]
        mask = np.ones(K_raw, dtype=bool)
        mask[const_idx] = False
        B_nonconst = B_raw[:, mask]
        K_budget_nonconst = max(0, K_budget_total - 1)

    kept_idx = None
    svals_kept = None

    if B_nonconst.shape[1] == 0 or K_budget_nonconst == 0:
        B_eff_nonconst = np.empty((N, 0), dtype=float)
        kept_idx = np.array([], dtype=int)
    else:
        if whiten_basis:
            # Orthonormalize (mean-center first)
            Bc = B_nonconst - B_nonconst.mean(axis=0, keepdims=True)
            U, S, Vt = svd(Bc, full_matrices=False, check_finite=True)
            if S.size == 0:
                B_eff_nonconst = np.empty((N, 0), dtype=float)
                kept_idx = np.array([], dtype=int)
            else:
                S0 = float(S[0])
                rel_keep = max(1e-12, float(svd_rel_tol))
                keep = (S0 > 0.0) & ((S / (S0 if S0 > 0 else 1.0)) >= rel_keep)
                if not np.any(keep):
                    keep = np.zeros_like(S, dtype=bool)
                    keep[0] = True
                r = int(min(np.sum(keep), K_budget_nonconst))
                B_eff_nonconst = U[:, :r]
                kept_idx = np.nonzero(keep)[0][:r]
                svals_kept = S[:r].copy()
        else:
            # Simple pruning by column std (drop near-constant, keep largest-variance)
            colstd = B_nonconst.std(axis=0, ddof=1)
            good = colstd > 1e-12
            idx = np.where(good)[0]
            if idx.size == 0:
                idx = np.arange(min(1, B_nonconst.shape[1]))
            if idx.size > K_budget_nonconst:
                order = np.argsort(colstd[idx])[::-1][:K_budget_nonconst]
                idx = idx[order]
            B_eff_nonconst = B_nonconst[:, idx]
            kept_idx = idx

    B_eff = np.hstack([const_col, B_eff_nonconst]) if const_col is not None else B_eff_nonconst
    meta = {
        "K_raw": int(K_raw),
        "K_eff": int(B_eff.shape[1]),
        "K_budget_total": int(K_budget_total),
        "K_budget_nonconst": int(K_budget_nonconst),
        "const_idx_raw": const_idx,
        "whiten_basis": bool(whiten_basis),
        "svd_rel_tol": float(svd_rel_tol),
        "kept_nonconst_idx": kept_idx,
        "singular_values_kept": svals_kept,
    }
    return np.ascontiguousarray(B_eff), meta


# =============================================================================
# Structural BETEL (GenELV2-backed)
# =============================================================================

class StructuralBETELV2(GenELV2):
    def __init__(
        self,
        *,
        Y: np.ndarray,
        X: np.ndarray,
        W: np.ndarray,
        g_yh: Callable[[np.ndarray, np.ndarray], np.ndarray],
        theta0: np.ndarray,
        phi_map: Union[Callable[[np.ndarray], np.ndarray], np.ndarray],
        d_h: int = 1,

        # --- instrument basis ---
        instrument_basis: Union[str, np.ndarray, None] = None,
        # for "patsy_cr_paper_tilde"
        df_first_w: int = 3,
        df_rest_w: int = 3,
        center_basis: bool = True,
        scale_basis: bool = True,
        add_constant: bool = True,
        # for "cr_df_1d"
        df_w: int = 5,

        whiten_basis: bool = True,
        svd_rel_tol: float = 1e-10,
        max_moment_ratio: float = 0.6,

        # --- regularization on theta ---
        reg_lambda: float = 0.0,

        # --- optional analytic base-moment Jacobian ---
        base_moment_jac: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
        moment_is_residual: Optional[bool] = None,

        # --- structural init ---
        init_theta: Optional[str] = None,

        # --- passthrough to GenELV2 ---
        alpha: float = 0.0,
        m: int = 100,
        B_nums: int = 1000,
        random_state: int = 42,
        bounds: Optional[Any] = None,
        newton_tol: float = 1e-6,
        newton_maxiter: int = 100,
        newton_ridge: float = 1e-8,
        use_sc_newton: bool = True,
        enable_penalized: bool = False,
        lam_max: float = 50.0,
        per_v_gmm_start: bool = False,

        outer_use_analytic_grad: bool = True,
        outer_fd_eps: float = 1e-6,
        outer_grad_ridge: float = 1e-10,
    ):
        # --- data ---
        Y = _as_2d(Y, name="Y")
        X = _as_2d(X, name="X")
        W = _as_2d(W, name="W")
        if not (Y.shape[0] == X.shape[0] == W.shape[0]):
            raise ValueError("Y, X, W must have the same number of rows.")
        self.Y = Y
        self.X_endog = X
        self.W_inst = W
        self.g_yh = g_yh
        self.d_h = int(d_h)
        if self.d_h <= 0:
            raise ValueError("d_h must be positive.")

        # --- Φ(X) ---
        if callable(phi_map):
            Phi = np.asarray(phi_map(self.X_endog), float)
        else:
            Phi = np.asarray(phi_map, float)
        if Phi.ndim != 2 or Phi.shape[0] != X.shape[0]:
            raise ValueError("phi_map must yield an array of shape (N,p).")
        if not np.isfinite(Phi).all():
            raise ValueError("Phi(X) contains NaN/Inf.")
        self.Phi = np.ascontiguousarray(Phi)
        self.p = int(self.Phi.shape[1])

        theta0 = np.asarray(theta0, float).reshape(-1)
        if theta0.size != self.p * self.d_h:
            raise ValueError(f"theta0 length {theta0.size} != p*d_h = {self.p*self.d_h}.")
        self._theta_shape = (self.p, self.d_h)

        # --- regularization ---
        self.reg_lambda = float(reg_lambda)
        if self.reg_lambda < 0:
            raise ValueError("reg_lambda must be nonnegative.")

        # --- build instrument basis B(W) ---
        if instrument_basis is None:
            instrument_basis = "patsy_cr_paper_tilde"

        if isinstance(instrument_basis, str):
            mode = instrument_basis.lower()
            if mode in {"patsy_cr_paper_tilde", "patsy"}:
                B_raw, basis_meta = make_basis_ns_patsy_paper(
                    self.W_inst,
                    df_first=df_first_w,
                    df_rest=df_rest_w,
                    center=center_basis,
                    scale=scale_basis,
                    add_constant=add_constant,
                )
            elif mode in {"cr_df_1d", "df_1d"}:
                if self.W_inst.shape[1] != 1:
                    raise ValueError("instrument_basis='cr_df_1d' requires W to be 1D.")
                B_raw, basis_meta = make_basis_cr_df_1d(
                    self.W_inst[:, 0],
                    df=df_w,
                    add_constant=add_constant,
                    center=center_basis,
                    scale=scale_basis,
                )
            else:
                raise ValueError(
                    f"Unknown instrument_basis='{instrument_basis}'. "
                    f"Try: 'patsy_cr_paper_tilde', 'cr_df_1d', or pass an array."
                )
        else:
            B_raw = np.asarray(instrument_basis, float)
            if B_raw.ndim != 2 or B_raw.shape[0] != X.shape[0]:
                raise ValueError("instrument_basis array must have shape (N, K).")
            basis_meta = {"basis_mode": "user_array", "K_raw": int(B_raw.shape[1])}

        # --- determine base moment dimension q0 at theta0 ---
        H0 = self._h_of_theta(theta0)
        G0 = self._eval_g0(self.Y, H0)
        if G0.ndim == 1:
            G0 = G0.reshape(X.shape[0], 1)
        if G0.shape[0] != X.shape[0]:
            raise ValueError("g_yh returned wrong number of rows.")
        self.q0 = int(G0.shape[1])
        if self.q0 <= 0:
            raise ValueError("g_yh must return at least one moment.")

        # --- stable basis ---
        self.B_basis, whiten_meta = _whiten_and_prune_basis(
            B_raw,
            q0=self.q0,
            max_moment_ratio=max_moment_ratio,
            whiten_basis=whiten_basis,
            svd_rel_tol=svd_rel_tol,
        )
        self.K_basis = int(self.B_basis.shape[1])
        self._basis_meta: Dict[str, Any] = {**basis_meta, **whiten_meta}

        # --- Jacobian control ---
        self.base_moment_jac = base_moment_jac
        self._moment_is_residual = False
        if moment_is_residual is not None:
            self._moment_is_residual = bool(moment_is_residual)
        else:
            # heuristic auto-detect: g_yh(Y,H) == Y - H ?
            try:
                yy = self.Y[:5]
                hh = H0[:5]
                gt = np.asarray(self.g_yh(yy, hh), float)
                if gt.ndim == 1:
                    gt = gt.reshape(yy.shape[0], 1)
                if gt.shape == (yy.shape[0], hh.shape[1]) and np.allclose(gt, yy - hh, atol=1e-12, rtol=1e-10):
                    self._moment_is_residual = True
            except Exception:
                self._moment_is_residual = False

        # --- wrappers for GenELV2 (so parent can use its machinery) ---
        def _g_wrapper(_X_ignored: np.ndarray, th: np.ndarray) -> np.ndarray:
            return self._G(th)

        def _g_jac_wrapper(_X_ignored: np.ndarray, th: np.ndarray) -> np.ndarray:
            return self._G_jac(th)

        super().__init__(
            X=self.X_endog,
            g=_g_wrapper,
            theta0=theta0,
            g_jac=_g_jac_wrapper,
            init_theta=None,  # structural init handled below
            alpha=alpha,
            m=m,
            B_nums=B_nums,
            random_state=random_state,
            bounds=bounds,
            newton_tol=newton_tol,
            newton_maxiter=newton_maxiter,
            newton_ridge=newton_ridge,
            use_sc_newton=use_sc_newton,
            enable_penalized=enable_penalized,
            lam_max=lam_max,
            per_v_gmm_start=per_v_gmm_start,
            outer_use_analytic_grad=outer_use_analytic_grad,
            outer_fd_eps=outer_fd_eps,
            outer_grad_ridge=outer_grad_ridge,
        )

        # Expanded moments are already stabilized; avoid additional scaling overhead.
        self.use_column_scaling = False

        # Optional structural ETEL init: one uniform-v outer solve
        if init_theta is not None:
            mode = str(init_theta).lower()
            if mode == "etel":
                v_uniform = np.full(self.n, 1.0 / self.n, dtype=float)
                out = self.fit_per_v(v=v_uniform, theta0=self.theta0, method="L-BFGS-B",
                                     options={"maxiter": 200, "ftol": 1e-8, "eps": 1e-6})
                self.theta0 = np.asarray(out["theta"], float).reshape(-1).copy()
                self._lambda_ws = np.zeros(self.q, dtype=float)
            else:
                raise ValueError("init_theta for StructuralBETELV2: use 'etel' or None.")

    # -------------------------------------------------------------------------
    # Structural pieces: h_theta, g0, expanded G(theta), Jacobian
    # -------------------------------------------------------------------------

    def _h_of_theta(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, float).reshape(-1)
        Θ = theta.reshape(self._theta_shape)      # (p, d_h)
        return self.Phi @ Θ                       # (N, d_h)

    def _eval_g0(self, Y: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Evaluate base moments g(Y,H) -> (N, q0)."""
        try:
            G0 = np.asarray(self.g_yh(Y, H), float)
            if G0.ndim == 1:
                G0 = G0.reshape(Y.shape[0], 1)
            if G0.shape[0] != Y.shape[0]:
                raise ValueError
            return G0
        except Exception:
            # per-row fallback
            rows = [np.asarray(self.g_yh(Y[i], H[i]), float).reshape(-1) for i in range(Y.shape[0])]
            G0 = np.vstack(rows)
            if G0.ndim == 1:
                G0 = G0.reshape(Y.shape[0], 1)
            return G0

    def _eval_g0_jac(self, theta: np.ndarray, H: np.ndarray, G0: np.ndarray) -> np.ndarray:
        """
        Jacobian of base moments g(Y,H(theta)) wrt theta.

        Returns J0 with shape (N, q0, d_theta).
        """
        theta = np.asarray(theta, float).reshape(-1)
        N = self.n
        d_theta = theta.size

        # User-supplied base Jacobian has priority.
        if self.base_moment_jac is not None:
            J0 = np.asarray(self.base_moment_jac(self.Y, H, theta), float)
            if J0.ndim != 3 or J0.shape != (N, self.q0, d_theta):
                raise ValueError(
                    f"base_moment_jac must return (N,q0,d_theta)=({N},{self.q0},{d_theta}); got {J0.shape}"
                )
            return J0

        # Fast path for residual moments: g(Y,H) = Y - H.
        if self._moment_is_residual:
            # H = Phi @ Θ, Θ ∈ R^{p×d_h}, theta = vec(Θ)
            # g0 is (N,d_h) -> q0 == d_h
            if self.q0 != self.d_h:
                # If user returns (N,1) even when d_h>1, we can't infer structure.
                raise ValueError("Residual moment detected but q0 != d_h; please supply base_moment_jac.")
            J0 = np.zeros((N, self.q0, d_theta), dtype=float)
            # Block diagonal: for each output j, d(Y_j - H_j)/d theta_{:,j} = -Phi
            for j in range(self.d_h):
                col0 = j * self.p
                col1 = (j + 1) * self.p
                J0[:, j, col0:col1] = -self.Phi
            return J0

        # Otherwise: numerical finite differences of g0 via perturbing theta (expensive but general).
        # NOTE: This is only used when outer_use_analytic_grad=True (needs a Jacobian). If you
        # hit this path often, consider supplying base_moment_jac.
        eps = float(self.outer_fd_eps)
        if not (eps > 0.0 and np.isfinite(eps)):
            raise ValueError("outer_fd_eps must be positive for numerical Jacobian.")
        J0 = np.empty((N, self.q0, d_theta), dtype=float)
        for k in range(d_theta):
            h = eps * max(1.0, float(abs(theta[k])))
            tp = theta.copy(); tp[k] += h
            tm = theta.copy(); tm[k] -= h
            Hp = self._h_of_theta(tp)
            Hm = self._h_of_theta(tm)
            Gp = self._eval_g0(self.Y, Hp)
            Gm = self._eval_g0(self.Y, Hm)
            if Gp.ndim == 1:
                Gp = Gp.reshape(N, 1)
            if Gm.ndim == 1:
                Gm = Gm.reshape(N, 1)
            J0[:, :, k] = (Gp - Gm) / (2.0 * h)
        return J0

    def _G(self, theta: np.ndarray) -> np.ndarray:
        """Expanded moments G(theta) ∈ R^{N × (K*q0)}."""
        theta = np.asarray(theta, float).reshape(-1)
        H = self._h_of_theta(theta)
        G0 = self._eval_g0(self.Y, H)             # (N,q0)
        if G0.ndim == 1:
            G0 = G0.reshape(self.n, 1)
        B = self.B_basis                          # (N,K)
        N, K = B.shape
        q0 = G0.shape[1]
        out = np.empty((N, K * q0), dtype=float)
        off = 0
        for k in range(K):
            out[:, off:off + q0] = B[:, [k]] * G0
            off += q0
        return out

    def _G_jac(self, theta: np.ndarray) -> np.ndarray:
        """Jacobian of expanded moments wrt theta: (N, K*q0, d_theta)."""
        theta = np.asarray(theta, float).reshape(-1)
        H = self._h_of_theta(theta)
        G0 = self._eval_g0(self.Y, H)
        if G0.ndim == 1:
            G0 = G0.reshape(self.n, 1)

        J0 = self._eval_g0_jac(theta, H, G0)      # (N,q0,d)
        B = self.B_basis                          # (N,K)
        N, K = B.shape
        q0 = self.q0
        d = theta.size

        J = np.empty((N, K * q0, d), dtype=float)
        off = 0
        for k in range(K):
            J[:, off:off + q0, :] = B[:, [k], None] * J0
            off += q0
        return J

    # -------------------------------------------------------------------------
    # Regularized loss (ETEL + ridge)
    # -------------------------------------------------------------------------

    def _loss_only(self, theta: np.ndarray, v: np.ndarray):
        loss, inner = super()._loss_only(theta, v)
        if self.reg_lambda > 0.0:
            th = np.asarray(theta, float).reshape(-1)
            loss = float(loss + self.reg_lambda * float(th @ th))
        return float(loss), inner

    def _loss_and_grad(self, theta: np.ndarray, v: np.ndarray):
        loss, grad, inner = super()._loss_and_grad(theta, v)
        if self.reg_lambda > 0.0:
            th = np.asarray(theta, float).reshape(-1)
            loss = float(loss + self.reg_lambda * float(th @ th))
            grad = np.asarray(grad, float).reshape(-1) + (2.0 * self.reg_lambda) * th
        return float(loss), np.asarray(grad, float), inner
