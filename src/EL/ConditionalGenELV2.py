from __future__ import annotations
from typing import Callable, Optional, Tuple, Dict, Any, List
import inspect

import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Import GenELV2 / GenELResult with robust fallback
# -----------------------------------------------------------------------------
try:  # package-style
    from .GenELV2 import GenELV2, GenELResult
except Exception:  # script-style
    from GenELV2 import GenELV2, GenELResult


try:
    # used by the default spline basis builder below
    from patsy import dmatrix
except Exception:  # pragma: no cover
    dmatrix = None


# -----------------------------------------------------------------------------
# Basis builder: natural cubic regression spline w/ “paper-style tilde” for j>=1
# -----------------------------------------------------------------------------
def make_basis_ns_patsy_paper(
    W: np.ndarray,
    *,
    df_first: int = 3,
    df_rest: int = 3,
    center: bool = False,
    scale: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build a spline basis matrix B(W) **without** an intercept column.

    Construction follows the “paper-style tilde” rule:
      - For W[:,0]: cr(x, df=df_first) contributes df_first columns.
      - For W[:,j], j>=1: use cr(x, df=df_rest) and transform to
            Bj[:, 1:] - Bj[:, [0]]
        contributing df_rest-1 columns per additional dimension.

    Parameters
    ----------
    W : array_like, shape (N, d_w)
    df_first, df_rest : int
    center, scale : bool
        Applied to the non-constant block (this function returns no intercept).
    """

    if dmatrix is None:
        raise ImportError(
            "patsy is required for make_basis_ns_patsy_paper. "
            "Install patsy or replace the basis builder with your own."
        )

    W = np.asarray(W, float)
    if W.ndim == 1:
        W = W[:, None]
    if not np.isfinite(W).all():
        raise ValueError("W contains NaN/Inf.")

    N, d = W.shape
    if df_first < 2 or df_rest < 2:
        raise ValueError("df_first and df_rest must be >= 2.")

    parts: List[np.ndarray] = []

    if d >= 1:
        B0 = np.asarray(dmatrix(f"0 + cr(x, df={df_first})", {"x": W[:, 0]}), float)
        if B0.shape[1] != df_first:
            raise RuntimeError("Unexpected patsy output for j=0.")
        parts.append(B0)

        for j in range(1, d):
            Bj = np.asarray(dmatrix(f"0 + cr(x, df={df_rest})", {"x": W[:, j]}), float)
            if Bj.shape[1] < 2:
                raise RuntimeError(f"patsy cr produced <2 columns for j={j}.")
            parts.append(Bj[:, 1:] - Bj[:, [0]])

    B_noconst = np.concatenate(parts, axis=1) if parts else np.empty((N, 0), dtype=float)

    if center or scale:
        Bc = B_noconst.copy()
        if center:
            Bc -= Bc.mean(axis=0, keepdims=True)
        if scale and Bc.shape[1] > 0:
            s = Bc.std(axis=0, ddof=1, keepdims=True)
            s[s < 1e-12] = 1.0
            Bc /= s
        B_noconst = Bc

    meta = {
        "basis_mode": "patsy_cr_paper_tilde",
        "df_first": int(df_first),
        "df_rest": int(df_rest),
        "d_w": int(d),
        "K_raw_noconst": int(B_noconst.shape[1]),
        "center": bool(center),
        "scale": bool(scale),
        "shapes": {"B_noconst": B_noconst.shape},
    }
    return B_noconst, meta


# -----------------------------------------------------------------------------
# Parallel worker helpers (per-process singleton)
# -----------------------------------------------------------------------------
_COND_V2_WORKER: Optional["ConditionGenELV2"] = None


def _init_worker_cond_v2(init_kwargs: Dict[str, Any]):
    """Initializer for ProcessPoolExecutor workers.

    Note: ProcessPoolExecutor.initializer only supports positional initargs,
    so we pass a single dict.
    """
    global _COND_V2_WORKER
    _COND_V2_WORKER = ConditionGenELV2(**init_kwargs)


def _solve_one_cond_v2(task):
    i, seed, method, options = task
    v = _COND_V2_WORKER.draw_dirichlet(seed)
    out = _COND_V2_WORKER.fit_per_v(v=v, method=method, options=options)
    out.pop("opt_result", None)
    return i, out


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------
class ConditionGenELV2(GenELV2):
    """Conditional GenELV2 via basis expansion.

    Parameters
    ----------
    X : (N, d_x)
    W : (N, d_w)
    g : callable
        Base moment function; should return shape (N, q0) or (N,).
    g_jac : callable, optional
        Jacobian of the base moment function. Expected shape:
            (N, q0, d_theta)

        If provided, this class constructs the expanded Jacobian.
        If not provided, and `outer_use_analytic_grad=True`, this class will
        compute a finite-difference Jacobian of the base moments and expand it.
    """

    def __init__(
        self,
        X: np.ndarray,
        W: np.ndarray,
        g: Callable[[np.ndarray, np.ndarray], np.ndarray],
        theta0: np.ndarray,
        *,
        g_jac: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        # basis controls
        df_first: int = 3,
        df_rest: int = 3,
        center_basis: bool = True,
        scale_basis: bool = True,
        add_constant: bool = True,
        whiten_basis: bool = True,
        svd_tol: float = 1e-10,
        max_moment_ratio: float = 0.9,
        # initialization
        init_theta: Optional[str] = "etel",
        # GenELV2 passthrough
        alpha: float = 0.0,
        m: int = 100,
        B_nums: int = 1000,
        random_state: int = 42,
        bounds: Optional[Any] = None,
        newton_tol: float = 1e-8,
        newton_maxiter: int = 100,
        newton_ridge: float = 1e-8,
        use_sc_newton: bool = True,
        enable_penalized: bool = False,
        lam_max: float = 50.0,
        per_v_gmm_start: bool = False,
        # newer GenELV2 outer-gradient knobs (ignored by older GenELV2)
        outer_use_analytic_grad: bool = True,
        outer_fd_eps: float = 1e-6,
        outer_grad_ridge: float = 1e-10,
    ):
        X = np.asarray(X, float)
        W = np.asarray(W, float)
        if W.ndim == 1:
            W = W[:, None]
        if X.shape[0] != W.shape[0]:
            raise ValueError("X and W must have the same number of rows.")
        if not (np.isfinite(X).all() and np.isfinite(W).all()):
            raise ValueError("X/W contain NaN/Inf values.")

        self.W = W
        self._g_user = g
        self._g_jac_user = g_jac
        self._outer_fd_eps_local = float(outer_fd_eps)

        theta0 = np.asarray(theta0, float).reshape(-1)
        self.d_theta = int(theta0.shape[0])

        # ---- Probe g to learn q0 and vectorization ----
        self._g_user_vectorized = True
        try:
            G0 = np.asarray(g(X, theta0), float)
            if G0.ndim == 1:
                if G0.shape[0] != X.shape[0]:
                    raise ValueError
                G0 = G0.reshape(X.shape[0], 1)
            if G0.shape[0] != X.shape[0]:
                raise ValueError
        except Exception:
            self._g_user_vectorized = False
            rows = [np.asarray(g(X[i], theta0), float).reshape(-1) for i in range(X.shape[0])]
            G0 = np.vstack(rows)
            if G0.ndim == 1:
                G0 = G0.reshape(X.shape[0], 1)

        self.q0 = int(G0.shape[1])
        N = int(X.shape[0])

        # ---- Build raw basis (no intercept) ----
        B_raw, basis_meta = make_basis_ns_patsy_paper(
            W,
            df_first=df_first,
            df_rest=df_rest,
            center=center_basis,
            scale=scale_basis,
        )

        B_for_svd = np.asarray(B_raw, float)
        if B_for_svd.ndim == 1:
            B_for_svd = B_for_svd.reshape(N, 1)
        if B_for_svd.shape[0] != N:
            raise ValueError("Basis has wrong number of rows.")

        # ---- Budget: enforce (K_eff * q0) <= floor(max_moment_ratio * N) ----
        max_moment_ratio = float(max_moment_ratio)
        if not (0.0 < max_moment_ratio <= 1.0):
            raise ValueError("max_moment_ratio must be in (0, 1].")

        K_total_budget = int(np.floor(max_moment_ratio * N / max(1, self.q0)))
        if add_constant:
            K_budget_nonconst = max(0, K_total_budget - 1)
        else:
            K_budget_nonconst = max(1, K_total_budget)

        whiten_basis = bool(whiten_basis)
        svd_tol = float(svd_tol)

        if whiten_basis and B_for_svd.shape[1] > 0:
            Bc = B_for_svd - B_for_svd.mean(axis=0, keepdims=True)
            U, S, _ = np.linalg.svd(Bc, full_matrices=False)

            S0 = float(S[0]) if S.size else 0.0
            rel_tol = max(1e-12, svd_tol)
            keep = (S0 > 0.0) & (S / (S0 if S0 > 0.0 else 1.0) >= rel_tol)
            if not np.any(keep) and S.size:
                keep = np.zeros_like(S, dtype=bool)
                keep[0] = True

            r = int(min(int(np.sum(keep)), K_budget_nonconst))
            B_orth = U[:, :r] if r > 0 else np.empty((N, 0), dtype=float)
            kept_idx = np.nonzero(keep)[0][:r]
            svals_kept = S[:r].copy()
        else:
            if B_for_svd.shape[1] > 0:
                colstd = B_for_svd.std(axis=0)
                good = colstd > 1e-12
                idx = np.where(good)[0]
                if idx.size == 0:
                    idx = np.arange(min(1, B_for_svd.shape[1]))
                if idx.size > K_budget_nonconst:
                    order = np.argsort(colstd[idx])[::-1][:K_budget_nonconst]
                    idx = idx[order]
                B_orth = B_for_svd[:, idx]
                kept_idx = idx
                svals_kept = None
            else:
                B_orth = np.empty((N, 0), dtype=float)
                kept_idx = np.array([], dtype=int)
                svals_kept = None

        if add_constant:
            const_col = np.ones((N, 1), dtype=float)
            self.B_basis = np.ascontiguousarray(np.hstack([const_col, B_orth]))
        else:
            if B_orth.shape[1] == 0:
                raise ValueError("No basis columns left after pruning and add_constant=False.")
            self.B_basis = np.ascontiguousarray(B_orth)

        self.K_basis = int(self.B_basis.shape[1])
        self._basis_meta = {
            **basis_meta,
            "add_constant": bool(add_constant),
            "whiten_basis": bool(whiten_basis),
            "svd_rel_tol": float(svd_tol),
            "K_eff": int(self.K_basis),
            "K_budget_total": int(K_total_budget),
            "K_budget_nonconst": int(K_budget_nonconst),
            "kept_columns_from_raw": kept_idx,
            "singular_values_kept": svals_kept,
            "q0": int(self.q0),
            "q_expanded": int(self.K_basis * self.q0),
        }

        self._cond_cfg = dict(
            df_first=int(df_first),
            df_rest=int(df_rest),
            center_basis=bool(center_basis),
            scale_basis=bool(scale_basis),
            add_constant=bool(add_constant),
            whiten_basis=bool(whiten_basis),
            svd_tol=float(svd_tol),
            max_moment_ratio=float(max_moment_ratio),
        )

        # ---- Call parent init, filtering kwargs for compatibility ----
        super_kwargs: Dict[str, Any] = dict(
            X=X,
            g=g,  # unused by our overridden _G, but required by GenELV2
            theta0=theta0,
            init_theta=None,
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
            # new API
            g_jac=self._g_jac_expanded,
            outer_use_analytic_grad=bool(outer_use_analytic_grad),
            outer_fd_eps=float(outer_fd_eps),
            outer_grad_ridge=float(outer_grad_ridge),
        )
        sig = inspect.signature(GenELV2.__init__)
        filtered = {k: v for k, v in super_kwargs.items() if k in sig.parameters}
        super().__init__(**filtered)

        # Column scaling rarely helps for large expanded q; keep it off by default.
        if hasattr(self, "use_column_scaling"):
            self.use_column_scaling = False

        # Optional ETEL init (in expanded space)
        if init_theta is not None:
            mode = str(init_theta).lower()
            if mode == "etel":
                theta_pilot = self._gmm_pilot_start(self.theta0)
                self.theta0 = self._etel_start(theta_pilot)
                self._lambda_ws = np.zeros(self.q, dtype=float)
            else:
                raise ValueError(f"Unknown init_theta='{init_theta}'. Try: 'etel' or None.")

    @property
    def basis_meta(self) -> Dict[str, Any]:
        return dict(self._basis_meta)

    # ------------------------------------------------------------------
    # Base moments and Jacobian
    # ------------------------------------------------------------------
    def _eval_g0(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, float).reshape(-1)
        N = self.X.shape[0]

        if self._g_user_vectorized:
            G0 = np.asarray(self._g_user(self.X, theta), float)
            if G0.ndim == 1:
                if G0.shape[0] != N:
                    raise ValueError("g returned wrong length")
                G0 = G0.reshape(N, 1)
            if G0.shape[0] != N:
                raise ValueError("g returned wrong number of rows")
            return G0

        rows = [np.asarray(self._g_user(self.X[i], theta), float).reshape(-1) for i in range(N)]
        G0 = np.vstack(rows)
        if G0.ndim == 1:
            G0 = G0.reshape(N, 1)
        if G0.shape[0] != N:
            raise ValueError("Per-row g returned inconsistent shapes")
        return G0

    def _eval_g0_jac(self, theta: np.ndarray) -> np.ndarray:
        """Return Jacobian of base moments: (N, q0, d_theta)."""
        theta = np.asarray(theta, float).reshape(-1)
        N = self.X.shape[0]

        if self._g_jac_user is not None:
            J0 = np.asarray(self._g_jac_user(self.X, theta), float)
            if J0.ndim != 3:
                raise ValueError("g_jac must return array with ndim=3: (N, q0, d_theta)")
            if J0.shape[0] != N or J0.shape[1] != self.q0 or J0.shape[2] != self.d_theta:
                raise ValueError(
                    f"g_jac returned shape {J0.shape}, expected ({N}, {self.q0}, {self.d_theta})"
                )
            return J0

        # Finite-difference Jacobian (moments only). This is cheap relative to
        # finite-differencing the *loss* because it does NOT re-solve λ.
        eps = float(self._outer_fd_eps_local)
        if eps <= 0:
            raise ValueError("outer_fd_eps must be positive when g_jac is None")

        base = self._eval_g0(theta)  # (N, q0)
        J0 = np.empty((N, self.q0, self.d_theta), dtype=float)
        for j in range(self.d_theta):
            th_p = theta.copy(); th_p[j] += eps
            th_m = theta.copy(); th_m[j] -= eps
            Gp = self._eval_g0(th_p)
            Gm = self._eval_g0(th_m)
            J0[:, :, j] = (Gp - Gm) / (2.0 * eps)
        return J0

    # ------------------------------------------------------------------
    # Expanded moments and Jacobian
    # ------------------------------------------------------------------
    def _G(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, float).reshape(-1)

        G0 = self._eval_g0(theta)  # (N, q0)
        B = self.B_basis           # (N, K)
        N = self.X.shape[0]
        if B.shape[0] != N:
            raise RuntimeError("Basis and X have different row counts")

        q0 = int(G0.shape[1])
        K = int(B.shape[1])
        out = np.empty((N, K * q0), dtype=float)
        off = 0
        for k in range(K):
            out[:, off:off + q0] = B[:, [k]] * G0
            off += q0
        return out

    def _G_jac(self, theta: np.ndarray) -> np.ndarray:
        """Jacobian of expanded moments: (N, K*q0, d_theta)."""
        theta = np.asarray(theta, float).reshape(-1)
        N = self.X.shape[0]
        B = self.B_basis
        K = int(B.shape[1])
        q0 = int(self.q0)

        J0 = self._eval_g0_jac(theta)  # (N, q0, d_theta)
        J = np.empty((N, K * q0, self.d_theta), dtype=float)
        off = 0
        for k in range(K):
            # broadcast: (N,1,1) * (N,q0,d)
            J[:, off:off + q0, :] = B[:, [k], None] * J0
            off += q0
        return J

    def _g_jac_expanded(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Signature-compatible wrapper for GenELV2: (X,θ)->(N,q,d)."""
        _ = X  # unused; we use self.X/self.B_basis
        return self._G_jac(theta)

    # ------------------------------------------------------------------
    # Parallel bootstrap
    # ------------------------------------------------------------------
    def fit_in_parallel(
        self,
        max_workers: int | None = None,
        method: str = "BFGS",
        options: dict | None = None,
        store_weights: bool = False,
        store_b_weights: bool = False,
    ) -> GenELResult:
        """Parallel bootstrap that correctly replicates conditional basis in workers."""
        if self.alpha != 0:
            raise NotImplementedError("GenAI version (alpha > 0) not implemented yet.")

        # Allocate
        thetas = np.zeros((self.B, self.d_theta))
        losses = np.zeros(self.B, dtype=float)
        lambdas = np.zeros((self.B, self.q))
        weights = np.zeros((self.B, self.n)) if store_weights else None
        successes = np.zeros(self.B, dtype=bool)
        b_weights = np.zeros((self.B, self.n)) if store_b_weights else None

        # Seeds
        base_ss = np.random.SeedSequence(int(getattr(self, "random_state", 42)))
        child_ss = base_ss.spawn(self.B)
        seeds = [int(s.generate_state(1)[0]) for s in child_ss]

        if max_workers is None:
            max_workers = os.cpu_count() or 1
        chunksize = max(1, self.B // (4 * max_workers))

        cfg = self._cond_cfg

        init_kwargs = dict(
            X=self.X,
            W=self.W,
            g=self._g_user,
            g_jac=self._g_jac_user,
            theta0=self.theta0,
            # basis
            df_first=cfg["df_first"],
            df_rest=cfg["df_rest"],
            center_basis=cfg["center_basis"],
            scale_basis=cfg["scale_basis"],
            add_constant=cfg["add_constant"],
            whiten_basis=cfg["whiten_basis"],
            svd_tol=cfg["svd_tol"],
            max_moment_ratio=cfg["max_moment_ratio"],
            # GenELV2
            init_theta=None,
            alpha=0.0,
            m=0,
            B_nums=1,
            random_state=0,
            bounds=self.bounds,
            newton_tol=self.newton_tol,
            newton_maxiter=self.newton_maxiter,
            newton_ridge=self.newton_ridge,
            use_sc_newton=self.use_sc_newton,
            enable_penalized=self.enable_penalized,
            lam_max=self.lam_max,
            per_v_gmm_start=self.per_v_gmm_start,
            # new outer-gradient knobs (if supported)
            outer_use_analytic_grad=getattr(self, "outer_use_analytic_grad", True),
            outer_fd_eps=getattr(self, "outer_fd_eps", self._outer_fd_eps_local),
            outer_grad_ridge=getattr(self, "outer_grad_ridge", 1e-10),
        )

        opts = options or {"maxiter": 500, "gtol": 1e-6}

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker_cond_v2,
            initargs=(init_kwargs,),
        ) as ex:
            tasks = ((i, seeds[i], method, opts) for i in range(self.B))
            for i, out in tqdm(
                ex.map(_solve_one_cond_v2, tasks, chunksize=chunksize),
                total=self.B,
                desc="Bootstrap (cond)",
                unit="draw",
            ):
                thetas[i] = out["theta"]
                losses[i] = out["loss"]
                lambdas[i] = out["lambda_star"]
                successes[i] = out["success"]
                if store_weights:
                    weights[i] = out["weights"]
                if store_b_weights:
                    b_weights[i] = self.draw_dirichlet(seeds[i])

        # Build result object in a version-tolerant way
        res = GenELResult()
        if hasattr(res, "theta_draws"):
            res.theta_draws = thetas
        if hasattr(res, "thetas"):
            res.thetas = thetas
        if hasattr(res, "loss_draws"):
            res.loss_draws = losses
        if hasattr(res, "loss"):
            res.loss = losses
        if hasattr(res, "lambda_draws"):
            res.lambda_draws = lambdas
        if hasattr(res, "lambda_stars"):
            res.lambda_stars = lambdas
        if hasattr(res, "weights"):
            res.weights = weights
        if hasattr(res, "b_weights"):
            res.b_weights = b_weights
        if hasattr(res, "success"):
            res.success = successes
        return res
