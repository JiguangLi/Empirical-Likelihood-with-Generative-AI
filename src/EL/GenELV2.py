from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any, Literal
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize as _minimize
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm.auto import tqdm

# ------------ Parallel ---------------------------------------------------------
_WORKER = None  # per-process singleton


def _solve_one(args):
    """
    Each worker: draw v from the provided seed, then solve one bootstrap draw.
    """
    i, seed, method, options = args
    v = _WORKER.draw_dirichlet(seed)
    out = _WORKER.fit_per_v(v=v, method=method, options=options)
    out.pop("opt_result", None)
    return i, out


def _init_worker(
    X,
    g,
    theta0,
    g_jac,
    bounds,
    newton_tol,
    newton_maxiter,
    newton_ridge,
    use_sc_newton,
    enable_penalized,
    lam_max,
    per_v_gmm_start,
    outer_use_analytic_grad,
    outer_fd_eps,
    outer_grad_ridge,
):
    """
    Create one GenELV21 instance in each worker process.
    """
    global _WORKER
    _WORKER = GenELV2(
        X=X,
        g=g,
        theta0=theta0,
        g_jac=g_jac,
        init_theta=None,
        alpha=0.0,
        m=0,
        B_nums=1,
        random_state=0,
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


def _logsumexp_weighted(scores: np.ndarray, v: np.ndarray) -> float:
    """
    Compute log( sum_i v_i * exp(scores_i) ) with max subtraction.
    """
    if np.any(v < 0):
        raise ValueError("All v_i must be nonnegative.")
    if np.all(v == 0):
        return -np.inf
    m = np.max(scores)
    return float(m + np.log(np.sum(v * np.exp(scores - m))))


def _safe_clip(x: np.ndarray, bound: float) -> np.ndarray:
    if bound <= 0:
        return x
    return np.clip(x, -bound, bound)


@dataclass
class GenELResult:
    thetas: Optional[np.ndarray] = None        # (B, d_theta)
    loss: Optional[np.ndarray] = None          # (B,)
    lambda_stars: Optional[np.ndarray] = None  # (B, q)
    weights: Optional[np.ndarray] = None       # (B, N)
    b_weights: Optional[np.ndarray] = None     # (B, N) bootstrap v's
    success: Optional[np.ndarray] = None       # (B,)


class GenELV2:
    """
    GenEL v2.1: DP/Bayesian-bootstrap Empirical Likelihood with Exponential Tilting.

    Key changes vs V1:
      - λ is hard-bounded by lam_max everywhere (prevents numerical blow-ups).
      - SC Newton is kept as a fast path (optional, on by default) but clipped.
      - Penalized Newton is opt-in (enable_penalized=False by default).
      - Robust fallback inner solver is bounded L-BFGS-B on f(λ)=log sum v exp(λ'g).
      - Acceptance is based on residual + finiteness (not SciPy success flags).
      - Safe warm-start update to avoid poisoning subsequent evaluations.

    Parameters
    ----------
    use_sc_newton : bool
        Whether to try SC Newton first.
    enable_penalized : bool
        Whether to allow penalized Newton fallback (opt-in).
    lam_max : float
        Componentwise bound on λ (default 50.0).
    per_v_gmm_start : bool
        Whether to compute a v-weighted GMM pilot θ for each v draw.
    """

    def __init__(
        self,
        X: np.ndarray,
        g: Callable[[np.ndarray, np.ndarray], np.ndarray],
        theta0: np.ndarray,
        g_jac: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        init_theta: Optional[str] = "etel",
        alpha: float = 0.0,
        m: int = 100,
        B_nums: int = 1000,
        random_state: int = 42,
        bounds: Optional[Any] = None,
        newton_tol: float = 1e-8,
        newton_maxiter: int = 100,
        newton_ridge: float = 1e-8,
        # v2.1
        use_sc_newton: bool = True,
        enable_penalized: bool = False,
        lam_max: float = 50.0,
        per_v_gmm_start: bool = False,

        # Outer optimization (theta) gradient control
        outer_use_analytic_grad: bool = False,
        outer_fd_eps: float = 1e-5,
        outer_grad_ridge: float = 1e-10,
    ):
        X = np.asarray(X, dtype=float)
        theta0 = np.asarray(theta0, dtype=float)
        if theta0.ndim != 1:
            raise ValueError("theta0 must be a 1-D array.")

        self.X = X
        self.n = self.X.shape[0]
        self.theta0 = theta0.copy()
        self.d_theta = theta0.shape[0]
        self.bounds = bounds
        self.g_fun = g
        self.g_jac_fun = g_jac
        self.alpha = float(alpha)
        self.m = 0 if self.alpha == 0 else int(m)
        self.B = int(B_nums)
        self.rng = np.random.default_rng(int(random_state))
        self.random_state = int(random_state)

        # Determine q (moment dimension) using a probe
        self._g_is_vectorized = True
        Gi = self._G(theta0)  # (N, q)
        if Gi.ndim != 2 or Gi.shape[0] != self.n:
            raise ValueError("g must produce an array of shape (N, q).")
        self.q = Gi.shape[1]
        try:
            Gtest = np.asarray(self.g_fun(self.X, self.theta0), float)
            if Gtest.ndim == 1 and Gtest.shape[0] == self.n:
                Gtest = Gtest.reshape(self.n, 1)
            assert Gtest.shape[0] == self.n
        except Exception:
            self._g_is_vectorized = False

        # λ-solver controls
        self.newton_tol = float(newton_tol)
        self.newton_maxiter = int(newton_maxiter)
        self.newton_ridge = float(newton_ridge)

        self.use_sc_newton = bool(use_sc_newton)
        self.enable_penalized = bool(enable_penalized)
        self.lam_max = float(lam_max)
        if self.lam_max <= 0:
            raise ValueError("lam_max must be positive.")

        # Warm start for λ*
        self._lambda_ws = np.zeros(self.q, dtype=float)

        # Outer-gradient knobs
        self.outer_use_analytic_grad = bool(outer_use_analytic_grad)
        self.outer_fd_eps = float(outer_fd_eps)
        if self.outer_fd_eps <= 0:
            raise ValueError("outer_fd_eps must be positive.")
        self.outer_grad_ridge = float(outer_grad_ridge)
        if self.outer_grad_ridge < 0:
            raise ValueError("outer_grad_ridge must be nonnegative.")

        # Feasibility tolerance (inner residual)
        self.res_tol = float(newton_tol)

        # Penalized Newton knobs (opt-in)
        self.penalty_rho0 = 1e-3
        self.penalty_rho_up = 5.0
        self.max_penalty_tries = 0  # v2.1 default: OFF (user can increase if enable_penalized=True)
        self.penalty_resid_threshold = 1e-2

        # Column scaling for inner solvers
        self.use_column_scaling = True

        # Armijo backtracking (SC Newton)
        self.armijo_c = 1e-4
        self.armijo_beta = 0.5
        self.max_ls_steps = 20

        # Optional residual penalty in the outer loss
        self.loss_resid_penalty = 0.0

        # Outer safety: if loss becomes non-finite, return a very large finite value
        self.outer_loss_big = 1e30

        # per-v θ pilot
        self.per_v_gmm_start = bool(per_v_gmm_start)

        # init theta
        if init_theta is not None:
            mode = str(init_theta).lower()
            if mode == "etel":
                theta_pilot = self._gmm_pilot_start(self.theta0)
                self.theta0 = self._etel_start(theta_pilot)
                self._lambda_ws = np.zeros(self.q, dtype=float)
            else:
                raise ValueError(f"Unknown init_theta='{init_theta}'. Try: 'etel' or None.")

    # ---------- Utilities ----------

    def _G(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, float).reshape(-1)
        if self._g_is_vectorized:
            G = np.asarray(self.g_fun(self.X, theta), float)
            if G.ndim == 1:
                if G.shape[0] != self.n:
                    raise ValueError("g returned wrong length")
                G = G.reshape(self.n, 1)
            if G.shape[0] != self.n:
                raise ValueError("g returned wrong number of rows")
            return G
        else:
            rows = [np.asarray(self.g_fun(self.X[i], theta), float).reshape(-1)
                    for i in range(self.n)]
            G = np.vstack(rows)
            if G.ndim == 1:
                G = G.reshape(self.n, 1)
            if G.shape[0] != self.n:
                raise ValueError("Per-row g returned inconsistent shapes.")
            return G

    def _G_jac(self, theta: np.ndarray) -> np.ndarray:
        """Jacobian of the moment matrix G(theta) wrt theta.

        Returns an array of shape (n, q, d_theta) where
            J[i, :, k] = d g_i(theta) / d theta_k.

        Notes
        -----
        * If `g_jac` was provided at construction, it is used.
        * Otherwise we fall back to central finite differences on `_G`.
          This is typically **much** more stable than finite differencing
          the full profiled ETEL loss because it avoids re-solving the
          inner lambda problem for each perturbation.
        """
        theta = np.asarray(theta, float).reshape(-1)

        # User-supplied Jacobian
        if self.g_jac_fun is not None:
            J = np.asarray(self.g_jac_fun(self.X, theta), float)
            # Allow (n, d_theta) when q==1
            if J.ndim == 2 and self.q == 1 and J.shape == (self.n, self.d_theta):
                J = J.reshape(self.n, 1, self.d_theta)
            if J.ndim != 3:
                raise ValueError(
                    f"g_jac must return an array of shape (n,q,d_theta); got {J.shape}"
                )
            if J.shape != (self.n, self.q, self.d_theta):
                raise ValueError(
                    f"g_jac returned {J.shape}, expected ({self.n},{self.q},{self.d_theta})"
                )
            return J

        # Numerical Jacobian (central differences on G)
        eps = float(self.outer_fd_eps)
        if not (eps > 0.0 and np.isfinite(eps)):
            raise ValueError("outer_fd_eps must be positive for numerical Jacobian.")

        J = np.empty((self.n, self.q, self.d_theta), dtype=float)
        for k in range(self.d_theta):
            h = eps * max(1.0, float(abs(theta[k])))
            tp = theta.copy(); tp[k] += h
            tm = theta.copy(); tm[k] -= h
            Gp = self._G(tp)
            Gm = self._G(tm)
            J[:, :, k] = (Gp - Gm) / (2.0 * h)
        return J

    def _gmm_pilot_start(
        self,
        theta0: Optional[np.ndarray] = None,
        method: str = "BFGS",
        options: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if theta0 is None:
            theta0 = self.theta0
        theta0 = np.asarray(theta0, dtype=float).reshape(-1)

        def fun(theta_vec: np.ndarray) -> float:
            G = self._G(theta_vec)
            m = G.mean(axis=0)
            return 0.5 * float(m @ m)

        res = minimize(
            fun,
            theta0,
            method=method,
            jac=None,
            options=options or {"maxiter": 200, "gtol": 1e-6},
        )
        return np.asarray(res.x, dtype=float).reshape(-1)

    def _gmm_pilot_start_per_v(
        self,
        v: np.ndarray,
        theta0: np.ndarray,
        options: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        v-weighted GMM pilot: minimize 0.5 * || G(theta)^T v ||^2.
        Used only as a warm start (does not change the ETEL objective).
        """
        theta0 = np.asarray(theta0, float).reshape(-1)
        v = np.asarray(v, float).reshape(-1)

        def fun(theta_vec: np.ndarray) -> float:
            G = self._G(theta_vec)
            m = G.T @ v
            return 0.5 * float(m @ m)

        res = minimize(
            fun,
            theta0,
            method="BFGS",
            jac=None,
            options=options or {"maxiter": 200, "gtol": 1e-6},
        )
        x = np.asarray(res.x, float).reshape(-1)
        if not np.all(np.isfinite(x)):
            return theta0
        return x

    def _etel_start(
        self,
        theta0: np.ndarray | None = None,
        method: str = "L-BFGS-B",
        options: dict | None = None
    ) -> np.ndarray:
        if theta0 is None:
            theta0 = self.theta0
        v_uniform = np.full(self.n, 1.0 / self.n, dtype=float)
        out = self.fit_per_v(v=v_uniform, theta0=np.asarray(theta0, float),
                             method=method, options=options)
        return np.asarray(out["theta"], float).reshape(-1)

    # ---------- Inner λ solvers ----------

    def _scale_G_columns(self, G: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        v-weighted column scaling:
            scale_j = sqrt( sum_i v_i * G_{ij}^2 )
        Returns (G_scaled, scale).
        """
        var = (v[:, None] * (G ** 2)).sum(axis=0)
        scale = np.sqrt(np.clip(var, 1e-12, None))
        return G / scale, scale

    def _compute_p_r(self, G: np.ndarray, v: np.ndarray, lam: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Returns (logZ, p, r) for given λ:
          p_i = v_i exp(g_i'λ)/Z
          r   = sum_i p_i g_i
        """
        scores = G @ lam
        logZ = _logsumexp_weighted(scores, v)
        p = v * np.exp(scores - logZ)
        r = (p[:, None] * G).sum(axis=0)
        return float(logZ), p, r

    def _lambda_to_w(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        w = np.zeros_like(v, dtype=float)
        mask = v > 0
        w[mask] = p[mask] / v[mask]
        return w

    def _solve_lambda_lbfgsb(
        self,
        G: np.ndarray,
        v: np.ndarray,
        lam0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, np.ndarray, Dict[str, Any]]:
        """
        Robust bounded convex solve of:
            min_λ log sum_i v_i exp( g_i' λ )
        using L-BFGS-B with analytic gradient.
        """
        v = np.asarray(v, float).reshape(-1)
        N, q = G.shape

        # optional column scaling
        if self.use_column_scaling:
            Gs, scale = self._scale_G_columns(G, v)
        else:
            Gs, scale = G, np.ones(q, float)

        if lam0 is None:
            lam0_u = np.zeros(q, float)
        else:
            lam0_u = np.asarray(lam0, float).reshape(-1)

        # convert to scaled coordinates: lam_s = lam_u * scale
        lam0_s = lam0_u * scale
        lam0_s = _safe_clip(lam0_s, self.lam_max * float(np.max(scale)))

        # bounds in scaled space correspond to lam_u in [-lam_max, lam_max]
        bounds_s = [(-self.lam_max * float(scale[j]), self.lam_max * float(scale[j])) for j in range(q)]

        def fun(lam_s: np.ndarray) -> float:
            lam_u = lam_s / scale
            lam_u = _safe_clip(lam_u, self.lam_max)
            scores = G @ lam_u
            return _logsumexp_weighted(scores, v)

        def jac(lam_s: np.ndarray) -> np.ndarray:
            lam_u = lam_s / scale
            lam_u = _safe_clip(lam_u, self.lam_max)
            logZ, p, r = self._compute_p_r(G, v, lam_u)
            # chain rule: d/d lam_s = (d/d lam_u) * (d lam_u / d lam_s) = r * (1/scale)
            return r / scale

        res = _minimize(
            fun,
            lam0_s,
            method="L-BFGS-B",
            jac=jac,
            bounds=bounds_s,
            options={
                "maxiter": max(self.newton_maxiter, 50),
                "ftol": 1e-12,
                "gtol": self.newton_tol,
                "maxls": 50,
            },
        )

        lam_s = np.asarray(res.x, float).reshape(-1)
        lam_u = lam_s / scale
        lam_u = _safe_clip(lam_u, self.lam_max)

        logZ, p, r = self._compute_p_r(G, v, lam_u)
        rn = float(np.linalg.norm(r, 2))
        w = self._lambda_to_w(p, v)

        info: Dict[str, Any] = {
            "method": "L-BFGS-B",
            "opt_success": bool(res.success),
            "message": str(res.message),
            "iters": int(getattr(res, "nit", 0)),
            "residual_norm": rn,
            # define "converged" ourselves:
            "converged": bool(np.isfinite(logZ) and np.all(np.isfinite(lam_u)) and rn <= self.res_tol),
            "used_penalty": False,
            "feasible": bool(rn <= self.res_tol),
        }
        return lam_u, float(logZ), w, info

    def _solve_lambda_sc(
        self,
        G: np.ndarray,
        v: np.ndarray,
        lam0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, np.ndarray, Dict[str, Any]]:
        """
        Self-concordant damped Newton on f(λ)=log sum_i v_i exp(λ^T g_i).
        Clipped at lam_max for numerical safety.
        """
        v = np.asarray(v, float).reshape(-1)
        N, q = G.shape
        lam = np.zeros(q, float) if lam0 is None else np.asarray(lam0, float).reshape(-1).copy()
        lam = _safe_clip(lam, self.lam_max)

        ridge = max(self.newton_ridge, 1e-12)
        info: Dict[str, Any] = {"method": "SC-Newton", "iters": 0, "converged": False, "residual_norm": None}

        for it in range(self.newton_maxiter):
            logZ, p, r = self._compute_p_r(G, v, lam)
            rn = float(np.linalg.norm(r, 2))
            info["iters"] = it
            info["residual_norm"] = rn

            if rn <= self.newton_tol:
                w = self._lambda_to_w(p, v)
                info["converged"] = True
                return lam, float(logZ), w, info

            # Hessian: Cov_p(g)
            H = G.T @ (p[:, None] * G) - np.outer(r, r)
            H = 0.5 * (H + H.T)

            ok = False
            local_ridge = ridge
            for _ in range(8):
                try:
                    L = np.linalg.cholesky(H + local_ridge * np.eye(q))
                    ok = True
                    break
                except np.linalg.LinAlgError:
                    local_ridge *= 10.0

            if not ok:
                step = -np.linalg.pinv(H + local_ridge * np.eye(q)) @ r
            else:
                y = np.linalg.solve(L, -r)
                step = np.linalg.solve(L.T, y)

            if not np.all(np.isfinite(step)):
                break

            grad_dot_step = float(r @ step)
            dec2 = float(-grad_dot_step)

            if dec2 <= 0.0 or dec2 * 0.5 <= self.newton_tol:
                w = self._lambda_to_w(p, v)
                info["converged"] = True
                return lam, float(logZ), w, info

            t = 1.0 / (1.0 + np.sqrt(dec2))
            f = float(logZ)

            for _ in range(self.max_ls_steps):
                lam_try = lam + t * step
                lam_try = _safe_clip(lam_try, self.lam_max)
                scores_t = G @ lam_try
                logZ_t = _logsumexp_weighted(scores_t, v)
                if np.isfinite(logZ_t) and logZ_t <= f + self.armijo_c * t * grad_dot_step:
                    lam = lam_try
                    break
                t *= self.armijo_beta
            else:
                break

            ridge = max(self.newton_ridge, local_ridge * 0.3)

        # best-effort
        logZ, p, r = self._compute_p_r(G, v, lam)
        info["residual_norm"] = float(np.linalg.norm(r, 2))
        w = self._lambda_to_w(p, v)
        return lam, float(logZ), w, info

    def _solve_lambda_penalized(
        self,
        G: np.ndarray,
        v: np.ndarray,
        lam0: Optional[np.ndarray],
        rho: float,
    ) -> Tuple[np.ndarray, float, np.ndarray, Dict[str, Any]]:
        """
        Penalized Newton for φ_ρ(λ)=logZ(λ)+(ρ/2)||r(λ)||^2.
        Clipped at lam_max for numerical safety.
        """
        v = np.asarray(v, float).reshape(-1)
        N, q = G.shape
        lam = np.zeros(q, float) if lam0 is None else np.asarray(lam0, float).reshape(-1).copy()
        lam = _safe_clip(lam, self.lam_max)

        info: Dict[str, Any] = {"method": "Penalized-Newton", "iters": 0, "converged": False, "penalty_rho": float(rho)}

        for it in range(self.newton_maxiter):
            logZ, p, r = self._compute_p_r(G, v, lam)
            rn = float(np.linalg.norm(r, 2))

            # J = Cov_p(g)
            J = G.T @ (p[:, None] * G) - np.outer(r, r)
            np.fill_diagonal(J, J.diagonal() + self.newton_ridge)

            grad = r + rho * (J @ r)

            H = J + rho * (J @ J)
            np.fill_diagonal(H, H.diagonal() + 1e-8)

            info["iters"] = it
            info["residual_norm"] = rn

            if rn <= self.res_tol:
                info["converged"] = True
                w = self._lambda_to_w(p, v)
                return lam, float(logZ), w, info

            try:
                step = np.linalg.solve(H, -grad)
            except np.linalg.LinAlgError:
                step = -np.linalg.pinv(H) @ grad

            alpha = 1.0
            base = rn
            accepted = False
            for _ in range(20):
                lam_try = lam + alpha * step
                lam_try = _safe_clip(lam_try, self.lam_max)
                _, p_t, r_t = self._compute_p_r(G, v, lam_try)
                if float(np.linalg.norm(r_t, 2)) <= (1 - 1e-1 * alpha) * base:
                    lam = lam_try
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                lam = _safe_clip(lam + 1e-2 * step / max(1.0, np.linalg.norm(step)), self.lam_max)

        # finalize
        logZ, p, r = self._compute_p_r(G, v, lam)
        info["residual_norm"] = float(np.linalg.norm(r, 2))
        w = self._lambda_to_w(p, v)
        return lam, float(logZ), w, info

    def _solve_lambda(
        self,
        G: np.ndarray,
        v: np.ndarray,
        lam0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, np.ndarray, Dict[str, Any]]:
        """
        v2.1 controller:
          1) SC Newton (optional fast path) on scaled G, with clipping.
          2) robust bounded L-BFGS-B always as fallback.
          3) penalized Newton only if enable_penalized=True and max_penalty_tries>0,
             and only when residual is still large.
        """
        v = np.asarray(v, float).reshape(-1)
        if v.shape[0] != G.shape[0]:
            raise ValueError("v length must match number of rows in G.")
        if np.any(v < 0) or not np.isfinite(v).all():
            raise ValueError("v must be nonnegative and finite.")

        # sanitize warm start
        if lam0 is None or (not np.all(np.isfinite(lam0))):
            lam0_u = np.zeros(self.q, float)
        else:
            lam0_u = _safe_clip(np.asarray(lam0, float).reshape(-1), self.lam_max)

        best = None  # (rn, lam, logZ, w, info)

        # (1) SC Newton fast path (on scaled G)
        if self.use_sc_newton:
            if self.use_column_scaling:
                Gs, scale = self._scale_G_columns(G, v)
            else:
                Gs, scale = G, np.ones(self.q, float)
            lam0_s = lam0_u * scale
            lam0_s = _safe_clip(lam0_s, self.lam_max * float(np.max(scale)))
            lam_s, logZ_s, w_s, info_s = self._solve_lambda_sc(Gs, v, lam0_s)
            lam_u = lam_s / scale
            lam_u = _safe_clip(lam_u, self.lam_max)

            # recompute w/logZ on original G to avoid any mismatch
            logZ_u, p_u, r_u = self._compute_p_r(G, v, lam_u)
            rn = float(np.linalg.norm(r_u, 2))
            w_u = self._lambda_to_w(p_u, v)
            info_s = dict(info_s)
            info_s["residual_norm"] = rn
            info_s["feasible"] = bool(rn <= self.res_tol)
            info_s["converged"] = bool(np.isfinite(logZ_u) and np.all(np.isfinite(lam_u)) and rn <= self.res_tol)
            best = (rn, lam_u, float(logZ_u), w_u, info_s)

            if info_s["converged"]:
                return lam_u, float(logZ_u), w_u, info_s

        # (2) Robust bounded L-BFGS-B (always)
        lam_l, logZ_l, w_l, info_l = self._solve_lambda_lbfgsb(G, v, lam0=lam0_u)
        rn_l = info_l.get("residual_norm", np.inf)

        if best is None or rn_l < best[0]:
            best = (rn_l, lam_l, logZ_l, w_l, info_l)

        if info_l.get("converged", False):
            return lam_l, logZ_l, w_l, info_l

        # (3) Penalized Newton opt-in
        if self.enable_penalized and self.max_penalty_tries > 0 and rn_l > self.penalty_resid_threshold:
            rho = self.penalty_rho0
            lam_ws = best[1]
            for _ in range(self.max_penalty_tries):
                lam_p, logZ_p, w_p, info_p = self._solve_lambda_penalized(G, v, lam_ws, rho)
                rn_p = info_p.get("residual_norm", np.inf)
                if rn_p < best[0]:
                    best = (rn_p, lam_p, logZ_p, w_p, info_p)
                if rn_p <= self.res_tol and np.all(np.isfinite(lam_p)) and np.isfinite(logZ_p):
                    info_p["feasible"] = True
                    info_p["converged"] = True
                    info_p["used_penalty"] = True
                    return lam_p, float(logZ_p), w_p, info_p
                lam_ws = lam_p
                rho *= self.penalty_rho_up

        # best effort
        rn_b, lam_b, logZ_b, w_b, info_b = best
        info_b = dict(info_b)
        info_b["feasible"] = bool(rn_b <= self.res_tol)
        info_b.setdefault("used_penalty", False)
        return lam_b, float(logZ_b), w_b, info_b

    # ---------- Loss / outer θ optimization ----------

    def _update_lambda_ws(self, lam_star: np.ndarray, inner_info: Dict[str, Any]) -> None:
        """
        Safe warm-start update: never propagate a bad/near-boundary λ.
        """
        rn = float(inner_info.get("residual_norm", np.inf))
        if (not np.all(np.isfinite(lam_star))) or (np.max(np.abs(lam_star)) > 0.95 * self.lam_max) or (rn > 1e6):
            self._lambda_ws = np.zeros(self.q, float)
        else:
            self._lambda_ws = np.asarray(lam_star, float).reshape(-1).copy()

    def _loss_only(
        self,
        theta: np.ndarray,
        v: np.ndarray,
    ) -> Tuple[float, Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]]:
        """
        Loss(theta) = - sum_i v_i log w_i*(theta), with
          log w_i* = λ^T g_i - logZ.

        Returns (loss, (lambda*, w*, logZ, info)).
        """
        theta = np.asarray(theta, float).reshape(-1)

        try:
            G = self._G(theta)  # (N, q)
            lam_star, logZ, w_star, inner_info = self._solve_lambda(G, v, lam0=self._lambda_ws)
            self._update_lambda_ws(lam_star, inner_info)

            # L = - lam' (G'v) + (sum v) logZ
            v_sum = float(np.sum(v))
            Gv = G.T @ v
            loss = -float(lam_star @ Gv) + v_sum * float(logZ)

            if self.loss_resid_penalty > 0.0:
                rn = inner_info.get("residual_norm", None)
                if rn is not None and np.isfinite(rn):
                    loss += 0.5 * float(self.loss_resid_penalty) * float(rn ** 2)

            if not np.isfinite(loss):
                loss = self.outer_loss_big

            return float(loss), (lam_star, w_star, float(logZ), inner_info)

        except Exception as e:
            # Never let the outer optimizer crash from inner failures.
            inner_info = {"exception": repr(e), "converged": False, "feasible": False, "residual_norm": np.inf}
            lam_star = np.zeros(self.q, float)
            w_star = np.ones(self.n, float)
            logZ = 0.0
            return float(self.outer_loss_big), (lam_star, w_star, logZ, inner_info)

    def _loss_and_grad(
        self,
        theta: np.ndarray,
        v: np.ndarray,
    ) -> Tuple[
        float,
        np.ndarray,
        Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]],
    ]:
        """Compute (loss, grad, inner_tuple) at a given (theta, v).

        This uses an implicit-differentiation gradient for the profiled
        ETEL-style objective. It requires either:
          - a user-supplied g_jac (preferred), or
          - a finite-difference approximation of the Jacobian of G(theta).

        Notes
        -----
        * If the inner lambda solver is at/near the lam_max bound, the
          implicit gradient is only approximate (the true optimum is then
          constrained). In that case we still return a finite vector so
          outer optimization can proceed.
        """
        theta = np.asarray(theta, float).reshape(-1)
        v = np.asarray(v, float).reshape(-1)

        # --- loss + inner quantities ---
        G = self._G(theta)  # (n, q)
        lam_star, logZ, w_star, inner_info = self._solve_lambda(G, v, lam0=self._lambda_ws)
        self._update_lambda_ws(lam_star, inner_info)

        v_sum = float(np.sum(v))
        Gv = G.T @ v
        loss = -float(lam_star @ Gv) + v_sum * float(logZ)
        if self.loss_resid_penalty > 0.0:
            rn = inner_info.get("residual_norm", None)
            if rn is not None and np.isfinite(rn):
                loss += 0.5 * float(self.loss_resid_penalty) * float(rn ** 2)
        if not np.isfinite(loss):
            loss = self.outer_loss_big

        # --- gradient ---
        # p_i = v_i w_i ; r = sum p_i g_i
        p = v * w_star
        r = (G.T @ p).reshape(-1)  # (q,)

        # Jacobian dG/dtheta: (n, q, d)
        Jg = self._G_jac(theta)

        # D_v = sum v_i d g_i/dtheta ; D_p = sum p_i d g_i/dtheta
        D_v = np.tensordot(v, Jg, axes=(0, 0))  # (q, d)
        D_p = np.tensordot(p, Jg, axes=(0, 0))  # (q, d)

        # t_i = lambda' * d g_i/dtheta  -> (n, d)
        t = np.einsum("q,nqd->nd", lam_star.reshape(-1), Jg)

        # S = sum p_i g_i * t_i' = (G' diag(p)) @ t
        S = (G.T @ (p[:, None] * t))  # (q, d)

        dr = D_p + S  # (q, d)

        # J = Cov_p(g) = sum p_i g_i g_i' - r r'
        Cov = (G.T @ (p[:, None] * G)) - np.outer(r, r)
        ridge = float(self.outer_grad_ridge)
        if ridge > 0.0:
            Cov = Cov + ridge * np.eye(self.q)

        # u = Cov^{-1} * Gv
        try:
            u = np.linalg.solve(Cov, Gv)
        except np.linalg.LinAlgError:
            # fall back to least squares (still yields a direction)
            u = np.linalg.lstsq(Cov, Gv, rcond=None)[0]

        grad = (dr.T @ u) + (lam_star.reshape(-1) @ (D_p - D_v))
        grad = np.asarray(grad, float).reshape(-1)
        if grad.shape != (self.d_theta,):
            raise ValueError(f"Gradient shape {grad.shape} != ({self.d_theta},)")
        grad[~np.isfinite(grad)] = 0.0

        return float(loss), grad, (lam_star, w_star, float(logZ), inner_info)

    # ---------- Public API ----------

    def fit_per_v(
        self,
        v: np.ndarray,
        method: str = "L-BFGS-B",
        options: Optional[Dict[str, Any]] = None,
        theta0: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Minimize Loss(theta) for a fixed v using SciPy's minimize.
        """
        self._lambda_ws = np.zeros(self.q, dtype=float)

        if theta0 is None:
            theta0 = self.theta0
        theta0 = np.asarray(theta0, dtype=float)

        if self.per_v_gmm_start:
            theta0 = self._gmm_pilot_start_per_v(v=v, theta0=theta0)

        cache: Dict[str, Any] = {}

        if self.outer_use_analytic_grad:

            def fun_and_grad(theta_vec: np.ndarray):
                loss, grad, inner = self._loss_and_grad(theta_vec, v)
                cache["last"] = (theta_vec.copy(), *inner)
                return float(loss), grad

            res = minimize(
                fun_and_grad,
                theta0,
                method=method,
                jac=True,
                bounds=self.bounds,
                options=options,
            )

        else:

            def fun(theta_vec: np.ndarray) -> float:
                loss, inner = self._loss_only(theta_vec, v)
                cache["last"] = (theta_vec.copy(), *inner)
                return float(loss)

            res = minimize(
                fun,
                theta0,
                method=method,
                jac="2-point",
                bounds=self.bounds,
                options=options,
            )

        if "last" in cache:
            _, lam_star, w_star, logZ, inner_info = cache["last"]
        else:
            _, inner = self._loss_only(res.x, v)
            lam_star, w_star, logZ, inner_info = inner

        final_loss, _ = self._loss_only(res.x, v)

        return {
            "theta": np.asarray(res.x, float),
            "loss": float(final_loss),
            "lambda_star": np.asarray(lam_star, float),
            "success": bool(res.success),
            "weights": np.asarray(w_star, float),
            "opt_result": res,
            "inner_info": inner_info,
        }

    def bayesian_bootstrap(self) -> np.ndarray:
        """Return B x N matrix of Bayesian bootstrap weights v."""
        v = self.rng.dirichlet(alpha=np.ones(self.n), size=self.B)
        v = np.clip(v, 1e-12, None)
        v /= v.sum(axis=1, keepdims=True)
        return v

    def draw_dirichlet(self, seed: int | None = None) -> np.ndarray:
        rng = self.rng if seed is None else np.random.default_rng(int(seed))
        v = rng.dirichlet(alpha=np.ones(self.n))
        v = np.clip(v, 1e-12, None)
        v /= v.sum()
        return v

    def fit(
        self,
        store_weights: bool = False,
        store_b_weights: bool = False,
        maxiter: int = 200,
        ftol: float = 1e-8,
        eps: float = 1e-6,
        method: str = "L-BFGS-B",
        gtol: float = 1e-6,
        options: Optional[Dict[str, Any]] = None,
    ) -> GenELResult:
        """Fit with Bayesian bootstrap (alpha = 0)."""
        if self.alpha != 0:
            raise NotImplementedError("GenAI version (alpha > 0) not implemented yet.")

        if options is None:
            m = str(method).upper()
            if m in {"BFGS", "CG", "NEWTON-CG"}:
                options = {"maxiter": int(maxiter), "gtol": float(gtol)}
            else:
                options = {"maxiter": int(maxiter), "ftol": float(ftol), "eps": float(eps)}

        thetas = np.zeros((self.B, self.d_theta))
        losses = np.zeros(self.B, dtype=float)
        lambdas = np.zeros((self.B, self.q))
        weights = np.zeros((self.B, self.n)) if store_weights else None
        successes = np.zeros(self.B, dtype=bool)
        b_weights = np.zeros((self.B, self.n)) if store_b_weights else None

        with tqdm(total=self.B, desc="Bootstrap", unit="draw") as bar:
            for i in range(self.B):
                v = self.draw_dirichlet()
                out = self.fit_per_v(v=v, method=method, options=options)
                thetas[i] = out["theta"]
                losses[i] = out["loss"]
                lambdas[i] = out["lambda_star"]
                if store_weights:
                    weights[i] = out["weights"]
                if store_b_weights:
                    b_weights[i] = v
                successes[i] = out["success"]
                bar.update(1)

        return GenELResult(
            thetas=thetas,
            loss=losses,
            lambda_stars=lambdas,
            weights=weights,
            b_weights=b_weights,
            success=successes,
        )

    def fit_in_parallel(
        self,
        max_workers: int | None = None,
        method: str = "L-BFGS-B",
        options: dict | None = None,
        store_weights: bool = False,
        store_b_weights: bool = False,
    ) -> GenELResult:
        """
        Parallel version using per-draw seeds instead of shipping v arrays.
        Deterministic given self.random_state and B.
        """
        if self.alpha != 0:
            raise NotImplementedError("GenAI version (alpha > 0) not implemented yet.")

        thetas = np.zeros((self.B, self.d_theta))
        losses = np.zeros(self.B, dtype=float)
        lambdas = np.zeros((self.B, self.q))
        weights = np.zeros((self.B, self.n)) if store_weights else None
        successes = np.zeros(self.B, dtype=bool)
        b_weights = np.zeros((self.B, self.n)) if store_b_weights else None

        base_ss = np.random.SeedSequence(int(getattr(self, "random_state", 42)))
        child_ss = base_ss.spawn(self.B)
        seeds = [int(s.generate_state(1)[0]) for s in child_ss]

        initargs = (
            self.X,
            self.g_fun,
            self.theta0,
            self.g_jac_fun,
            self.bounds,
            self.newton_tol,
            self.newton_maxiter,
            self.newton_ridge,
            self.use_sc_newton,
            self.enable_penalized,
            self.lam_max,
            self.per_v_gmm_start,
            self.outer_use_analytic_grad,
            self.outer_fd_eps,
            self.outer_grad_ridge,
        )
        opts = options or {"maxiter": 200, "ftol": 1e-8, "eps": 1e-6}

        if max_workers is None:
            max_workers = os.cpu_count() or 1
        chunksize = max(1, self.B // (4 * max_workers))

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=initargs,
        ) as ex:
            tasks = ((i, seeds[i], method, opts) for i in range(self.B))
            for i, out in tqdm(
                ex.map(_solve_one, tasks, chunksize=chunksize),
                total=self.B, desc="Bootstrap", unit="draw"
            ):
                thetas[i] = out["theta"]
                losses[i] = out["loss"]
                lambdas[i] = out["lambda_star"]
                if store_weights:
                    weights[i] = out["weights"]
                successes[i] = out["success"]
                if store_b_weights:
                    b_weights[i] = self.draw_dirichlet(seeds[i])

        return GenELResult(
            thetas=thetas,
            loss=losses,
            lambda_stars=lambdas,
            weights=weights,
            b_weights=b_weights,
            success=successes,
        )


