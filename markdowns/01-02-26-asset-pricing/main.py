from __future__ import annotations
import pickle
import time
import numpy as np
import pandas as pd
import EL  


def summarize_1d(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "median": float(np.median(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q95": float(np.quantile(x, 0.95)),
    }


def print_summary(name: str, s: dict):
    print(
        f"{name:>6}: "
        f"mean={s['mean']:+.6f}, sd={s['sd']:.6f}, median={s['median']:+.6f}, "
        f"q05={s['q05']:+.6f}, q95={s['q95']:+.6f}"
    )


def main(
    csv_path: str = "asset_pricing_data.csv",
    B: int = 500,
    seed: int = 123,
):
    # load data
    df = pd.read_csv(csv_path)
    cols = ["Mkt"] + [c for c in df.columns if c != "Mkt"]
    df = df[cols]
    f_all = df.values.astype(float)  # (540, 12)

    # Align samples
    X = f_all[1:, :]     # f_t   (539, 12)
    W = f_all[:-1, :]    # f_{t-1} (539, 12)
    N = X.shape[0]
    idx_mkt = 0
    x = X[:, idx_mkt]

    # Moment Conditions and Jacobian
    def g_uncond(Xmat: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Unconditional pricing moments (12):
            E[(1 - b*(x_t - mu_x)) * f_t] = 0
        """
        b = float(theta[0])
        mu = float(theta[1])
        xt = Xmat[:, idx_mkt]
        M = 1.0 - b * (xt - mu)     # (N,)
        return M[:, None] * Xmat    # (N, 12)

    def g_uncond_jac(Xmat: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Jacobian of g_uncond wrt theta=[b,mu], shape (N, 12, 2).
        d/db:  -(x-mu) * f_t
        d/dmu:  b * f_t
        """
        b = float(theta[0])
        mu = float(theta[1])
        xt = Xmat[:, idx_mkt]

        d_db = -(xt - mu)[:, None] * Xmat      # (N, 12)
        d_dmu = b * Xmat                       # (N, 12)

        J = np.zeros((Xmat.shape[0], Xmat.shape[1], 2), dtype=float)
        J[:, :, 0] = d_db
        J[:, :, 1] = d_dmu
        return J

    def g_cond(Xmat: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Base conditional restriction (scalar):
            E[x_t - mu_x | f_{t-1}] = 0
        MixedConditionalGenELV2 expands this into 25 unconditional moments via basis.
        """
        mu = float(theta[1])
        xt = Xmat[:, idx_mkt]
        return (xt - mu)  # (N,)

    def g_cond_jac(Xmat: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Jacobian of g_cond wrt theta=[b,mu], shape (N, 1, 2).
        d/db = 0
        d/dmu = -1
        """
        Nloc = Xmat.shape[0]
        J = np.zeros((Nloc, 1, 2), dtype=float)
        J[:, 0, 1] = -1.0
        return J


    # Initial guess: Use the market-only heuristic b0 = E[x]/Var(x), mu0=E[x]
    mu0 = float(np.mean(x))
    varx = float(np.var(x, ddof=0))
    b0 = float(mu0 / varx) if varx > 0 else 1.0
    theta0 = np.array([b0, mu0], dtype=float)
    print(f"Initial theta0 = [{theta0[0]:+.6f}, {theta0[1]:+.6f}]")

    # ---------------------------------------------------------------------
    #  Build MixedConditionalGenELV2
    #    Paper basis target: K = 3 + 11*(3-1) = 25, and q_total = 12 + 25 = 37
    # ---------------------------------------------------------------------
    model = EL.MixedConditionalGenELV2(
        X=X,
        W=W,
        g_uncond=g_uncond,
        g_cond=g_cond,
        g_uncond_jac=g_uncond_jac,
        g_cond_jac=g_cond_jac,
        theta0=theta0,
        # Basis to match the JRSSB construction
        df_first=3,
        df_rest=3,
        center_basis=False,
        scale_basis=False,
        add_constant=False,   
        whiten_basis=False,   
        svd_tol=1e-10,
        max_moment_ratio=0.9,
        # GenELV2 controls
        init_theta=None,
        alpha=0.0,
        m=0,
        B_nums=B,
        random_state=seed,
        bounds=None,
        # Inner lambda solver
        newton_tol=1e-6,
        newton_maxiter=100,
        newton_ridge=1e-8,
        use_sc_newton=True,
        enable_penalized=False,
        lam_max=500.0,
        per_v_gmm_start=False,
        outer_use_analytic_grad=True,
    )

    t0 = time.perf_counter()
    res = model.fit(
        store_weights=False,
        store_b_weights=False,
        maxiter=200,
        method="BFGS",
        gtol=1e-6,
        ftol=1e-8,
        eps=1e-6,
    )
    t1 = time.perf_counter()

    return res


if __name__ == "__main__":
    res= main(csv_path = "asset_pricing_data.csv", B= 50000, seed = 42)
    thetas = res.thetas
    b_draws = thetas[:, 0]
    mu_draws = thetas[:, 1]
    print_summary("b", summarize_1d(b_draws))
    print_summary("mu_x", summarize_1d(mu_draws))
    with open("ap_result.pkl", "wb") as f:
        pickle.dump(res, f)
