import time
import numpy as np
from scipy.stats import skewnorm
import pickle
import EL 


def simulate_data(n: int = 250, seed: int = 42, theta0: float = 1.0, theta1: float = 1.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 2.5, size=n)

    h = np.sqrt(np.exp(1.0 + 0.7 * x + 0.2 * x**2))
    s = 1.0 + x**2
    delta = s / np.sqrt(1.0 + s**2)
    m = -h * np.sqrt(2.0 / np.pi) * delta  # ensures E[eps|X]=0

    eps = skewnorm.rvs(a=s, loc=m, scale=h, size=n, random_state=rng)
    y = theta0 + theta1 * x + eps
    return y, x


def ols_start(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    Xdesign = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(Xdesign, y, rcond=None)
    return beta.astype(float)


def g_residual(data: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Base (scalar) conditional moment:
      residual = y - theta0 - theta1*x
    """
    y = data[:, 0]
    x = data[:, 1]
    return y - theta[0] - theta[1] * x  


def g_residual_jac(data: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Jacobian of residual wrt theta=[theta0, theta1], shape (n, 1, 2):
      d/dtheta0 = -1
      d/dtheta1 = -x
    """
    x = data[:, 1]
    n = data.shape[0]
    J = np.zeros((n, 1, 2), dtype=float)
    J[:, 0, 0] = -1.0
    J[:, 0, 1] = -x
    return J


def summarize_draws(thetas: np.ndarray) -> dict:
    out = {}
    names = ["theta0", "theta1"]
    for j, nm in enumerate(names):
        a = thetas[:, j]
        out[nm] = {
            "mean": float(np.mean(a)),
            "sd": float(np.std(a, ddof=1)),
            "median": float(np.median(a)),
            "q05": float(np.quantile(a, 0.05)),
            "q95": float(np.quantile(a, 0.95)),
        }
    return out


def print_summary_table(stats: dict, K: int):
    print(f"Posterior summaries for K={K}:")
    header = f"{'param':<8}  {'mean':>10}  {'sd':>10}  {'median':>10}  {'q05':>10}  {'q95':>10}"
    print(header)
    print("-" * len(header))
    for nm in ["theta0", "theta1"]:
        s = stats[nm]
        print(
            f"{nm:<8}  "
            f"{s['mean']:>10.4f}  {s['sd']:>10.4f}  {s['median']:>10.4f}  "
            f"{s['q05']:>10.4f}  {s['q95']:>10.4f}"
        )
    print()


def build_model_for_K(
    y: np.ndarray,
    x: np.ndarray,
    K: int,
    B: int,
    random_state: int = 42,
):
    n = len(x)
    data = np.column_stack([y, x])
    W = x.reshape(-1, 1)

    theta_init = ols_start(y, x)

    df_first = max(3, K - 1)
    whiten_basis = True
    max_moment_ratio = 0.90
    add_constant = True

    model = EL.ConditionGenELV2(
        X=data,
        W=W,
        g=g_residual,
        g_jac=g_residual_jac,        
        theta0=theta_init,
        df_first=df_first,
        df_rest=3,
        center_basis=True,
        scale_basis=True,
        add_constant=add_constant,
        whiten_basis=whiten_basis,
        svd_tol=1e-10,
        max_moment_ratio=max_moment_ratio,
        init_theta=None,
        alpha=0.0,
        m=0,
        B_nums=B,
        random_state=random_state,
        bounds=None,
        # inner lambda solve
        newton_tol=1e-6,
        newton_maxiter=100,
        newton_ridge=1e-10,
        use_sc_newton=True,
        enable_penalized=False,
        lam_max=500.0,
        per_v_gmm_start=True,
        outer_use_analytic_grad=True,
    )
    return model


def run_experiment(n=250, B=1000, Ks=(3, 5), seed_data=42, seed_boot=42):
    y, x = simulate_data(n=n, seed=seed_data, theta0=1.0, theta1=1.0)
    beta_ols = ols_start(y, x)

    print(f"n={n}")
    print(f"True theta: theta0=1.0000, theta1=1.0000")
    print(f"OLS  theta: theta0={beta_ols[0]:.4f}, theta1={beta_ols[1]:.4f}")
    print(f"Bootstrap draws B={B}")
    print()

    result = {}
    for K in Ks:
        model = build_model_for_K(y=y, x=x, K=K, B=B, random_state=seed_boot)

        t0 = time.perf_counter()
        res = model.fit(
            store_weights=False,
            store_b_weights=False,
            maxiter=200,
            method="BFGS",
            gtol=1e-6,
            ftol=1e-9,
            eps=1e-6,
        )
        t1 = time.perf_counter()

        thetas = res.thetas
        result[K] = {"thetas": thetas, "success": res.success}

        stats = summarize_draws(thetas)
        elapsed = t1 - t0
        print(K)
        print(f"Time: {elapsed:.2f} sec  ({B/elapsed:.1f} draws/sec)")
        print_summary_table(stats, K=K)

        with open("GenEL_K_draws.pkl", "wb") as f:
            pickle.dump(result, f)


if __name__ == "__main__":
    run_experiment(n=250, B=50000, Ks=(3, 5, 10, 15, 20), seed_data=2026, seed_boot=2026)
