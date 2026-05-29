import numpy as np
import pandas as pd
import time
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
        f"{name:>10}: "
        f"mean={s['mean']:+.6f}, sd={s['sd']:.6f}, median={s['median']:+.6f}, "
        f"q05={s['q05']:+.6f}, q95={s['q95']:+.6f}"
    )

def sigmoid(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, float)
    out = np.empty_like(u)
    pos = u >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
    expu = np.exp(u[~pos])
    out[~pos] = expu / (1.0 + expu)
    return out


def load_lottery_design(
    path: str = "lottery_processed.csv",
    standardize: bool = True,
):
    df = pd.read_csv(path)

    # 12 regressors 
    X_raw = df[
        ["TB","YS","WT","EYB1","Age","SEYB5","YW","EYB5","YW_sq","TB_YW","TB_sq","WT_YW"]
    ].to_numpy(dtype=float)

    Y = df["Yi"].to_numpy(dtype=float)
    W = df["winner"].to_numpy(dtype=float)

    scaler = None
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_std = scaler.fit_transform(X_raw)
        X = np.column_stack([np.ones(len(df)), X_std])  # add intercept
    else:
        X = np.column_stack([np.ones(len(df)), X_raw])  # add intercept
    X_all = np.column_stack([X, Y, W])
    return X_all, X.shape[1], scaler



# Moment function + Jacobian
def make_ate_moments(p: int, eta_clip: float = 1e-6):
    """
    p: number of columns in X (INCLUDING intercept)
    Returns g(Xall, beta), g_jac(Xall, beta).
    beta = [gamma (p), tau]
    g returns (n, q) with q=p+1
    jac returns (n, q, p+1)
    """

    def g_ate(Xall: np.ndarray, beta: np.ndarray) -> np.ndarray:
        X = Xall[:, :p]
        Y = Xall[:, p]
        W = Xall[:, p + 1]

        beta = np.asarray(beta, float).reshape(-1)
        gamma = beta[:p]
        tau = float(beta[p])

        eta = sigmoid(X @ gamma)
        eta = np.clip(eta, eta_clip, 1.0 - eta_clip)
        denom = eta * (1.0 - eta)

        # Logistic score moments: X*(W - eta)
        m1 = X * (W - eta)[:, None]  # (n, p)

        # ATE IPW moment: (W-eta)Y/(eta(1-eta)) - tau
        m2 = ((W - eta) * Y) / denom - tau  # (n,)

        return np.column_stack([m1, m2])  # (n, p+1)

    def g_ate_jac(Xall: np.ndarray, beta: np.ndarray) -> np.ndarray:
        X = Xall[:, :p]
        Y = Xall[:, p]
        W = Xall[:, p + 1]

        beta = np.asarray(beta, float).reshape(-1)
        gamma = beta[:p]

        eta = sigmoid(X @ gamma)
        eta = np.clip(eta, eta_clip, 1.0 - eta_clip)
        denom = eta * (1.0 - eta)

        n = X.shape[0]
        J = np.zeros((n, p + 1, p + 1), dtype=float)

        # ---- First block derivatives: g1_k = X_k*(W-eta)
        J[:, :p, :p] = -(denom[:, None, None]) * (X[:, :, None] * X[:, None, :])
        # ---- Second block derivative wrt gamma:
        # g2 = ((W-eta)Y)/denom - tau
        # simplified derivative:
        # d/d gamma_j g2 = - Y * X_j * [ W*(1-2*eta) + eta^2 ] / denom
        numer = W * (1.0 - 2.0 * eta) + eta**2  # (n,)
        J[:, p, :p] = -(Y[:, None] * X) * (numer / denom)[:, None]

        # d/d tau g2 = -1
        J[:, p, p] = -1.0

        return J

    return g_ate, g_ate_jac



# Map standardized gamma back to original scale (not needed for now)
def unstandardize_gamma_draws(gamma_draws_std: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    gamma_draws_std: (B, p) where p = 1 + k, intercept + standardized regressors.
    scaler is fitted on the k raw regressors.
    Returns gamma_draws_raw: (B, p) in original regressor units.
    """
    if scaler is None:
        return gamma_draws_std

    means = scaler.mean_          # (k,)
    scales = scaler.scale_        # (k,)
    B, p = gamma_draws_std.shape
    k = len(means)
    assert p == 1 + k

    out = np.empty_like(gamma_draws_std)
    out[:, 1:] = gamma_draws_std[:, 1:] / scales[None, :]
    out[:, 0]  = gamma_draws_std[:, 0] - np.sum(gamma_draws_std[:, 1:] * (means / scales)[None, :], axis=1)
    return out



def run_ate_inference(
    csv_path: str = "lottery_processed.csv",
    draws: int = 10_000,
    seed: int = 42,
    standardize: bool = True,
):
    # 1) Load data
    X_all, p, scaler = load_lottery_design(csv_path, standardize=standardize)
    X = X_all[:, :p]
    Y = X_all[:, p]
    W = X_all[:, p + 1]

    # 2) Moments + Jacobian
    g_fun, g_jac = make_ate_moments(p=p, eta_clip=1e-6)

    # 3) Initial beta0: propensity MLE + implied tau
    lr = LogisticRegression(
        fit_intercept=False,
        penalty=None,
        solver="lbfgs",
        max_iter=5000,
    )
    lr.fit(X, W)
    gamma0 = lr.coef_.ravel()

    eta0 = np.clip(sigmoid(X @ gamma0), 1e-6, 1.0 - 1e-6)
    tau0 = float(np.mean(((W - eta0) * Y) / (eta0 * (1.0 - eta0))))
    beta0 = np.concatenate([gamma0, [tau0]])

    print("Initial beta0 (last element is tau0):")
    print(beta0)

    # 4) Build GenELV2
    model = EL.GenELV2(
        X=X_all,
        g=g_fun,
        g_jac=g_jac,
        theta0=beta0,
        alpha=0.0,
        m=0,
        B_nums=draws,
        random_state=seed,
        lam_max=500.0,
        per_v_gmm_start = True,
        outer_use_analytic_grad=True,
    )

    # 6) Posterior draws
    t0 = time.time()
    res = model.fit(
        store_weights=False,
        store_b_weights=False,
        method="BFGS",
        maxiter=200,
        gtol=1e-8,  
    )
    t1 = time.time()
    print(f"\nRun time: {t1 - t0:.2f} s")

    return res


if __name__ == "__main__":
    res= run_ate_inference(
        csv_path="lottery_processed.csv",
        draws=10000,
        seed=42,
        standardize=True,
    )
    thetas = res.thetas
    tau_draws = thetas[:, -1]
    print_summary("tau", summarize_1d(tau_draws))
    with open("ate_result.pkl", "wb") as f:
        pickle.dump(res, f)
