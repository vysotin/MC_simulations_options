from dataclasses import dataclass
from typing import Tuple

import numpy as np
from typing import Callable
from scipy import stats as st
RND_SEED = 42

@dataclass
class SimulationModel:
    def __init__(self, name: str, model: Callable, extra_args: dict):
        self.name: str = name
        self.model: Callable = model
        self.extra_args: dict = extra_args

def european_option_price(option_type: str, simulation_model: SimulationModel, K: float, S0: float,  r: float,
                          T: float, steps: int,
                          paths: int, confidence_level: float=0.95) -> float:
    """
    Parameters:
    option_type (str): 'call' or 'put'
    simulation_model (SimulationModel): a dataclass containing the model name, model function and extra arguments
    K (float): strike price
    S0 (float): initial stock price
    r (float): risk-free rate (%)
    T (float): time to maturity (years)
    steps (int): number of time steps
    paths (int): number of simulations
    confidence_level (float, optional): confidence level for confidence interval. Defaults to 0.95.
    Returns:
    mean (float): mean of option price
    std_error (float): standard deviation of option price
    ci_lower (float): lower limit of confidence interval
    ci_upper (float): upper limit of confidence interval
    """

    S = simulation_model.model(S0, r, T, steps, paths, **simulation_model.extra_args)
    S = S.T
    # Calculate the payoff for each path
    discount_factor = np.exp(-r * T)
    payoff = np.maximum(S[:, -1] - K, 0)
    std_error= np.std(payoff) * discount_factor / np.sqrt(steps)
    # Discount the payoff back to present value
    if option_type == 'call':
        payoff = np.maximum(S[:, -1] - K, 0)
    elif option_type == 'put':
        payoff = np.maximum(K - S[:, -1], 0)
    option_price = discount_factor * np.mean(payoff)
    z_score = st.norm.ppf(1 - (1 - confidence_level) / 2)
    margin_error = z_score * std_error
    ci_lower = option_price - margin_error
    ci_upper = option_price + margin_error
    return option_price, std_error, ci_lower, ci_upper

def gbm_model_fun(S0: float, r: float, T: float, steps: int, paths: int, sigma: float) -> float:
    # S0: initial stock price
    # sigma: volatility (%)
    # r: risk-free rate (%)
    # T: time to maturity (years)
    # steps: number of time steps
    # paths: number of simulations
    # returns: simulated prices for underlying

    dt = T / steps
    np.random.seed(RND_SEED)
    S = np.zeros((steps + 1, paths))
    S[0] = S0
    for t in range(1, steps + 1):
        Z = np.random.standard_normal(paths) # Generate random variables
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return S

def merton_jump_diffusion_model_fun(S0, r, T, paths, steps,  sigma, lambda_, kappa) -> np.ndarray:
    dt = T / steps
    S = np.zeros((steps + 1, paths))
    S[0] = S0
    for t in range(1, steps + 1):
        Z = np.random.normal(0, 1, paths)
        J = np.random.poisson(lambda_ * dt, paths)
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + (J *(np.exp(kappa) - 1)))
    return S

def double_exp_jump_model_fun(S0, r, T, paths, steps, sigma, lambda_, eta1, eta2, p) -> np.ndarray:
    dt = T / steps
    S = np.zeros((steps + 1, paths))
    S[0] = S0
    for t in range(1, steps + 1):
        Z = np.random.normal(0, 1, paths)
        J = np.random.poisson(lambda_ * dt, paths)
        jump_sizes = np.where(np.random.rand(paths) < p, np.random.exponential(eta1, paths), -
        np.random.exponential(eta2, paths))
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z + J * jump_sizes)
    return S

def heston_model_fun(S0, r, T, paths, steps, v0=0.04, kappa=2.0, theta=0.04, xi=0.1, rho=-0.7) -> np.ndarray:
    """
    Heston model for stock price.
    :param S0:
    :param r:
    :param T:
    :param paths:
    :param steps:
    :param v0: initial variance
    :param kappa: kappa parameter
    :param theta:
    :param xi:
    :param rho:
    :return:
    """
    dt = T / steps
    S = np.zeros((steps + 1, paths))
    v = np.zeros((steps + 1, paths))
    S[0] = S0
    v[0] = v0
    for t in range(1, steps + 1):
        Z1 = np.random.normal(0, 1, paths)
        Z2 = np.random.normal(0, 1, paths)
        W1 = Z1 * np.sqrt(dt)
        W2 = rho * Z1 * np.sqrt(dt) + np.sqrt(1 - rho**2) * Z2 * np.sqrt(dt)
        v[t] = v[t-1] + kappa * (theta - v[t-1]) * dt + xi * np.sqrt(v[t-1]) * W2
        S[t] = S[t-1] * np.exp((r - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1]) * W1)
    return S


def svcj_model_fun(S0, r, T, paths, steps, v0, kappa, theta, sigma_v, rho, lambda_, mu_y, sigma_y, rho_j,
                   mu_v) -> np.ndarray:
    """
    Simulates asset price paths using the Stochastic Volatility with Correlated Jumps (SVCJ) model.

    This implementation is based on the model described by Duffie, Pan, and Singleton (2000)
    and used by Hou et al. (2020) for cryptocurrency options. The model includes jumps
    in both asset returns and volatility, which can be correlated.

    The continuous-time dynamics are[cite: 143]:
    d log(S_t) = (mu - 0.5 * V_t) dt + sqrt(V_t) dW_t^(S) + Z_t^y dN_t
    d V_t = kappa * (theta - V_t) dt + sigma_v * sqrt(V_t) dW_t^(V) + Z_t^v dN_t

    Where:
    - S_t is the asset price, V_t is the variance.
    - W_t^(S), W_t^(V) are correlated Brownian motions with correlation rho.
    - N_t is a Poisson process with intensity lambda_.
    - Z_t^y and Z_t^v are the jump sizes for log-price and variance, respectively.
    - The variance jump Z_t^v follows an exponential distribution with mean mu_v[cite: 153].
    - The price jump Z_t^y is conditionally normal: Z_t^y | Z_t^v ~ N(mu_y + rho_j * Z_t^v, sigma_y^2)[cite: 152, 154].

    :param S0: Initial asset price.
    :param r: The drift rate of the asset (risk-free rate for risk-neutral simulation)[cite: 165].
    :param T: Time to maturity in years.
    :param paths: Number of simulation paths.
    :param steps: Number of time steps.
    :param v0: Initial variance.
    :param kappa: Mean-reversion rate for variance (0.132)[-beta in SVCJ paper, p257]
    :param theta: Long-run mean of variance (0.076)[-alpha/beta in SVCJ paper, p257]
    :param sigma_v: Volatility of volatility (0.008)[-alpha/beta in SVCJ paper, p257]
    :param rho: Correlation between the two Brownian motions for price and variance[cite: 150, 156].
    :param lambda_: The mean jump-arrival rate (annualized) (0.041)[CVCJ paper, p.257].
    :param mu_y: The mean of the price jump component (-0.084)[SVCJ paper, p257]
    :param sigma_y: The standard deviation of the price jump component (2.155)[SVCJ paper, p257]
    :param rho_j: The correlation between the price jump and the volatility jump (-0.573)[SVCJ paper, p257]
    :param mu_v: The mean of the exponential jump size in variance (0.620)[SVCJ paper, p257]
    :return: A numpy array of simulated asset price paths.
    """
    dt = T / steps
    S = np.zeros((steps + 1, paths))
    v = np.zeros((steps + 1, paths))
    S[0] = S0
    v[0] = v0

    # Pre-generate random numbers for efficiency
    Z1 = np.random.normal(size=(steps, paths))
    Z2 = np.random.normal(size=(steps, paths))
    poisson_shocks = np.random.poisson(lambda_ * dt, size=(steps, paths))
    exp_shocks = np.random.exponential(scale=mu_v, size=(steps, paths))
    jump_corr_shocks = np.random.normal(size=(steps, paths))

    for t in range(1, steps + 1):
        # Ensure variance is non-negative (full truncation scheme)
        v[t - 1] = np.maximum(v[t - 1], 0)
        sqrt_v = np.sqrt(v[t - 1])

        # --- JUMP Component ---
        # A jump occurs if the Poisson shock is greater than 0
        jump_event = (poisson_shocks[t - 1] > 0)

        # Calculate jump sizes. If no jump, sizes are 0.
        jump_v = jump_event * exp_shocks[t - 1]
        jump_y = jump_event * (mu_y + rho_j * jump_v + sigma_y * jump_corr_shocks[t - 1])

        # --- DIFFUSION Component ---
        # Correlated Wiener processes
        W1 = Z1[t - 1] * np.sqrt(dt)
        W2 = (rho * Z1[t - 1] + np.sqrt(1 - rho ** 2) * Z2[t - 1]) * np.sqrt(dt)

        # Variance process update
        dv = kappa * (theta - v[t - 1]) * dt + sigma_v * sqrt_v * W2 + jump_v
        v[t] = v[t - 1] + dv

        # Asset price process update
        d_log_S = (r - 0.5 * v[t - 1]) * dt + sqrt_v * W1 + jump_y
        S[t] = S[t - 1] * np.exp(d_log_S)

    return S


if __name__ == '__main__':
    #sim_model = SimulationModel("GBM", gbm_model_fun, {"sigma": 0.02})
    sim_model = SimulationModel("Heston", heston_model_fun, dict(v0=0.01, kappa=1.0, theta=0.04, xi=0.1, rho=-0.7))

    print(european_option_price("call", sim_model, S0=98000, K=10800, r=3.0, T=100, steps=100,  paths=100000))
    #print(european_put(100, 100, 0.2, 0.05, 1, 100, 10000))