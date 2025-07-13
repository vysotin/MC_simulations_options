from dataclasses import dataclass
from typing import Tuple

import numpy as np
from jedi.inference.gradual.typing import Callable
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


if __name__ == '__main__':
    #sim_model = SimulationModel("GBM", gbm_model_fun, {"sigma": 0.02})
    sim_model = SimulationModel("Heston", heston_model_fun, dict(v0=0.01, kappa=1.0, theta=0.04, xi=0.1, rho=-0.7))

    print(european_option_price("call", sim_model, S0=98000, K=10800, r=3.0, T=100, steps=100,  paths=100000))
    #print(european_put(100, 100, 0.2, 0.05, 1, 100, 10000))