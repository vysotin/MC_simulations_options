from typing import Dict


import numpy as np
import pandas as pd
from py_vollib.black_scholes import implied_volatility
from statsmodels.tsa.ar_model import AutoReg

from simulation_model_base import SimulationModel
SEED = 234098
np.random.seed(SEED)

class SVCJ_Model(SimulationModel):


    """
    Implementation of the Stochastic Volatility with Correlated Jumps (SVCJ) model.

    This class provides methods to:
    1. Fit the SVCJ model parameters to historical data using the Simulated Method of Moments (SMM).
    2. Price European options using the fitted model via Monte Carlo simulation.
    3. Calculate major option Greeks (Delta, Gamma, Vega, Theta) using finite differences.

    The model dynamics are based on Hou et al. (2020), "Pricing Cryptocurrency Options".
    d logS_t = mu dt + sqrt(V_t) dW_t^(S) + Z_t^y dN_t
    d V_t   = kappa(theta - V_t) dt + sigma_v sqrt(V_t) dW_t^(V) + Z_t^v dN_t

    Parameters:
    - mu: Mean drift of the log-return process.
    - kappa: Mean-reversion rate for variance.
    - theta: Long-run mean of variance.
    - sigma_v: Volatility of volatility.
    - rho: Correlation between the two Brownian motions for price and variance.
    - lambda_: The mean jump-arrival rate (annualized).
    - mu_y: The mean of the price jump component.
    - sigma_y: The standard deviation of the price jump component.
    - rho_j: The correlation between the price jump and the volatility jump.
    - mu_v: The mean of the exponential jump size in variance.
    """

    def __init__(self, params=None):
        """
        Initializes the SVCJ model with an optional set of parameters.

        Args:
            params (dict, optional): A dictionary of SVCJ model parameters.
        """
        self.params = params
        self.param_names = ['mu', 'kappa', 'theta', 'sigma_v', 'rho', 'lambda_', 'mu_y', 'sigma_y', 'rho_j', 'mu_v']

    def init(self, historical_log_returns = None, periods_in_year=252,
                                               jump_threshold_factor=3.0):
        self.params = initial_params_estimates(historical_log_returns=historical_log_returns, periods_in_year=252,
                                               jump_threshold_factor=3.0)

    def get_param_values(self) -> list[float]:
        return [self.params[p] for p in self.param_names]

    # =========================================================================
    # PART 1: MODEL FITTING VIA SIMULATED METHOD OF MOMENTS (SMM)
    # =========================================================================
    def simulate_p_measure(self, params, T_sim, dt):
        """
        Simulates a single path of returns and volatility under the P-measure (real world).

        This version now implements the exact discrete-time model from
        Hou et al. (2020), Equations (6) and (7), to ensure perfect consistency
        with the model used for parameter estimation in the paper.
        """
        mu, kappa, theta, sigma_v, rho, lambda_, mu_y, sigma_y, rho_j, mu_v = params

        # The paper's model is a discrete-time specification where parameters
        # are defined for a single time step. We derive the discrete parameters here.
        alpha = kappa * theta
        beta = 1 - kappa

        N_steps = int(T_sim / dt)
        returns = np.zeros(N_steps)
        volatility = np.zeros(N_steps)

        # Initial volatility set to the long-run mean
        volatility[0] = theta

        # Generate correlated random numbers for the Brownian motions
        z1 = np.random.normal(size=N_steps)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=N_steps)

        # Generate jump occurrences (Bernoulli process)
        # The jump probability is the only place where dt is used, as lambda is a rate.
        # poisson_shocks = np.random.poisson(lambda_ * dt, N_steps)
        # jumps = poisson_shocks > 0
        jumps = np.random.uniform(0, 1, N_steps) < (lambda_ * dt)

        # Generate volatility jumps (Exponential distribution)
        Z_v = np.random.exponential(mu_v, N_steps)

        # Generate return jumps (Normal distribution, conditional on vol jumps)
        Z_y = np.random.normal(mu_y + rho_j * Z_v, sigma_y, N_steps)

        for t in range(1, N_steps):
            # Ensure non-negative volatility for the sqrt term
            v_prev = max(0, volatility[t - 1])

            # Implement the paper's discrete-time volatility update (Eq. 7)
            # The parameters alpha, beta, and sigma_v are for a single step, so no dt scaling.
            vol_diffusion = sigma_v * np.sqrt(v_prev) * z2[t]
            vol_jump = Z_v[t] * jumps[t]
            volatility[t] = alpha + beta * v_prev + vol_diffusion + vol_jump

            # Implement the paper's discrete-time log-return update (Eq. 6)
            # The parameters mu and the diffusion term are for a single step.
            return_diffusion = np.sqrt(v_prev) * z1[t]
            return_jump = Z_y[t] * jumps[t]
            returns[t] = mu + return_diffusion + return_jump

        return pd.Series(returns)


    def simulate_q_measure(self, S0, v0, T, r, q, N_paths, N_steps):
        """
        Simulates multiple asset price paths under the Q-measure (risk-neutral).
        This is used for Monte Carlo option pricing.
         Args:
            S0 (float): Initial asset price.
            v0 (float): Initial variance.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
            q (float): Dividend yield.
            N_paths (int): Number of simulation paths.
            N_steps (int): Number of time steps per path.
        """
        if self.params is None:
            raise ValueError("Model parameters have not been set or fitted.")

        #['mu', 'kappa', 'theta', 'sigma_v', 'rho', 'lambda_', 'mu_y', 'sigma_y', 'rho_j', 'mu_v']
        mu, kappa, theta, sigma_v, rho, lambda_, mu_y, sigma_y, rho_j, mu_v = self.get_param_values()

        dt = T / N_steps

        # Calculate the risk-neutral jump compensator term
        # E_Q[exp(Z_y) - 1]
        # E_Q[exp(Z_y)] = exp(mu_y + 0.5*sigma_y^2) * E_Q[exp(rho_j * Z_v)]
        # MGF of Exponential(1/mu_v) is 1 / (1 - t*mu_v)
        mgf_exp = 1.0 / (1.0 - rho_j * mu_v)
        expected_jump_return = np.exp(mu_y + 0.5 * sigma_y ** 2) * mgf_exp
        compensator = lambda_ * (expected_jump_return - 1)

        # Initialize arrays
        S = np.full(N_paths, S0)
        v = np.full(N_paths, v0)

        for _ in range(N_steps):
            # Generate correlated random numbers
            z1 = np.random.normal(size=N_paths)
            z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=N_paths)

            # Generate jump occurrences
            jumps = np.random.uniform(0, 1, N_paths) < (lambda_ * dt)

            # Generate volatility jumps
            Z_v = np.random.exponential(mu_v, N_paths) * jumps

            # Generate return jumps
            Z_y = np.random.normal(mu_y + rho_j * Z_v, sigma_y) * jumps

            # Ensure non-negative volatility
            v_prev = np.maximum(0, v)

            # Update volatility
            v = (theta * kappa * dt) + (1 - kappa * dt) * v_prev \
                + sigma_v * np.sqrt(v_prev * dt) * z2 \
                + Z_v

            # Update log-price under Q-measure
            drift = (r - q - 0.5 * v_prev - compensator) * dt
            diffusion = np.sqrt(v_prev * dt) * z1

            S *= np.exp(drift + diffusion + Z_y)

        return S



class SVCJModel_Antithetic(SimulationModel):


    """
    Implementation of the Stochastic Volatility with Correlated Jumps (SVCJ) model.

    This class provides methods to:
    1. Fit the SVCJ model parameters to historical data using the Simulated Method of Moments (SMM).
    2. Price European options using the fitted model via Monte Carlo simulation.
    3. Calculate major option Greeks (Delta, Gamma, Vega, Theta) using finite differences.

    The model dynamics are based on Hou et al. (2020), "Pricing Cryptocurrency Options".
    d logS_t = mu dt + sqrt(V_t) dW_t^(S) + Z_t^y dN_t
    d V_t   = kappa(theta - V_t) dt + sigma_v sqrt(V_t) dW_t^(V) + Z_t^v dN_t

    Parameters:
    - mu: Mean drift of the log-return process.
    - kappa: Mean-reversion rate for variance.
    - theta: Long-run mean of variance.
    - sigma_v: Volatility of volatility.
    - rho: Correlation between the two Brownian motions for price and variance.
    - lambda_: The mean jump-arrival rate (annualized).
    - mu_y: The mean of the price jump component.
    - sigma_y: The standard deviation of the price jump component.
    - rho_j: The correlation between the price jump and the volatility jump.
    - mu_v: The mean of the exponential jump size in variance.
    """

    def __init__(self, params=None):
        """
        Initializes the SVCJ model with an optional set of parameters.

        Args:
            params (dict, optional): A dictionary of SVCJ model parameters.
        """
        self.params = params
        self.param_names = ['mu', 'kappa', 'theta', 'sigma_v', 'rho', 'lambda_', 'mu_y', 'sigma_y', 'rho_j', 'mu_v']

    def init(self, historical_log_returns=None, periods_in_year=365,
             jump_threshold_factor=3.0):
        self.params = initial_params_estimates(historical_log_returns=historical_log_returns,
                                               periods_in_year=periods_in_year,
                                               jump_threshold_factor=3.0)

    def get_param_values(self) -> list[float]:
        return [self.params[p] for p in self.param_names]


    def simulate_p_measure(self, param_list, T_sim, dt):
        """
        Simulates two paths of returns (original and antithetic) under the P-measure.
        This implements the antithetic variates variance reduction technique.
        """
        mu, kappa, theta, sigma_v, rho, lambda_, mu_y, sigma_y, rho_j, mu_v = param_list

        alpha = kappa * theta
        beta = 1 - kappa

        N_steps = int(T_sim / dt)

        # Initialize arrays for two paths
        returns = np.zeros(N_steps)
        volatility = np.zeros(N_steps)
        returns_anti = np.zeros(N_steps)
        volatility_anti = np.zeros(N_steps)

        # Set the starting value at t=0 to the long-run mean for both paths
        volatility[0] = theta
        volatility_anti[0] = theta

        # --- Antithetic Variates Setup ---
        # Generate one set of random numbers for both paths
        z1 = np.random.normal(size=N_steps)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=N_steps)

        Z_v_exp = np.random.exponential(mu_v, N_steps)
        Z_y_norm_base = np.random.normal(0, 1, N_steps)  # Base normal for correlated jumps

        # Jumps for the original path
        poisson_shocks = np.random.poisson(lambda_ * dt, N_steps)
        jumps = poisson_shocks > 0
        # jumps_uniform = np.random.uniform(0, 1, N_steps)
        # jumps = jumps_uniform < (lambda_ * dt)
        Z_v = Z_v_exp
        Z_y = mu_y + rho_j * Z_v + sigma_y * Z_y_norm_base

        # Jumps for the antithetic path (using inverted random numbers)
        # Note: For uniformity, the antithetic is 1-u. For exponential, this is complex.
        # For simplicity and stability, we only apply antithetics to the normal drivers.
        jumps_anti = jumps  # Keep jump timing the same for stability
        Z_v_anti = Z_v
        Z_y_anti = mu_y + rho_j * Z_v_anti - sigma_y * Z_y_norm_base  # Negate the normal shock

        for t in range(1, N_steps):
            # --- Original Path ---
            v_prev = max(0, volatility[t - 1])
            volatility[t] = alpha + beta * v_prev + sigma_v * np.sqrt(v_prev) * z2[t] + Z_v[t] * jumps[t]
            returns[t] = mu + np.sqrt(v_prev) * z1[t] + Z_y[t] * jumps[t]

            # --- Antithetic Path ---
            v_prev_anti = max(0, volatility_anti[t - 1])
            # Use negated Brownian motion shocks
            volatility_anti[t] = alpha + beta * v_prev_anti - sigma_v * np.sqrt(v_prev_anti) * z2[t] + Z_v_anti[t] * \
                                 jumps_anti[t]
            returns_anti[t] = mu - np.sqrt(v_prev_anti) * z1[t] + Z_y_anti[t] * jumps_anti[t]

        return pd.Series(returns), pd.Series(returns_anti)

    def simulate_q_measure(self, S0, v0, T, r, q, N_paths, N_steps):
        """
        Simulates multiple asset price paths under the Q-measure, now using
        antithetic variates for improved pricing accuracy.
        """
        if self.params is None:
            raise ValueError("Model parameters have not been set or fitted.")

        mu, kappa, theta, sigma_v, rho, lambda_, mu_y, sigma_y, rho_j, mu_v = self.get_param_values()


        dt = T / N_steps

        # Risk-neutral jump compensator
        mgf_exp = 1.0 / (1.0 - rho_j * mu_v)
        expected_jump_return = np.exp(mu_y + 0.5 * sigma_y ** 2) * mgf_exp
        compensator = lambda_ * (expected_jump_return - 1)

        # Initialize arrays for original and antithetic paths
        S = np.full(N_paths, S0)
        v = np.full(N_paths, v0)
        S_anti = np.full(N_paths, S0)
        v_anti = np.full(N_paths, v0)

        for _ in range(N_steps):
            # Generate one set of random numbers for both sets of paths
            z1 = np.random.normal(size=N_paths)
            z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=N_paths)
            jumps = np.random.uniform(0, 1, N_paths) < (lambda_ * dt)
            Z_v = np.random.exponential(mu_v, N_paths) * jumps
            Z_y_base = np.random.normal(0, 1, N_paths)
            Z_y = (mu_y + rho_j * Z_v + sigma_y * Z_y_base) * jumps
            Z_y_anti = (mu_y + rho_j * Z_v - sigma_y * Z_y_base) * jumps

            # --- Original Paths ---
            v_prev = np.maximum(0, v)
            v = (theta * kappa * dt) + (1 - kappa * dt) * v_prev + sigma_v * np.sqrt(v_prev * dt) * z2 + Z_v
            drift = (r - q - 0.5 * v_prev - compensator) * dt
            diffusion = np.sqrt(v_prev * dt) * z1
            S *= np.exp(drift + diffusion + Z_y)

            # --- Antithetic Paths ---
            v_prev_anti = np.maximum(0, v_anti)
            v_anti = (theta * kappa * dt) + (1 - kappa * dt) * v_prev_anti - sigma_v * np.sqrt(
                v_prev_anti * dt) * z2 + Z_v
            drift_anti = (r - q - 0.5 * v_prev_anti - compensator) * dt
            diffusion_anti = -np.sqrt(v_prev_anti * dt) * z1
            S_anti *= np.exp(drift_anti + diffusion_anti + Z_y_anti)

        # Return the combined set of final prices
        return np.concatenate((S, S_anti))





def compute_implied_volatility(sim_model: SimulationModel, S0, v0, K, T, r, q, option_type='call', N_paths=10000, N_steps=100):
    """
    Computes the implied volatility for a given option using the Bisection method.
    """
    estimated_price = price_option(sim_model=sim_model, S0=S0, v0=v0, K=K, T=T, r=r, q=q, option_type=option_type,
                                   N_paths=N_paths, N_steps=N_steps)
    estimated_iv = implied_volatility.implied_volatility(estimated_price, S0, K, T, r, option_type[0])
    return estimated_price, estimated_iv

def price_option(sim_model: SimulationModel, S0, v0, K, T, r, q, option_type='call', N_paths=10000, N_steps=100):
    """
    Prices a European option using Monte Carlo simulation.

    Args:
        S0 (float): Initial asset price.
        v0 (float): Initial variance.
        K (float): Strike price.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        option_type (str): 'call' or 'put'.
        N_paths (int): Number of simulation paths.
        N_steps (int): Number of time steps per path.

    Returns:
        float: The estimated option price.
    """
    final_S = sim_model.simulate_q_measure(S0, v0, T, r, q, N_paths, N_steps)

    if option_type.lower() == 'call':
        payoffs = np.maximum(final_S - K, 0)
    elif option_type.lower() == 'put':
        payoffs = np.maximum(K - final_S, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    return np.mean(payoffs) * np.exp(-r * T)

def calculate_greeks(sim_model: SimulationModel, S0, v0, K, T, r, q, option_type='call', N_paths=10000, N_steps=100, dS=0.01, dv=0.001,
                     dt_greek=1 / 365.0):
    """
    Calculates major option Greeks using the finite difference (bump-and-revalue) method.

    Args:
        dS (float): Small change in stock price for Delta and Gamma.
        dv (float): Small change in variance for Vega.
        dt_greek (float): Small change in time for Theta.

    Returns:
        dict: A dictionary containing Delta, Gamma, Vega, and Theta.
    """
    # Original price
    price_orig = sim_model.price_option(S0, v0, K, T, r, q, option_type, N_paths, N_steps)

    # --- Delta ---
    price_up_S = sim_model.price_option(S0 + dS, v0, K, T, r, q, option_type, N_paths, N_steps)
    price_down_S = sim_model.price_option(S0 - dS, v0, K, T, r, q, option_type, N_paths, N_steps)
    delta = (price_up_S - price_down_S) / (2 * dS)

    # --- Gamma ---
    gamma = (price_up_S - 2 * price_orig + price_down_S) / (dS ** 2)

    # --- Vega ---
    # Note: Sensitivity to initial variance v0. Result is per point of variance.
    # To get per point of volatility (sqrt(v0)), divide by 2*sqrt(v0).
    price_up_v = sim_model.price_option(S0, v0 + dv, K, T, r, q, option_type, N_paths, N_steps)
    price_down_v = sim_model.price_option(S0, v0 - dv, K, T, r, q, option_type, N_paths, N_steps)
    vega = (price_up_v - price_down_v) / (2 * dv)

    # --- Theta ---
    # Note: Time decay per year.
    if T > dt_greek:
        price_time_decay = sim_model.price_option(S0, v0, K, T - dt_greek, r, q, option_type, N_paths, N_steps)
        theta = (price_time_decay - price_orig) / dt_greek
    else:
        theta = -price_orig / dt_greek  # Approximation for very short expiry

    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}


def initial_params_estimates(historical_log_returns: pd.DataFrame, periods_in_year: int=365, jump_threshold_factor=3.0) -> Dict[str, float]:
    def evaluate_jump_params(historical_log_returns, periods_in_year = 252, jump_threshold_factor = 3.0):
        std_dev = np.std(historical_log_returns)
        jump_threshold = jump_threshold_factor * std_dev
        is_jump_period = np.abs(historical_log_returns) > jump_threshold
        jump_returns = historical_log_returns[is_jump_period]
        #normal_returns = historical_log_returns[~is_jump_period]
        lambda_initial = len(jump_returns) / len(historical_log_returns) * periods_in_year
        mu_y_initial = np.mean(jump_returns)
        sigma_y_initial = np.std(jump_returns)
        rho_initial = historical_log_returns[1:].corr((historical_log_returns ** 2).diff(), method="spearman")
        mu_v_initial = np.mean(historical_log_returns[is_jump_period] ** 2)
        rho_j_initial = historical_log_returns[is_jump_period].corr((historical_log_returns[is_jump_period] ** 2), method="spearman")
        return lambda_initial, mu_y_initial, sigma_y_initial, rho_initial, mu_v_initial, rho_j_initial

    def estimate_initial_kappa(hist_returns: pd.Series) -> float:

        squared_returns = (hist_returns ** 2).values
        print("Fitting AR(1) model to squared returns...")
        model = AutoReg(squared_returns, lags=1, old_names=False)
        results = model.fit()

        print("\nAR(1) Model Fit Results:")
        print(results.summary())

        try:
            beta = results.params['y.L1']
        except Exception:
            # Handle older statsmodels versions if necessary
            beta = results.params[1]

        #print(f"\nEstimated AR(1) coefficient (beta): {beta:.4f}")
        kappa = 1.0 - beta
        return kappa

    mu = np.mean(historical_log_returns)  # daily long-term mean
    theta = np.var(historical_log_returns) #  daily variance
    kappa = estimate_initial_kappa(historical_log_returns)
    lambda_initial, mu_y_initial, sigma_y_initial, rho_initial, mu_v_initial,  rho_j_initial = (
            evaluate_jump_params(historical_log_returns, periods_in_year, jump_threshold_factor=jump_threshold_factor))
    return {
        'mu': mu, 'kappa': kappa, 'theta': theta, 'sigma_v': sigma_y_initial, 'rho': rho_initial,
        'lambda_': lambda_initial, 'mu_y': mu_y_initial, 'sigma_y': sigma_y_initial,
        'rho_j': rho_j_initial, 'mu_v': mu_v_initial
    }


# @jit(nopython=True)
# def normal_array(size, loc=0.0, scale=1.0):
#     """
#     Generates an array of normally distributed random numbers within a Numba-jitted function.
#     """
#     result = np.empty(size, dtype=np.float64)
#     for i in range(size):
#         result[i] = np.random.normal(loc=loc, scale=scale)
#     return result