import json
from datetime import datetime
from pprint import pprint
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import skew, kurtosis
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from simulation_model_base import SimulationModel
from simulation_model_svcj import SVCJ_Model, initial_params_estimates, SVCJModel_Antithetic, compute_implied_volatility
import matplotlib.pyplot as plt

from utils import store_dict_as_json, load_json, seconds_to_hours_mins


class SMMFitter:
    def __init__(self, model: SimulationModel, use_gmm_moments: bool = True,
                 historical_log_returns= None, autocorr_moment_lags=5, max_moment_order=5,
                 use_gmm_objective_function = False):
        self.model = model
        if historical_log_returns is not None and use_gmm_moments:
            self.model.init(historical_log_returns = historical_log_returns,
                            periods_in_year=365,
                            jump_threshold_factor=3.0)
        self.autocorr_moment_lags = autocorr_moment_lags
        self.max_moment_order = max_moment_order
        self.moment_function = generic_and_autocorr_moments if use_gmm_moments else simple_and_autocorrelation_moments
        self.param_names = self.model.param_names
        if use_gmm_objective_function:
            self.weighting_matrix = self.compute_weighting_matrix(n_reps=1000, T_sim=100, dt=1 / 365)
        self.objective_function = self.objective_function_gmm if use_gmm_objective_function else self.objective_function_simple
        self.temp_params = np.zeros(len(self.param_names))

    def fit_bayes(self, historical_returns, initial_guess=None, bounds=None, T_sim=100, dt=1 / 365,
            n_initial_points=25, random_state=123, maxiter=200):
        """
        Fits the SVCJ model parameters using Bayesian Optimization (gp_minimize).

        Args:
            n_calls (int): Total number of evaluations of the objective function.
            n_initial_points (int): Number of random points to evaluate before
                                  starting the intelligent search.
            random_state (int): Seed for reproducibility.
        """
        print("--- Starting Bayesian Optimization for SVCJ Parameter Fitting ---")

        # --- FIX: Define the parameter search space for scikit-optimize ---
        space = []
        for name, (low, high) in zip(self.param_names, bounds):
            space.append(Real(low, high, name=name))

        # Step 2: Calculate the fixed target moments from historical data
        target_moments = self.moment_function(historical_returns, autocorr_lags=self.autocorr_moment_lags,
                                              max_moment_order=self.max_moment_order)
        print("Target moments calculated.")

        # Step 3: Define the objective function wrapper for the optimizer
        # The @use_named_args decorator allows skopt to pass parameters by name
        @use_named_args(space)
        def objective(**params):
            # Convert the named parameters into the array format our simulator expects
            params_array = [params[name] for name in self.param_names]
            return self.objective_function(params_array, target_moments, T_sim, dt)

        # Step 4: Run the Bayesian optimization
        print(f"Running gp_minimize for {maxiter} total evaluations...")
        result = gp_minimize(
            func=objective,
            dimensions=space,
            x0=initial_guess,
            n_calls=maxiter,
            n_initial_points=n_initial_points,
            acq_func='gp_hedge',  # A robust acquisition function
            random_state=random_state,
            verbose=True,
            n_jobs=-1
        )

        # Step 5: Process and return the results
        if result.x:
            print("\nBayesian optimization successful!")
            print(f"Best objective function value found: {result.fun:.6f}")
            self.params = dict(zip(self.param_names, result.x))
            self.model.params = self.params
            return self.params
        else:
            print("\nBayesian optimization failed.")
            return None

    def fit_diff_evolution(self, historical_returns, initial_guess, bounds=None, T_sim=80, dt=1/365,
                           maxiter=1000, popsize=10, mutation=(0.5, 1.0), recombination=0.7, opt_method ='best1bin', tol=1e-3):
        """
        Fits the simulated model parameters to historical return data using SMM.

        Args:
            historical_returns (pd.Series): Time series of historical log returns.
            initial_guess (list): Initial guess for the 10 model parameters.
            bounds (list of tuples): Bounds for each parameter for the optimizer.
            T_sim (int): Length of simulation in years for each optimization step.
            dt (float): Time step for simulation (e.g., 1/252 for daily).

        Returns:
            dict: A dictionary of the fitted model parameters.
        """
        print("Calculating target moments from historical data...")
        target_moments = self.moment_function(historical_returns, autocorr_lags=self.autocorr_moment_lags,
                                              max_moment_order=self.max_moment_order)
        print("Starting SMM optimization...")
        temp_result = None

        def save_iter_results(intermediate_result):
            global temp_result
            temp_result = intermediate_result.x
            print(f"Optimisation iteration completed: obj function={intermediate_result.fun}, "
                  f"intermediate_result={intermediate_result}")


        result = differential_evolution(
            func=self.objective_function,
            bounds=bounds,
            x0=initial_guess,
            args=(target_moments, T_sim, dt),
            strategy=opt_method,
            maxiter=maxiter,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            disp=True,  # Print progress
            polish=False,  # Turn off final local optimization, as it can be misled by noise
            callback=save_iter_results,
            tol=tol
        )

        if result.success:
            print("\nOptimization successful!")
        else:
            print(f"\nOptimization did not complete. Reason {result.message}")
        self.params = dict(zip(self.param_names, result.x))
        self.model.params = self.params
        return self.params

    def fit_loc_minimize(self, historical_returns, initial_guess, bounds=None, T_sim=80, dt=1 / 365,
            maxiter=1000, tol=1e-3, opt_method='L-BFGS-B'):
        """
        Fits the simulated model parameters to historical return data using SMM.

        Args:
            historical_returns (pd.Series): Time series of historical log returns.
            initial_guess (list): Initial guess for the 10 model parameters.
            bounds (list of tuples): Bounds for each parameter for the optimizer.
            T_sim (int): Length of simulation in years for each optimization step.
            dt (float): Time step for simulation (e.g., 1/252 for daily).

        Returns:
            dict: A dictionary of the fitted model parameters.
        """
        print("Calculating target moments from historical data...")
        target_moments = self.moment_function(historical_returns, autocorr_lags=self.autocorr_moment_lags,
                                              max_moment_order=self.max_moment_order)
        print("Starting SMM optimization...")

        def print_iter_message(intermediate_result):
            print(f"Optimisation iteration completed: obj function={intermediate_result.fun}, "
                  f"intermediate_result={intermediate_result}")


        result = minimize(
            fun=self.objective_function,
            x0=np.array(initial_guess),
            args=(target_moments, T_sim, dt),
            method=opt_method,  # choice with numerical gradient estimations 'L-BFGS-B','Nelder-Mead'
            bounds=bounds,
            tol=tol,
            options={'maxiter': maxiter, 'finite_diff_rel_step': 0.1},
            callback=print_iter_message
        )

        if result.success:
            print("\nOptimization successful!")
            self.params = dict(zip(self.param_names, result.x))
            self.model.params = self.params
        else:
            print(f"\nOptimization failed. Reason {result.message}")
            self.params = None

        return self.params

    def objective_function_simple(self, params_array, target_moments, T_sim, dt):
        """
        The SMM objective function with simple diagonal weighting matrix. It calculates the weighted squared difference between

        target moments (from real data) and moments from a simulated path.
        """
        simulated_returns = self.model.simulate_p_measure(tuple(params_array), T_sim, dt)
        simulated_moments = self.moment_function(simulated_returns, autocorr_lags=self.autocorr_moment_lags,
                                                 max_moment_order=self.max_moment_order)

        # Use identity matrix for weighting (sum of squared errors)
        moment_errors = (simulated_moments - target_moments)/(np.abs(target_moments)+0.000001)
        result = np.sum(moment_errors ** 2)
        #print(f"Objective function {result}")
        return result


    def objective_function_gmm(self, params_array, target_moments, T_sim, dt):
        """The GMM objective function using a specified weighting matrix."""
        #print(f"Params array = {params_array}, \nDelta= {params_array - self.temp_params}")
        #self.temp_params = params_array
        simulated_returns = self.model.simulate_p_measure(tuple(params_array), T_sim, dt)
        simulated_moments = self.moment_function(simulated_returns, max_moment_order = self.max_moment_order,
                                                 autocorr_lags = self.autocorr_moment_lags)
        moment_errors = simulated_moments - target_moments
        #print(f"Moment errors =  {moment_errors}")
        results = moment_errors.T @ self.weighting_matrix @ moment_errors
        #print(f"Objective function val: {results}")
        return results

    def compute_weighting_matrix(self, n_reps, T_sim, dt=1 / 365):
        """
        Estimates the optimal GMM weighting matrix (S^-1) using simulation.
        """
        print(f"\nEstimating optimal weighting matrix with {n_reps} replications...")
        moment_replications = []
        for i in range(n_reps):
            if (i + 1) % 100 == 0:
                print(f"  Replication {i + 1}/{n_reps}")
            sim_returns = self.model.simulate_p_measure(tuple(self.model.get_param_values()), T_sim, dt)
            moment_replications.append(generic_and_autocorr_moments(sim_returns,
                                                                    max_moment_order=self.max_moment_order,
                                                                    autocorr_lags=self.autocorr_moment_lags))

        # Calculate the covariance matrix of the moments
        S = np.cov(np.array(moment_replications).T)

        # The optimal weighting matrix is the inverse of S
        try:
            W = np.linalg.inv(S)
            print("Successfully inverted moment covariance matrix.")
            return W
        except np.linalg.LinAlgError:
            print("Warning: Moment covariance matrix is singular.")
            print("Falling back to a diagonal weighting matrix (inverse of variances).")
            variances = np.diag(S)
            # Add a small epsilon to prevent division by zero
            return np.diag(1.0 / (variances + 1e-8))


def simple_and_autocorrelation_moments(returns, autocorr_moment_lags=3) -> np.ndarray:
    """
    Calculates a vector of statistical moments for a given return series.
    Includes standard moments and autocorrelation of squared returns to capture volatility clustering.
    """
    if isinstance(returns, tuple) and len(returns) ==2:
        return (simple_and_autocorrelation_moments(returns[0], autocorr_moment_lags) +
                simple_and_autocorrelation_moments(returns[1], autocorr_moment_lags))/2.0
    returns_np = returns.to_numpy()
    moments = [
        np.mean(returns_np),
        np.var(returns_np),
        skew(returns_np),
        kurtosis(returns_np, fisher=False)  # Pearson's kurtosis (normal = 3)
    ]

    squared_returns = returns ** 2
    for i in range(1, autocorr_moment_lags+1):
        autocorr_val = squared_returns.autocorr(lag=i)
        moments.append(0 if np.isnan(autocorr_val) else autocorr_val)

    return np.array(moments)


def generic_and_autocorr_moments(returns, max_moment_order=4, autocorr_lags=5):
    """
    Calculates a generic vector of moments for a given return series.

    Args:
        returns (pd.Series): The time series of log returns.
        moment_orders (list): A list of the orders of the moments to calculate.
                              Order 1 = Mean
                              Order 2 = Variance
                              Orders > 2 = Standardized moments (e.g., 3=skew, 4=kurtosis)
        autocorr_lags (int): Number of autocorrelation lags of squared returns to include.

    Returns:
        np.array: A vector of the calculated moments.
    """
    if isinstance(returns, tuple) and len(returns) ==2:
        return (generic_and_autocorr_moments(returns[0], max_moment_order, autocorr_lags) +
                generic_and_autocorr_moments(returns[1], max_moment_order, autocorr_lags))/2.0
    # --- Standard Moments ---
    sample_mean = np.mean(returns)
    sample_var = np.var(returns)
    sample_std = np.sqrt(sample_var)

    moments = []

    for N in range(1, max_moment_order+1):
        if N == 1:
            moments.append(sample_mean)
        elif N == 2:
            moments.append(sample_var)
        else:  # Calculate N-th standardized moment for N > 2
            # Avoid division by zero if standard deviation is zero
            if sample_std < 1e-12:
                standardized_moment = 0.0
            else:
                # Calculate N-th central moment
                central_moment = np.mean((returns - sample_mean) ** N)
                # Standardize it
                standardized_moment = central_moment / (sample_std ** N)
            moments.append(standardized_moment)

    # --- Autocorrelation Moments ---
    squared_returns = returns ** 2
    for i in range(1, autocorr_lags + 1):
        autocorr_val = squared_returns.autocorr(lag=i)
        moments.append(0 if np.isnan(autocorr_val) else autocorr_val)

    return np.array(moments)


def compute_model_fitting():
    historical_prices = pd.read_csv("./data/backtest_data/BTC_USD_deribit.csv",
                                    parse_dates=True, header=None,
                                    names=['timestamp', 'open', 'high', 'low', 'close',
                                           'volume'])
    historical_prices.index = pd.to_datetime(historical_prices['timestamp'], unit='ms')
    historical_prices = historical_prices[historical_prices.index <= '2024-09-01']
    print(f"Historical sample with {len(historical_prices)} daily returns. \nTail\n{historical_prices.tail()}" )


    hist_log_returns = np.log(historical_prices['close'] / historical_prices['close'].shift(1)).dropna()
    print(f"Historical sample with {len(hist_log_returns)} daily returns.")
    print("Sample Data Head:\n", hist_log_returns[0:10])

    true_params_dict = {
        'mu': 0.041, 'kappa': 1.132, 'theta': 0.0088, 'sigma_v': 0.008, 'rho': 0.407,
        'lambda_': 0.041, 'mu_y': -0.084, 'sigma_y': np.sqrt(2.155), 'rho_j': -0.573, 'mu_v': 0.620
    }
    true_params_array = list(true_params_dict.values())

    print("--- Generating a sample historical return series from known SVCJ parameters ---")
    #svcj_model = SVCJ_Model()  # SVCJ Model with regular prices
    svcj_model = SVCJModel_Antithetic()
    svcj_model.init(historical_log_returns=hist_log_returns, periods_in_year=365, jump_threshold_factor=3.0)
    pprint(f"Initial approximations for SVCJ model params: {svcj_model.params}")
    # sample_returns = svcj_model.simulate_p_measure(true_params_array, T_sim=5, dt=1 / 252.0)
    # print(f"Generated {len(sample_returns)} daily returns.")
    # print("Sample Data Head:\n", sample_returns.head())

# Define initial guess and bounds for the SMM optimizer
    #initial_guess = [0.05, 0.5, 0.01, 0.01, 0.3, 0.05, -0.1, 1.5, -0.5, 0.5]

    #initial_dict = initial_params_estimates(hist_log_returns, 252)
    #store_dict_as_json(svcj_model.params, filename="./output/smm_initial_params.json")


    initial_estimate = svcj_model.get_param_values()
    initial_params = svcj_model.params
    bounds = [
        (-0.5, 1.0),   # mu
        (1e-3, 10),    # kappa
        (1e-5, 1),     # theta
        (1e-5, 10),    # sigma_v
        (-1.0, 1.0),   # rho
        (1e-6, initial_estimate[5] * 20),      # lambda_
        (-1.0, 1.0),                           # mu_y
        (1e-3, initial_estimate[7] * 10),      # sigma_y
        (-1.0, 1.0),                           # rho_j
        (1e-5, 2)                              # mu_v
    ]

    # Instantiate and fit the model
    smm_fitter = SMMFitter(svcj_model,
                           historical_log_returns=hist_log_returns,
                           autocorr_moment_lags=3,
                           max_moment_order=5,
                           use_gmm_objective_function=False)
    # For better accuracy, increase T_sim (e.g., to 80 or more).
    start_time = datetime.now()
    #bounds = None
    fitted_params = smm_fitter.fit_diff_evolution(hist_log_returns, initial_estimate.copy(), bounds, T_sim=500,
                                                  maxiter=100)
    end_time = datetime.now()
    calc_time_secs = (end_time - start_time).total_seconds()
    caltime_hours, calc_time_mins = seconds_to_hours_mins(calc_time_secs)
    print(f"Fitting time: {caltime_hours} hours and {calc_time_mins} minutes.")

    if fitted_params:
        store_dict_as_json(fitted_params, filename="./output/smm_fitted_params6.json")
        print("\n--- Initial vs. Reference vs. Fitted Parameters ---")
        print(f"{'Parameter':<16}{'True':>16}{'Initial':>16}{'Fitted':>16}")
        print("-" * 40)
        for name in smm_fitter.param_names:
            print(f"{name:<10}{true_params_dict[name]:>16.9f}{initial_params[name]:>16.9f}"
                  f"{fitted_params[name]:>16.9f}")



def compute_price_and_IV(S0=108031.0, K = 108100.0, T = 1/12, r = 0.04, q = 0.0,
                         params_path="./output/smm_fitted_params1.json"):
    params = load_json(params_path)
    svcj_model = SVCJModel_Antithetic(params=params)

    # Use the long-run mean volatility as the starting point
    v0 = svcj_model.params['theta']
    price_est, iv = compute_implied_volatility(svcj_model, S0, v0, K, T, r, q, option_type='call',
                                               N_paths=100000, N_steps=1000)
    print(f"SVCJ Model Option Price: {price_est:.4f},  IV={iv}")

def test_distr(params_path="./output/smm_fitted_params_inter.json"):
    historical_prices = pd.read_csv("./data/backtest_data/BTC_USD_deribit.csv",
                                    index_col=0, parse_dates=True, header=None,
                                    names=['timestamp', 'open', 'high', 'low', 'close',
                                           'volume'])
    hist_log_returns = np.log(historical_prices['close'] / historical_prices['close'].shift(1)).dropna()

    # params_dict_from_paper = {
    #     'mu': 0.14, 'kappa': 1.132, 'theta': 0.0088, 'sigma_v': 0.008, 'rho': 0.407,
    #     'lambda_': 0.041, 'mu_y': -0.084, 'sigma_y': np.sqrt(2.155), 'rho_j': -0.573, 'mu_v': 0.620
    # }

    #param_estimates = initial_params_estimates(historical_log_returns=hist_log_returns, jump_threshold_factor=3.0)
    param_estimates = load_json(params_path)
    print(f"Param estimates: \n{json.dumps(param_estimates, indent=4)}")
    params_array = list(param_estimates.values())
    #svcj_model = SVCJ_Model()
    svcj_model = SVCJModel_Antithetic()
    sample_returns = svcj_model.simulate_p_measure(params_array, dt=1 / 365)

    print(f"Generated {len(sample_returns[0])} daily returns.")
    (hist_log_returns).hist(bins=100, density=True)
    sample_returns_inst = sample_returns[0] if isinstance(sample_returns, tuple) else sample_returns
    (sample_returns_inst).hist(bins=100, density=True)
    plt.show()

    # sample_returns.hist(bins=30)

    # print(f"Historical sample with {len(hist_log_returns)} daily returns.")
    # print("Historical Data Head:\n", hist_log_returns[0:10])

    #sample_moments = simple_and_autocorrelation_moments(sample_returns, autocor_lags=6)
    sample_moments = generic_and_autocorr_moments(sample_returns, max_moment_order=5, autocorr_lags=3)
    print(f"Generated sample moments: {sample_moments}")

    hist_moments = generic_and_autocorr_moments(hist_log_returns, max_moment_order=5, autocorr_lags=3)
    print(f"Historical moments: {hist_moments}")

if __name__ == '__main__':
    compute_model_fitting()
    #test_pricing()
    #test_distr()
    #compute_price_and_IV()














