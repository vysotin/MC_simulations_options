from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd


class SimulationModel(ABC):

    def init(self, **kwargs):
        pass

    def get_param_values(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def simulate_p_measure(self, params, T_sim, dt) -> Union[pd.Series, tuple[pd.Series, pd.Series]]:
        raise NotImplementedError

    @abstractmethod
    def simulate_q_measure(self, param_list, T_sim, dt) -> np.ndarray:
        """
        Simulates multiple asset price paths under the Q-measure (risk-neutral).

        This is used for Monte Carlo option pricing.

        Returns:
            ndarray: A 2D array of shape (N_paths, N_steps) containing the simulated asset price paths.
        """
        raise NotImplementedError
