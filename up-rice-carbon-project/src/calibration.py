"""
Calibration and Gaussian Process surrogate utilities for the RothC soil
carbon model.
"""

from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from .rothc_model import run_simulation


def calibrate_carbon_input(
    target_rate: float,
    climate_months: List[Tuple[float, float]],
    initial_soc: float,
    years: int,
    bounds: Tuple[float, float] = (0.5, 20.0),
) -> float:
    """Calibrate RothC's annual carbon input to match a target SOC change rate.

    Args:
        target_rate: Target average SOC change rate, in t C/ha/yr (e.g. from
            published literature for the relevant cropping system).
        climate_months: List of 12 (temp_c, rain_mm) tuples, one per month.
        initial_soc: Starting total soil organic carbon, in t C/ha.
        years: Number of years to simulate when evaluating the achieved rate.
        bounds: Search bounds for the carbon_input_annual parameter.

    Returns:
        The calibrated carbon_input_annual value (t C/ha/yr) that best
        reproduces target_rate over the given simulation length.
    """

    def objective(carbon_input: float) -> float:
        results = run_simulation(
            years=years, climate_months=climate_months,
            carbon_input_annual=carbon_input, initial_soc=initial_soc,
        )
        end_soc = results[-1]["total_soc"]
        achieved_rate = (end_soc - initial_soc) / years
        return (achieved_rate - target_rate) ** 2

    res = minimize_scalar(objective, bounds=bounds, method="bounded")
    return res.x


def build_gp_surrogate(
    climate_months: List[Tuple[float, float]],
    initial_soc: float,
    years: int,
    n_samples: int = 60,
    carbon_input_range: Tuple[float, float] = (1.0, 20.0),
    clay_pct_range: Tuple[float, float] = (10.0, 40.0),
    random_state: int = 42,
) -> Tuple[GaussianProcessRegressor, np.ndarray, np.ndarray]:
    """Train a Gaussian Process surrogate that approximates RothC's average
    SOC change rate as a function of (carbon_input_annual, clay_pct).

    Args:
        climate_months: List of 12 (temp_c, rain_mm) tuples, one per month.
        initial_soc: Starting total soil organic carbon, in t C/ha.
        years: Number of years to simulate for each training sample.
        n_samples: Number of RothC runs to generate as training data.
        carbon_input_range: (min, max) range to sample carbon_input_annual from.
        clay_pct_range: (min, max) range to sample clay percentage from.
        random_state: Seed for reproducibility.

    Returns:
        A tuple (fitted_gp_model, X, y) where X is the (n_samples, 2) input
        array and y is the (n_samples,) array of RothC-simulated SOC rates.
    """
    rng = np.random.default_rng(random_state)
    carbon_inputs = rng.uniform(*carbon_input_range, n_samples)
    clay_pcts = rng.uniform(*clay_pct_range, n_samples)

    X = np.column_stack([carbon_inputs, clay_pcts])
    y = np.zeros(n_samples)

    for i in range(n_samples):
        results = run_simulation(
            years=years, climate_months=climate_months,
            carbon_input_annual=carbon_inputs[i], initial_soc=initial_soc,
            clay_pct=clay_pcts[i],
        )
        y[i] = (results[-1]["total_soc"] - initial_soc) / years

    kernel = ConstantKernel(1.0) * RBF(length_scale=[5.0, 10.0]) + WhiteKernel(
        noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=random_state
    )
    gp.fit(X, y)

    return gp, X, y
