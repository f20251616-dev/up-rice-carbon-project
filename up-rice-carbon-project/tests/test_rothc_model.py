"""
Tests for the RothC soil carbon model.

Run with: pytest tests/
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rothc_model import (
    temp_factor,
    moisture_factor,
    cover_factor,
    rothc_step,
    run_simulation,
)


def test_temp_factor_increases_with_temperature():
    """Warmer temperatures should produce a higher (faster) decomposition factor."""
    assert temp_factor(35) > temp_factor(15) > temp_factor(0)


def test_temp_factor_near_zero_for_extreme_cold():
    """Below -5C, decomposition should effectively stop."""
    assert temp_factor(-10) == 0.0


def test_moisture_factor_within_bounds():
    """Moisture factor should always stay within RothC's [0.2, 1.0] range."""
    for rain in [0, 50, 200, 400]:
        f = moisture_factor(rain)
        assert 0.2 <= f <= 1.0


def test_cover_factor_vegetated_slower_than_bare():
    """Vegetated soil should decompose slower than bare soil."""
    assert cover_factor(is_vegetated=True) < cover_factor(is_vegetated=False)


def test_iom_pool_never_changes():
    """The Inert Organic Matter pool should never change -- it is inert by definition."""
    _, _, _, _, iom_after = rothc_step(
        dpm=1, rpm=1, bio=1, hum=1, iom=5.0,
        temp_c=30, rain_mm=300, is_vegetated=True, carbon_input=1.0,
    )
    assert iom_after == 5.0


def test_no_decomposition_without_input_in_extreme_cold():
    """With zero carbon input and extreme cold, pools should barely change."""
    dpm, rpm, bio, hum, iom = 1.0, 1.0, 1.0, 1.0, 1.0
    total_before = dpm + rpm + bio + hum + iom
    dpm2, rpm2, bio2, hum2, iom2 = rothc_step(
        dpm, rpm, bio, hum, iom,
        temp_c=-10, rain_mm=0, is_vegetated=False, carbon_input=0,
    )
    total_after = dpm2 + rpm2 + bio2 + hum2 + iom2
    assert abs(total_after - total_before) < 0.01


def test_simulation_produces_no_negative_soc():
    """Total SOC should never go negative over a realistic simulation."""
    climate = list(zip(
        [15.0, 18.5, 24.0, 29.5, 32.0, 31.5, 28.5, 28.0, 27.5, 24.5, 19.5, 15.5],
        [15, 12, 8, 5, 40, 180, 320, 290, 180, 40, 5, 8],
    ))
    results = run_simulation(years=10, climate_months=climate,
                              carbon_input_annual=2.5, initial_soc=30.0)
    socs = [r["total_soc"] for r in results]
    assert all(s >= 0 for s in socs)


def test_simulation_no_erratic_jumps():
    """Month-to-month SOC changes should be smooth, not spiky, given realistic inputs."""
    climate = list(zip(
        [15.0, 18.5, 24.0, 29.5, 32.0, 31.5, 28.5, 28.0, 27.5, 24.5, 19.5, 15.5],
        [15, 12, 8, 5, 40, 180, 320, 290, 180, 40, 5, 8],
    ))
    results = run_simulation(years=10, climate_months=climate,
                              carbon_input_annual=2.5, initial_soc=30.0)
    socs = [r["total_soc"] for r in results]
    max_jump = max(abs(socs[i + 1] - socs[i]) for i in range(len(socs) - 1))
    assert max_jump < 2.0


def test_higher_carbon_input_produces_higher_end_soc():
    """A sanity/monotonicity check: more residue input should mean more ending SOC,
    all else equal."""
    climate = list(zip(
        [15.0, 18.5, 24.0, 29.5, 32.0, 31.5, 28.5, 28.0, 27.5, 24.5, 19.5, 15.5],
        [15, 12, 8, 5, 40, 180, 320, 290, 180, 40, 5, 8],
    ))
    low = run_simulation(years=5, climate_months=climate,
                          carbon_input_annual=2.0, initial_soc=30.0)
    high = run_simulation(years=5, climate_months=climate,
                           carbon_input_annual=15.0, initial_soc=30.0)
    assert high[-1]["total_soc"] > low[-1]["total_soc"]
