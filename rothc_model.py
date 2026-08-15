"""
RothC (Rothamsted Carbon Model) - simplified monthly implementation
Based on Coleman & Jenkinson's RothC-26.3 structure

Pools:
  DPM - Decomposable Plant Material (fast)
  RPM - Resistant Plant Material (slow)
  BIO - Microbial Biomass
  HUM - Humified Organic Matter
  IOM - Inert Organic Matter (does not decompose)
"""
import numpy as np

# Decomposition rate constants (per year), standard RothC values
K_DPM = 10.0
K_RPM = 0.3
K_BIO = 0.66
K_HUM = 0.02

def temp_factor(temp_c):
    """RothC temperature rate modifier."""
    if temp_c < -5:
        return 0.0
    return 47.9 / (1 + np.exp(106.0 / (temp_c + 18.3)))

def moisture_factor(rain_mm, evap_mm=None, is_bare=False):
    """Simplified RothC moisture rate modifier based on rainfall.
    Uses a simplified soil moisture deficit proxy (full RothC uses TSMD)."""
    if evap_mm is None:
        evap_mm = 0.75 * rain_mm if rain_mm > 0 else 20  # rough PET proxy
    deficit = max(0, evap_mm - rain_mm)
    max_deficit = 25.0 if not is_bare else 25.0 * 0.556
    if deficit <= 0.444 * max_deficit:
        return 1.0
    b = 0.2 + (1.0 - 0.2) * (max_deficit - deficit) / (max_deficit - 0.444 * max_deficit)
    return max(0.2, min(1.0, b))

def cover_factor(is_vegetated):
    """RothC plant-cover rate modifier."""
    return 0.6 if is_vegetated else 1.0

def rothc_step(dpm, rpm, bio, hum, iom, temp_c, rain_mm, is_vegetated,
               carbon_input, dpm_rpm_ratio=1.44, clay_pct=23.0):
    """Run one month of RothC decomposition and return updated pools."""
    a = temp_factor(temp_c)
    b = moisture_factor(rain_mm, is_bare=not is_vegetated)
    c = cover_factor(is_vegetated)
    rate_mod = a * b * c / 12.0  # convert annual rate constants to monthly

    # Decomposition of each pool
    dpm_decomp = dpm * (1 - np.exp(-K_DPM * rate_mod))
    rpm_decomp = rpm * (1 - np.exp(-K_RPM * rate_mod))
    bio_decomp = bio * (1 - np.exp(-K_BIO * rate_mod))
    hum_decomp = hum * (1 - np.exp(-K_HUM * rate_mod))

    total_decomp = dpm_decomp + rpm_decomp + bio_decomp + hum_decomp

    # Clay-dependent split: CO2 lost vs. (BIO+HUM) formed
    x = 1.67 * (1.85 + 1.60 * np.exp(-0.0786 * clay_pct))
    co2_fraction = x / (x + 1)
    bio_hum_fraction = 1 / (x + 1)

    new_bio_hum = total_decomp * bio_hum_fraction
    new_bio = new_bio_hum * 0.46
    new_hum = new_bio_hum * 0.54

    # Update pools after decomposition
    dpm -= dpm_decomp
    rpm -= rpm_decomp
    bio = bio - bio_decomp + new_bio
    hum = hum - hum_decomp + new_hum

    # Add fresh carbon input, split by DPM:RPM ratio
    dpm_input = carbon_input * (dpm_rpm_ratio / (1 + dpm_rpm_ratio))
    rpm_input = carbon_input * (1 / (1 + dpm_rpm_ratio))
    dpm += dpm_input
    rpm += rpm_input

    return dpm, rpm, bio, hum, iom


def run_simulation(years, climate_months, carbon_input_annual, initial_soc,
                    clay_pct=23.0):
    """Run RothC over N years using a repeating 12-month climate cycle.
    initial_soc: total starting soil organic carbon (t C/ha)
    carbon_input_annual: annual fresh carbon input from crop residues/roots (t C/ha/yr)
    Returns a list of (year, month, total_soc, dpm, rpm, bio, hum, iom)
    """
    # Initialize pools using typical RothC equilibrium proportions
    iom = 0.049 * (initial_soc ** 1.139)  # Falloon et al. IOM estimate
    remaining = initial_soc - iom
    dpm = remaining * 0.015
    rpm = remaining * 0.115
    bio = remaining * 0.02
    hum = remaining * 0.85

    monthly_input = carbon_input_annual / 12.0
    results = []

    for year in range(1, years + 1):
        for m_idx, (temp_c, rain_mm) in enumerate(climate_months):
            month_name = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m_idx]
            is_vegetated = month_name in ['Jun','Jul','Aug','Sep','Oct']  # kharif rice season
            dpm, rpm, bio, hum, iom = rothc_step(
                dpm, rpm, bio, hum, iom, temp_c, rain_mm, is_vegetated,
                monthly_input, clay_pct=clay_pct
            )
            total_soc = dpm + rpm + bio + hum + iom
            results.append({
                'year': year, 'month': month_name,
                'total_soc': total_soc, 'dpm': dpm, 'rpm': rpm,
                'bio': bio, 'hum': hum, 'iom': iom
            })
    return results

if __name__ == "__main__":
    months_climate = list(zip(
        [15.0, 18.5, 24.0, 29.5, 32.0, 31.5, 28.5, 28.0, 27.5, 24.5, 19.5, 15.5],
        [15, 12, 8, 5, 40, 180, 320, 290, 180, 40, 5, 8]
    ))
    results = run_simulation(years=10, climate_months=months_climate,
                              carbon_input_annual=2.5, initial_soc=30.0)
    print(f"Start SOC: 30.0 t C/ha")
    print(f"End SOC (year 10): {results[-1]['total_soc']:.2f} t C/ha")
    print(f"Change: {results[-1]['total_soc'] - 30.0:+.2f} t C/ha over 10 years")
