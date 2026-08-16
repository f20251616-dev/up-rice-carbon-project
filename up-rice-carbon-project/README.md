# UP Rice Farm Carbon MRV Pipeline

A geospatial + process-based modeling pipeline for estimating soil organic carbon (SOC) dynamics on smallholder rice farms in Bahraich district, Uttar Pradesh -- built as a technical portfolio project demonstrating the core components of an agricultural carbon MRV (Measurement, Reporting, Verification) system.

## What this does

1. **Geospatial MRV (`notebooks/01_geospatial_ndvi_zonal_stats.ipynb`)** -- loads hand-digitized farm plot boundaries, pulls real Sentinel-2 satellite imagery via Google Earth Engine, computes NDVI (a vegetation health index), and runs zonal statistics to get a per-farm crop health score.

2. **Process-based soil carbon modeling (`notebooks/02_rothc_soil_carbon_model.ipynb`)** -- implements the RothC (Rothamsted Carbon Model) soil carbon model in Python, simulating how soil carbon pools evolve monthly based on temperature, rainfall, and crop cover.

3. **Calibration, validation & Bayesian surrogate (`notebooks/03_calibration_and_gp_surrogate.ipynb`)** -- calibrates the model against a real, literature-sourced SOC sequestration rate, validates the calibration on a held-out time period, and trains a Gaussian Process surrogate to approximate the model's output orders of magnitude faster.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── src/                          # Reusable, importable modules
│   ├── geospatial.py             # NDVI + zonal stats functions
│   ├── rothc_model.py            # RothC soil carbon model
│   └── calibration.py            # Calibration + GP surrogate functions
├── notebooks/                    # Exploration and demonstration notebooks
│   ├── 01_geospatial_ndvi_zonal_stats.ipynb
│   ├── 02_rothc_soil_carbon_model.ipynb
│   └── 03_calibration_and_gp_surrogate.ipynb
├── data/                         # Sample farm boundaries and satellite imagery
│   ├── up_rice_farms_clean.geojson
│   ├── up_rice_farms_sentinel2.tif
│   └── ndvi_output.tif
└── tests/                        # pytest test suite
    ├── test_rothc_model.py
    └── test_geospatial.py
```

## Setup

```bash
git clone <this-repo-url>
cd up-rice-carbon-project
pip install -r requirements.txt
```

## Reproducing the notebooks

Each notebook is self-contained and can be run top to bottom:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_geospatial_ndvi_zonal_stats.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_rothc_soil_carbon_model.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_calibration_and_gp_surrogate.ipynb
```

Or open them directly in Jupyter Lab/Notebook and run cell by cell.

## Running tests

```bash
pytest tests/ -v
```

13 tests covering: RothC's temperature/moisture/cover response functions, mass-conservation sanity checks (e.g. the inert carbon pool never changes), simulation stability (no negative or erratic SOC values), and geospatial functions (NDVI stays within [-1, 1], zonal stats produce one row per farm).

## Key design decisions and honest scope notes

- **Farm boundaries are hand-digitized**, not real farmer GPS data -- this is a demonstration dataset over a real agricultural area in Bahraich district, not production farmer records.
- **The process-based model is RothC, not DNDC/DayCent.** DNDC and DayCent are licensed, Windows-only desktop tools; RothC is a peer-reviewed, open-equation model implemented directly from its published formulation. The modeling principles (pool-based decomposition, calibration, validation) transfer directly, but the specific pool structure and equations differ from DNDC/DayCent.
- **The moisture response function is a simplified single-month approximation**, not RothC's full running Topsoil Moisture Deficit (TSMD) carryover between months -- a reasonable simplification for this scope, called out explicitly rather than left implicit.
- **The calibration target (0.34 t C/ha/yr) is sourced from a published cross-regional synthesis** of continuous paddy management SOC sequestration rates, not an arbitrary assumption.
- **Validation revealed a genuine finding, not just a pass/fail check**: calibrating on years 1-7 and checking years 8-10 showed the SOC gain rate decelerates over time due to pool-saturation dynamics, which means a literature-reported "average annual rate" shouldn't be assumed constant when extrapolating a process model beyond its calibration window.

## Author

Built by Krishna as part of a technical portfolio for carbon markets / agricultural GHG modeling roles.
