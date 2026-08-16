"""
Tests for geospatial processing functions.

Run with: pytest tests/
Requires the sample data files in data/ to be present.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.geospatial import load_farm_boundaries, compute_ndvi, compute_zonal_ndvi

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FARM_GEOJSON = os.path.join(DATA_DIR, "up_rice_farms_clean.geojson")
SENTINEL_TIF = os.path.join(DATA_DIR, "up_rice_farms_sentinel2.tif")


@pytest.mark.skipif(not os.path.exists(FARM_GEOJSON), reason="sample data not present")
def test_load_farm_boundaries_drops_empty_rows():
    gdf = load_farm_boundaries(FARM_GEOJSON)
    assert gdf["farm_id"].notna().all()
    assert "area_ha" in gdf.columns


@pytest.mark.skipif(not os.path.exists(FARM_GEOJSON), reason="sample data not present")
def test_farm_areas_are_positive_and_realistic():
    gdf = load_farm_boundaries(FARM_GEOJSON)
    assert (gdf["area_ha"] > 0).all()
    # Smallholder plots should be well under 10 ha each
    assert (gdf["area_ha"] < 10).all()


@pytest.mark.skipif(
    not (os.path.exists(FARM_GEOJSON) and os.path.exists(SENTINEL_TIF)),
    reason="sample data not present",
)
def test_ndvi_output_within_valid_range(tmp_path):
    ndvi_path = str(tmp_path / "ndvi_test.tif")
    compute_ndvi(SENTINEL_TIF, ndvi_path)

    import rasterio
    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1)

    valid = ndvi[~np.isnan(ndvi)]
    assert valid.min() >= -1.0
    assert valid.max() <= 1.0


@pytest.mark.skipif(
    not (os.path.exists(FARM_GEOJSON) and os.path.exists(SENTINEL_TIF)),
    reason="sample data not present",
)
def test_zonal_ndvi_has_one_row_per_farm(tmp_path):
    gdf = load_farm_boundaries(FARM_GEOJSON)
    ndvi_path = str(tmp_path / "ndvi_test.tif")
    compute_ndvi(SENTINEL_TIF, ndvi_path)

    result = compute_zonal_ndvi(gdf, ndvi_path)
    assert len(result) == len(gdf)
    assert "ndvi_mean" in result.columns
