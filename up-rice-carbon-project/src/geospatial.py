"""
Geospatial processing utilities for the UP Rice Carbon MRV pipeline.

Handles loading farm boundary vector data, computing NDVI from Sentinel-2
raster bands, and running zonal statistics to summarize NDVI per farm plot.
"""

import geopandas as gpd
import numpy as np
import rasterio
from rasterstats import zonal_stats


def load_farm_boundaries(geojson_path: str, drop_empty: bool = True) -> gpd.GeoDataFrame:
    """Load farm plot boundaries from a GeoJSON file.

    Args:
        geojson_path: Path to a GeoJSON file containing farm plot polygons.
        drop_empty: If True, drop rows with missing/empty identifiers or geometry.

    Returns:
        A GeoDataFrame with a 'farm_id' column and an 'area_ha' column
        (computed in an appropriate local UTM projection).
    """
    gdf = gpd.read_file(geojson_path)

    if drop_empty:
        id_col = "farm_id" if "farm_id" in gdf.columns else "name"
        gdf = gdf[gdf[id_col].notna()].copy()
        if id_col != "farm_id":
            gdf = gdf.rename(columns={id_col: "farm_id"})

    gdf = gdf.reset_index(drop=True)

    gdf_projected = gdf.to_crs(gdf.estimate_utm_crs())
    gdf["area_ha"] = (gdf_projected.area / 10000).round(3)

    return gdf


def check_crs_alignment(vector_gdf: gpd.GeoDataFrame, raster_path: str) -> gpd.GeoDataFrame:
    """Verify vector and raster CRS match; reproject the vector if not.

    Args:
        vector_gdf: A GeoDataFrame of farm boundaries.
        raster_path: Path to the raster file to check alignment against.

    Returns:
        The vector GeoDataFrame, reprojected to match the raster's CRS if needed.
    """
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    if vector_gdf.crs != raster_crs:
        return vector_gdf.to_crs(raster_crs)
    return vector_gdf


def compute_ndvi(raster_path: str, output_path: str, red_band: int = 3, nir_band: int = 4) -> str:
    """Compute NDVI from a multi-band raster and save it as a new single-band raster.

    Args:
        raster_path: Path to the input multi-band GeoTIFF (must include Red and NIR bands).
        output_path: Path to write the output single-band NDVI GeoTIFF.
        red_band: 1-indexed band number for the Red band (default 3, i.e. Sentinel-2 B4).
        nir_band: 1-indexed band number for the Near-Infrared band (default 4, i.e. Sentinel-2 B8).

    Returns:
        The output_path, for convenience chaining.
    """
    with rasterio.open(raster_path) as src:
        red = src.read(red_band).astype("float32")
        nir = src.read(nir_band).astype("float32")
        profile = src.profile

    np.seterr(divide="ignore", invalid="ignore")
    ndvi = (nir - red) / (nir + red)
    ndvi = np.where((nir + red) == 0, np.nan, ndvi)

    ndvi_profile = profile.copy()
    ndvi_profile.update(count=1, dtype="float32", nodata=np.nan)

    with rasterio.open(output_path, "w", **ndvi_profile) as dst:
        dst.write(ndvi, 1)

    return output_path


def compute_zonal_ndvi(farm_gdf: gpd.GeoDataFrame, ndvi_raster_path: str) -> gpd.GeoDataFrame:
    """Compute zonal NDVI statistics (mean, min, max, std, pixel count) per farm plot.

    Args:
        farm_gdf: GeoDataFrame of farm boundaries (CRS must match the NDVI raster).
        ndvi_raster_path: Path to a single-band NDVI GeoTIFF.

    Returns:
        A copy of farm_gdf with added columns: ndvi_mean, ndvi_min, ndvi_max,
        ndvi_std, pixel_count.
    """
    stats = zonal_stats(
        farm_gdf, ndvi_raster_path,
        stats=["mean", "min", "max", "std", "count"],
        nodata=float("nan"),
    )

    result = farm_gdf.copy()
    for i, s in enumerate(stats):
        result.loc[i, "ndvi_mean"] = s["mean"]
        result.loc[i, "ndvi_min"] = s["min"]
        result.loc[i, "ndvi_max"] = s["max"]
        result.loc[i, "ndvi_std"] = s["std"]
        result.loc[i, "pixel_count"] = s["count"]

    return result
