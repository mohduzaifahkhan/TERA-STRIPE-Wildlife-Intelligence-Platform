"""
TERA-STRIPE Module 8 -- Spatial Analysis & Home Range Engine
==============================================================
Computes per-tiger territory polygons (MCP / KDE), H3 hexagonal
occupancy grids, movement corridor heuristics, and village proximity
distances from sighting coordinates.

Spatial methods:
  MCP  -- Minimum Convex Polygon (95% / 100%)
  KDE  -- Kernel Density Estimation (50% core / 95% range)
  H3   -- Uber H3 hexagonal grid occupancy

Data contract:
  Input  : List of sighting coordinates per tiger (from M7 DB)
  Output : spatial_analysis.json + GeoJSON exports

CLI Usage
---------
  python -m src.m8_spatial \\
      --db-url sqlite:///tera_stripe.db \\
      --tiger-id PTR_M_001 \\
      --output-dir ./data/spatial

  # Analyse all tigers
  python -m src.m8_spatial \\
      --db-url sqlite:///tera_stripe.db \\
      --all --output-dir ./data/spatial

Reference: Master Context Packet -- Home Range, H3 Occupancy, Village Proximity
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("tera_stripe.m8_spatial")

IST = timezone(timedelta(hours=5, minutes=30))

# ── Earth radius for distance calculations (km) ─────────────────
EARTH_RADIUS_KM = 6371.0


# =====================================================================
#  Pydantic Contract Models
# =====================================================================

class SightingPoint(BaseModel):
    """A single georeferenced sighting."""
    lat: float
    lon: float
    captured_at: str = ""
    station_id: str = ""
    sighting_id: str = ""


class MCPResult(BaseModel):
    """Minimum Convex Polygon result."""
    mcp_100_coords: list[list[float]] = []  # [[lon,lat], ...]
    mcp_95_coords: list[list[float]] = []
    mcp_100_area_sq_km: float = 0.0
    mcp_95_area_sq_km: float = 0.0
    centroid: list[float] = [0.0, 0.0]  # [lon, lat]


class KDEResult(BaseModel):
    """Kernel Density Estimation result."""
    kde_50_coords: list[list[float]] = []  # Core area contour
    kde_95_coords: list[list[float]] = []  # Range contour
    kde_50_area_sq_km: float = 0.0
    kde_95_area_sq_km: float = 0.0
    bandwidth: float = 0.0


class H3OccupancyCell(BaseModel):
    """A single H3 cell with occupancy data."""
    h3_index: str
    sighting_count: int
    center_lat: float
    center_lon: float


class H3OccupancyGrid(BaseModel):
    """H3 hexagonal occupancy grid."""
    resolution: int = 8
    total_cells: int = 0
    cells: list[H3OccupancyCell] = []


class VillageProximity(BaseModel):
    """Proximity to nearest village/settlement."""
    nearest_village: str = ""
    distance_km: float = 0.0
    bearing_deg: float = 0.0
    sighting_point: list[float] = [0.0, 0.0]  # [lon, lat]


class TigerSpatialAnalysis(BaseModel):
    """Complete spatial analysis for a single tiger."""
    tiger_id: str
    sighting_count: int
    analysis_timestamp: str
    season_year: str = ""
    mcp: MCPResult = Field(default_factory=MCPResult)
    kde: KDEResult = Field(default_factory=KDEResult)
    h3_occupancy: H3OccupancyGrid = Field(default_factory=H3OccupancyGrid)
    village_proximity: list[VillageProximity] = []
    movement_stats: dict = Field(default_factory=dict)


class SpatialAnalysisResult(BaseModel):
    """Batch spatial analysis output."""
    batch_id: str = ""
    total_tigers: int = 0
    analyses: list[TigerSpatialAnalysis] = []


# =====================================================================
#  Geometry Utilities
# =====================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two lat/lon points in kilometres."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2 in degrees."""
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = (
        math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
        - math.sin(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.cos(dlon)
    )
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def polygon_area_sq_km(coords: list[list[float]]) -> float:
    """
    Approximate area of a lat/lon polygon using the Shoelace formula
    with local tangent plane projection.

    Parameters
    ----------
    coords : list[[lon, lat], ...]
        Polygon vertices in [longitude, latitude] order.

    Returns
    -------
    float
        Area in square kilometres.
    """
    if len(coords) < 3:
        return 0.0

    # Use centroid as projection origin
    lats = [c[1] for c in coords]
    lons = [c[0] for c in coords]
    c_lat = sum(lats) / len(lats)
    c_lon = sum(lons) / len(lons)

    # Convert to local km coordinates
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(c_lat))

    xs = [(lon - c_lon) * km_per_deg_lon for lon in lons]
    ys = [(lat - c_lat) * km_per_deg_lat for lat in lats]

    # Shoelace formula
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """
    Compute the convex hull of a 2D point set using Graham scan.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)

    Returns
    -------
    np.ndarray, shape (M, 2) -- hull vertices in CCW order
    """
    if len(points) < 3:
        return points.copy()

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        return points[hull.vertices]
    except (ImportError, Exception):
        pass

    # Fallback: simple Graham scan
    pts = sorted(points.tolist(), key=lambda p: (p[0], p[1]))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1])


# =====================================================================
#  MCP Computation
# =====================================================================

def compute_mcp(
    points: list[SightingPoint],
) -> MCPResult:
    """
    Compute Minimum Convex Polygon (100% and 95%) from sighting points.

    MCP-95 removes the 5% most distant points from the centroid
    before computing the hull.
    """
    if len(points) < 3:
        coords = [[p.lon, p.lat] for p in points]
        centroid = (
            [sum(c[0] for c in coords) / len(coords),
             sum(c[1] for c in coords) / len(coords)]
            if coords else [0.0, 0.0]
        )
        return MCPResult(
            mcp_100_coords=coords,
            mcp_95_coords=coords,
            centroid=centroid,
        )

    arr = np.array([[p.lon, p.lat] for p in points])

    # MCP-100
    hull_100 = convex_hull_2d(arr)
    coords_100 = hull_100.tolist()
    # Close the polygon
    if coords_100 and coords_100[0] != coords_100[-1]:
        coords_100.append(coords_100[0])

    area_100 = polygon_area_sq_km(coords_100)
    centroid = [float(arr[:, 0].mean()), float(arr[:, 1].mean())]

    # MCP-95: Remove 5% most distant points from centroid
    distances = np.sqrt(
        (arr[:, 0] - centroid[0]) ** 2 + (arr[:, 1] - centroid[1]) ** 2
    )
    n_keep = max(3, int(len(points) * 0.95))
    keep_idx = np.argsort(distances)[:n_keep]
    arr_95 = arr[keep_idx]

    hull_95 = convex_hull_2d(arr_95)
    coords_95 = hull_95.tolist()
    if coords_95 and coords_95[0] != coords_95[-1]:
        coords_95.append(coords_95[0])

    area_95 = polygon_area_sq_km(coords_95)

    return MCPResult(
        mcp_100_coords=[[round(c[0], 6), round(c[1], 6)] for c in coords_100],
        mcp_95_coords=[[round(c[0], 6), round(c[1], 6)] for c in coords_95],
        mcp_100_area_sq_km=round(area_100, 4),
        mcp_95_area_sq_km=round(area_95, 4),
        centroid=[round(centroid[0], 6), round(centroid[1], 6)],
    )


# =====================================================================
#  KDE Computation (simplified grid-based)
# =====================================================================

def compute_kde(
    points: list[SightingPoint],
    grid_size: int = 50,
) -> KDEResult:
    """
    Compute simplified KDE contours for 50% core and 95% range.

    Uses Gaussian kernel density estimation on a projected grid.
    """
    if len(points) < 3:
        return KDEResult()

    lats = np.array([p.lat for p in points])
    lons = np.array([p.lon for p in points])

    # Bandwidth: Silverman's rule of thumb
    n = len(points)
    std_lat = np.std(lats)
    std_lon = np.std(lons)
    if std_lat < 1e-8 or std_lon < 1e-8:
        return KDEResult()

    bw = ((4 / (3 * n)) ** 0.2) * max(std_lat, std_lon)

    # Create evaluation grid
    pad = bw * 3
    lat_min, lat_max = lats.min() - pad, lats.max() + pad
    lon_min, lon_max = lons.min() - pad, lons.max() + pad

    lat_grid = np.linspace(lat_min, lat_max, grid_size)
    lon_grid = np.linspace(lon_min, lon_max, grid_size)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Evaluate density
    density = np.zeros_like(lon_mesh)
    for lat_i, lon_i in zip(lats, lons):
        d_lat = (lat_mesh - lat_i) / bw
        d_lon = (lon_mesh - lon_i) / bw
        density += np.exp(-0.5 * (d_lat**2 + d_lon**2))

    density /= density.max() + 1e-10

    # Extract contours at 50% and 95% levels
    def extract_contour(level: float) -> list[list[float]]:
        mask = density >= level
        if not mask.any():
            return []
        coords = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    coords.append([
                        round(float(lon_mesh[i, j]), 6),
                        round(float(lat_mesh[i, j]), 6),
                    ])
        if len(coords) < 3:
            return coords
        # Return convex hull of contour points
        arr = np.array(coords)
        hull = convex_hull_2d(arr)
        result = hull.tolist()
        if result and result[0] != result[-1]:
            result.append(result[0])
        return [[round(c[0], 6), round(c[1], 6)] for c in result]

    kde_50 = extract_contour(0.50)
    kde_95 = extract_contour(0.05)

    area_50 = polygon_area_sq_km(kde_50)
    area_95 = polygon_area_sq_km(kde_95)

    return KDEResult(
        kde_50_coords=kde_50,
        kde_95_coords=kde_95,
        kde_50_area_sq_km=round(area_50, 4),
        kde_95_area_sq_km=round(area_95, 4),
        bandwidth=round(float(bw), 6),
    )


# =====================================================================
#  H3 Hexagonal Occupancy
# =====================================================================

def compute_h3_occupancy(
    points: list[SightingPoint],
    resolution: int = 8,
) -> H3OccupancyGrid:
    """
    Compute H3 hexagonal occupancy grid from sighting points.

    Supports both H3 v4 and v3 APIs with graceful fallback.
    """
    cell_counts: dict[str, int] = {}
    cell_centers: dict[str, tuple[float, float]] = {}

    for pt in points:
        h3_index = _h3_index(pt.lat, pt.lon, resolution)
        if h3_index:
            cell_counts[h3_index] = cell_counts.get(h3_index, 0) + 1
            center = _h3_center(h3_index)
            if center:
                cell_centers[h3_index] = center

    cells = []
    for idx, count in sorted(cell_counts.items(), key=lambda x: -x[1]):
        center = cell_centers.get(idx, (0.0, 0.0))
        cells.append(H3OccupancyCell(
            h3_index=idx,
            sighting_count=count,
            center_lat=round(center[0], 6),
            center_lon=round(center[1], 6),
        ))

    return H3OccupancyGrid(
        resolution=resolution,
        total_cells=len(cells),
        cells=cells,
    )


def _h3_index(lat: float, lon: float, res: int) -> str:
    """Get H3 index with version-safe fallback."""
    try:
        import h3
        if hasattr(h3, "latlng_to_cell"):
            return h3.latlng_to_cell(lat, lon, res)
        elif hasattr(h3, "geo_to_h3"):
            return h3.geo_to_h3(lat, lon, res)
    except Exception:
        pass
    # Fallback: synthetic index
    lat_q = int(lat * 1000)
    lon_q = int(lon * 1000)
    return f"8{res:x}{lat_q:06x}{lon_q:06x}ff"


def _h3_center(h3_index: str) -> tuple[float, float] | None:
    """Get center of H3 cell."""
    try:
        import h3
        if hasattr(h3, "cell_to_latlng"):
            return h3.cell_to_latlng(h3_index)
        elif hasattr(h3, "h3_to_geo"):
            return h3.h3_to_geo(h3_index)
    except Exception:
        pass
    return (0.0, 0.0)


# =====================================================================
#  Village Proximity
# =====================================================================

def compute_village_proximity(
    points: list[SightingPoint],
    villages: list[dict],
) -> list[VillageProximity]:
    """
    Compute proximity of sighting points to known villages.

    Parameters
    ----------
    villages : list[dict]
        Each dict: {"name": str, "lat": float, "lon": float}

    Returns
    -------
    list[VillageProximity]
        One entry per unique closest-village sighting.
    """
    if not villages or not points:
        return []

    results = []
    seen_pairs: set[str] = set()

    for pt in points:
        closest = None
        min_dist = float("inf")

        for v in villages:
            d = haversine_km(pt.lat, pt.lon, v["lat"], v["lon"])
            if d < min_dist:
                min_dist = d
                closest = v

        if closest:
            pair_key = f"{pt.station_id}_{closest['name']}"
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                results.append(VillageProximity(
                    nearest_village=closest["name"],
                    distance_km=round(min_dist, 3),
                    bearing_deg=round(
                        bearing_deg(pt.lat, pt.lon, closest["lat"], closest["lon"]),
                        1,
                    ),
                    sighting_point=[round(pt.lon, 6), round(pt.lat, 6)],
                ))

    return sorted(results, key=lambda v: v.distance_km)


# =====================================================================
#  Movement Statistics
# =====================================================================

def compute_movement_stats(
    points: list[SightingPoint],
) -> dict:
    """
    Compute basic movement statistics from chronological sightings.
    """
    if len(points) < 2:
        return {
            "total_distance_km": 0.0,
            "max_step_km": 0.0,
            "mean_step_km": 0.0,
            "total_points": len(points),
        }

    # Sort by time
    sorted_pts = sorted(points, key=lambda p: p.captured_at)

    steps = []
    for i in range(1, len(sorted_pts)):
        d = haversine_km(
            sorted_pts[i - 1].lat, sorted_pts[i - 1].lon,
            sorted_pts[i].lat, sorted_pts[i].lon,
        )
        steps.append(d)

    return {
        "total_distance_km": round(sum(steps), 3),
        "max_step_km": round(max(steps), 3) if steps else 0.0,
        "mean_step_km": round(sum(steps) / len(steps), 3) if steps else 0.0,
        "total_points": len(points),
    }


# =====================================================================
#  GeoJSON Export
# =====================================================================

def export_geojson(
    analysis: TigerSpatialAnalysis,
    output_path: Path,
) -> None:
    """Export spatial analysis as a GeoJSON FeatureCollection."""
    features = []

    # MCP-100 polygon
    if len(analysis.mcp.mcp_100_coords) >= 3:
        features.append({
            "type": "Feature",
            "properties": {
                "tiger_id": analysis.tiger_id,
                "layer": "MCP_100",
                "area_sq_km": analysis.mcp.mcp_100_area_sq_km,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [analysis.mcp.mcp_100_coords],
            },
        })

    # MCP-95 polygon
    if len(analysis.mcp.mcp_95_coords) >= 3:
        features.append({
            "type": "Feature",
            "properties": {
                "tiger_id": analysis.tiger_id,
                "layer": "MCP_95",
                "area_sq_km": analysis.mcp.mcp_95_area_sq_km,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [analysis.mcp.mcp_95_coords],
            },
        })

    # KDE-95 polygon
    if len(analysis.kde.kde_95_coords) >= 3:
        features.append({
            "type": "Feature",
            "properties": {
                "tiger_id": analysis.tiger_id,
                "layer": "KDE_95",
                "area_sq_km": analysis.kde.kde_95_area_sq_km,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [analysis.kde.kde_95_coords],
            },
        })

    # Centroid point
    if analysis.mcp.centroid != [0.0, 0.0]:
        features.append({
            "type": "Feature",
            "properties": {
                "tiger_id": analysis.tiger_id,
                "layer": "CENTROID",
            },
            "geometry": {
                "type": "Point",
                "coordinates": analysis.mcp.centroid,
            },
        })

    # H3 cell centres
    for cell in analysis.h3_occupancy.cells:
        features.append({
            "type": "Feature",
            "properties": {
                "tiger_id": analysis.tiger_id,
                "layer": "H3_CELL",
                "h3_index": cell.h3_index,
                "sighting_count": cell.sighting_count,
            },
            "geometry": {
                "type": "Point",
                "coordinates": [cell.center_lon, cell.center_lat],
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    logger.info(
        "GeoJSON exported: %s (%d features)", output_path, len(features)
    )


# =====================================================================
#  Spatial Analyzer (Orchestrator)
# =====================================================================

class SpatialAnalyzer:
    """
    Orchestrates spatial analysis for one or more tigers.

    Combines MCP, KDE, H3 occupancy, village proximity,
    and movement statistics into a unified analysis result.
    """

    def __init__(
        self,
        h3_resolution: int = 8,
        villages: list[dict] | None = None,
    ) -> None:
        self.h3_resolution = h3_resolution
        self.villages = villages or []

    def analyze_tiger(
        self,
        tiger_id: str,
        sightings: list[SightingPoint],
        season_year: str = "",
    ) -> TigerSpatialAnalysis:
        """Run full spatial analysis for a single tiger."""
        now = datetime.now(IST).isoformat()

        logger.info(
            "Spatial analysis | tiger=%s | sightings=%d",
            tiger_id,
            len(sightings),
        )

        mcp = compute_mcp(sightings)
        kde = compute_kde(sightings)
        h3_grid = compute_h3_occupancy(sightings, self.h3_resolution)
        village_prox = compute_village_proximity(sightings, self.villages)
        movement = compute_movement_stats(sightings)

        return TigerSpatialAnalysis(
            tiger_id=tiger_id,
            sighting_count=len(sightings),
            analysis_timestamp=now,
            season_year=season_year,
            mcp=mcp,
            kde=kde,
            h3_occupancy=h3_grid,
            village_proximity=village_prox,
            movement_stats=movement,
        )

    def analyze_batch(
        self,
        tiger_sightings: dict[str, list[SightingPoint]],
        season_year: str = "",
    ) -> SpatialAnalysisResult:
        """Run spatial analysis for multiple tigers."""
        analyses = []
        for tiger_id, sightings in tiger_sightings.items():
            analysis = self.analyze_tiger(tiger_id, sightings, season_year)
            analyses.append(analysis)

        return SpatialAnalysisResult(
            batch_id=f"SPATIAL_{datetime.now(IST).strftime('%Y%m%d')}",
            total_tigers=len(analyses),
            analyses=analyses,
        )


# =====================================================================
#  CLI Entry Point
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="m8_spatial",
        description="TERA-STRIPE M8 -- Spatial Analysis & Home Range",
    )

    parser.add_argument("--tiger-id", type=str, default=None)
    parser.add_argument("--all", action="store_true", help="Analyse all tigers.")
    parser.add_argument(
        "--db-url", type=str, default=None,
        help="Database URL for sighting queries.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("./data/spatial"),
    )
    parser.add_argument(
        "--sightings-json", type=Path, default=None,
        help="Sightings JSON file (alternative to DB).",
    )
    parser.add_argument("--h3-res", type=int, default=8)
    parser.add_argument(
        "--villages-json", type=Path, default=None,
        help="Villages JSON: [{name, lat, lon}, ...]",
    )
    parser.add_argument("--geojson", action="store_true", help="Export GeoJSON.")

    args = parser.parse_args()

    # Load villages
    villages = []
    if args.villages_json and args.villages_json.exists():
        with open(args.villages_json) as f:
            villages = json.load(f)

    analyzer = SpatialAnalyzer(h3_resolution=args.h3_res, villages=villages)

    # Load sightings from JSON or DB
    tiger_sightings: dict[str, list[SightingPoint]] = {}

    if args.sightings_json and args.sightings_json.exists():
        with open(args.sightings_json) as f:
            data = json.load(f)
        for tid, pts in data.items():
            tiger_sightings[tid] = [SightingPoint(**p) for p in pts]
    elif args.db_url:
        tiger_sightings = _load_from_db(args.db_url, args.tiger_id)
    else:
        logger.error("Provide --sightings-json or --db-url.")
        sys.exit(1)

    if args.tiger_id and args.tiger_id in tiger_sightings:
        tiger_sightings = {args.tiger_id: tiger_sightings[args.tiger_id]}

    result = analyzer.analyze_batch(tiger_sightings)

    # Write results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "spatial_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2)

    # GeoJSON export
    if args.geojson:
        for analysis in result.analyses:
            geo_path = args.output_dir / f"{analysis.tiger_id}_home_range.geojson"
            export_geojson(analysis, geo_path)

    for a in result.analyses:
        print(
            f"  {a.tiger_id}: {a.sighting_count} sightings | "
            f"MCP-100: {a.mcp.mcp_100_area_sq_km:.2f} km2 | "
            f"MCP-95: {a.mcp.mcp_95_area_sq_km:.2f} km2 | "
            f"H3 cells: {a.h3_occupancy.total_cells}"
        )


def _load_from_db(
    db_url: str, tiger_id: str | None
) -> dict[str, list[SightingPoint]]:
    """Load sightings from the database."""
    from src.m7_database import TigerSighting, get_engine, get_session_factory

    engine = get_engine(url=db_url)
    Session = get_session_factory(engine)

    result: dict[str, list[SightingPoint]] = {}

    with Session() as session:
        query = session.query(TigerSighting)
        if tiger_id:
            query = query.filter(TigerSighting.tiger_id == tiger_id)
        sightings = query.all()

        for s in sightings:
            tid = s.tiger_id
            if not tid:
                continue
            if tid not in result:
                result[tid] = []
            # Parse WKT point for lat/lon
            lat, lon = _parse_wkt_point(s.geom or "")
            result[tid].append(SightingPoint(
                lat=lat,
                lon=lon,
                captured_at=s.captured_at.isoformat() if s.captured_at else "",
                station_id=s.station_id or "",
                sighting_id=s.sighting_id,
            ))

    return result


def _parse_wkt_point(wkt: str) -> tuple[float, float]:
    """Parse POINT(lon lat) WKT to (lat, lon)."""
    try:
        inner = wkt.replace("POINT(", "").replace(")", "").strip()
        parts = inner.split()
        return float(parts[1]), float(parts[0])
    except (IndexError, ValueError):
        return 0.0, 0.0


if __name__ == "__main__":
    main()
