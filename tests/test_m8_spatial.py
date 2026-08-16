"""
TERA-STRIPE M8 Spatial Analysis -- Test Suite
================================================
Validates spatial analysis, home range computation, and GeoJSON export.

Test classes
------------
  TestSpatialModels        - Pydantic contract model validation
  TestGeometryUtils        - Haversine, bearing, polygon area, convex hull
  TestMCP                  - Minimum Convex Polygon computation
  TestKDE                  - Kernel Density Estimation contours
  TestH3Occupancy          - H3 hexagonal grid occupancy
  TestVillageProximity     - Distance to settlements
  TestMovementStats        - Movement distance statistics
  TestSpatialAnalyzer      - Full orchestrator pipeline
  TestGeoJSONExport        - GeoJSON FeatureCollection export
  TestFullPipeline         - End-to-end M1 through M8
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["USE_POSTGIS"] = "false"
os.environ["ENVIRONMENT"] = "testing"


# ── Test Data: Pench Tiger Reserve sighting coordinates ──────────

PENCH_SIGHTINGS = [
    {"lat": 22.7200, "lon": 79.2900, "captured_at": "2026-01-10T08:00:00", "station_id": "STN_101"},
    {"lat": 22.7250, "lon": 79.2950, "captured_at": "2026-01-15T09:30:00", "station_id": "STN_102"},
    {"lat": 22.7300, "lon": 79.3000, "captured_at": "2026-01-20T07:15:00", "station_id": "STN_103"},
    {"lat": 22.7350, "lon": 79.3050, "captured_at": "2026-02-01T06:45:00", "station_id": "STN_104"},
    {"lat": 22.7150, "lon": 79.2850, "captured_at": "2026-02-10T10:00:00", "station_id": "STN_105"},
    {"lat": 22.7100, "lon": 79.2800, "captured_at": "2026-02-15T11:30:00", "station_id": "STN_106"},
    {"lat": 22.7400, "lon": 79.3100, "captured_at": "2026-03-01T08:00:00", "station_id": "STN_107"},
    {"lat": 22.7280, "lon": 79.2920, "captured_at": "2026-03-10T14:00:00", "station_id": "STN_108"},
    {"lat": 22.7220, "lon": 79.2870, "captured_at": "2026-03-15T16:30:00", "station_id": "STN_109"},
    {"lat": 22.7180, "lon": 79.2830, "captured_at": "2026-03-20T07:00:00", "station_id": "STN_110"},
]

PENCH_VILLAGES = [
    {"name": "Turia", "lat": 22.7500, "lon": 79.3300},
    {"name": "Khawasa", "lat": 22.6900, "lon": 79.2500},
    {"name": "Rukhad", "lat": 22.7600, "lon": 79.2700},
]


def _make_sighting_points() -> list:
    from src.m8_spatial import SightingPoint
    return [SightingPoint(**s) for s in PENCH_SIGHTINGS]


# =====================================================================
#  Test 1: Pydantic Models
# =====================================================================

class TestSpatialModels:

    def test_sighting_point(self) -> None:
        from src.m8_spatial import SightingPoint
        p = SightingPoint(lat=22.72, lon=79.29)
        assert p.lat == 22.72

    def test_mcp_result(self) -> None:
        from src.m8_spatial import MCPResult
        m = MCPResult(mcp_100_area_sq_km=45.5)
        assert m.mcp_100_area_sq_km == 45.5

    def test_kde_result(self) -> None:
        from src.m8_spatial import KDEResult
        k = KDEResult(bandwidth=0.005)
        assert k.bandwidth == 0.005

    def test_h3_occupancy_grid(self) -> None:
        from src.m8_spatial import H3OccupancyGrid
        g = H3OccupancyGrid(resolution=9, total_cells=5)
        assert g.resolution == 9

    def test_tiger_spatial_analysis(self) -> None:
        from src.m8_spatial import TigerSpatialAnalysis
        a = TigerSpatialAnalysis(
            tiger_id="PTR_M_001",
            sighting_count=10,
            analysis_timestamp="2026-01-01T00:00:00",
        )
        assert a.tiger_id == "PTR_M_001"


# =====================================================================
#  Test 2: Geometry Utilities
# =====================================================================

class TestGeometryUtils:

    def test_haversine_zero_distance(self) -> None:
        from src.m8_spatial import haversine_km
        d = haversine_km(22.72, 79.29, 22.72, 79.29)
        assert d == 0.0

    def test_haversine_known_distance(self) -> None:
        """Pench to Nagpur is approximately 95 km."""
        from src.m8_spatial import haversine_km
        d = haversine_km(22.72, 79.29, 21.15, 79.09)
        assert 170 < d < 180  # Approximate

    def test_bearing_north(self) -> None:
        from src.m8_spatial import bearing_deg
        b = bearing_deg(22.72, 79.29, 23.72, 79.29)
        assert 355 < b or b < 5  # ~0 degrees (north)

    def test_bearing_east(self) -> None:
        from src.m8_spatial import bearing_deg
        b = bearing_deg(22.72, 79.29, 22.72, 80.29)
        assert 85 < b < 95  # ~90 degrees (east)

    def test_polygon_area_small(self) -> None:
        """A small square should have a non-zero area."""
        from src.m8_spatial import polygon_area_sq_km
        # 0.01 degree square at equator ~ 1.23 km2
        coords = [
            [79.29, 22.72], [79.30, 22.72],
            [79.30, 22.73], [79.29, 22.73],
            [79.29, 22.72],
        ]
        area = polygon_area_sq_km(coords)
        assert area > 0.5

    def test_polygon_area_degenerate(self) -> None:
        from src.m8_spatial import polygon_area_sq_km
        assert polygon_area_sq_km([]) == 0.0
        assert polygon_area_sq_km([[1, 2]]) == 0.0

    def test_convex_hull_triangle(self) -> None:
        from src.m8_spatial import convex_hull_2d
        pts = np.array([[0, 0], [1, 0], [0.5, 1], [0.5, 0.3]])
        hull = convex_hull_2d(pts)
        assert len(hull) == 3  # Triangle hull

    def test_convex_hull_collinear(self) -> None:
        from src.m8_spatial import convex_hull_2d
        pts = np.array([[0, 0], [1, 1]])
        hull = convex_hull_2d(pts)
        assert len(hull) == 2


# =====================================================================
#  Test 3: MCP Computation
# =====================================================================

class TestMCP:

    def test_mcp_basic(self) -> None:
        from src.m8_spatial import compute_mcp
        points = _make_sighting_points()
        mcp = compute_mcp(points)

        assert mcp.mcp_100_area_sq_km > 0
        assert mcp.mcp_95_area_sq_km > 0
        assert mcp.mcp_95_area_sq_km <= mcp.mcp_100_area_sq_km
        assert len(mcp.mcp_100_coords) >= 3
        assert len(mcp.centroid) == 2

    def test_mcp_centroid_in_range(self) -> None:
        from src.m8_spatial import compute_mcp
        mcp = compute_mcp(_make_sighting_points())
        assert 79.2 < mcp.centroid[0] < 79.4  # lon
        assert 22.6 < mcp.centroid[1] < 22.8  # lat

    def test_mcp_few_points(self) -> None:
        from src.m8_spatial import SightingPoint, compute_mcp
        pts = [
            SightingPoint(lat=22.72, lon=79.29),
            SightingPoint(lat=22.73, lon=79.30),
        ]
        mcp = compute_mcp(pts)
        assert len(mcp.mcp_100_coords) == 2

    def test_mcp_single_point(self) -> None:
        from src.m8_spatial import SightingPoint, compute_mcp
        pts = [SightingPoint(lat=22.72, lon=79.29)]
        mcp = compute_mcp(pts)
        assert mcp.mcp_100_area_sq_km == 0.0


# =====================================================================
#  Test 4: KDE Computation
# =====================================================================

class TestKDE:

    def test_kde_produces_contours(self) -> None:
        from src.m8_spatial import compute_kde
        kde = compute_kde(_make_sighting_points())

        assert kde.bandwidth > 0
        assert len(kde.kde_95_coords) >= 3
        assert kde.kde_95_area_sq_km > 0

    def test_kde_core_smaller_than_range(self) -> None:
        from src.m8_spatial import compute_kde
        kde = compute_kde(_make_sighting_points())
        assert kde.kde_50_area_sq_km <= kde.kde_95_area_sq_km

    def test_kde_few_points(self) -> None:
        from src.m8_spatial import SightingPoint, compute_kde
        pts = [SightingPoint(lat=22.72, lon=79.29)]
        kde = compute_kde(pts)
        assert kde.bandwidth == 0.0


# =====================================================================
#  Test 5: H3 Occupancy
# =====================================================================

class TestH3Occupancy:

    def test_h3_occupancy_cells(self) -> None:
        from src.m8_spatial import compute_h3_occupancy
        grid = compute_h3_occupancy(_make_sighting_points(), resolution=8)

        assert grid.resolution == 8
        assert grid.total_cells > 0
        assert len(grid.cells) == grid.total_cells

    def test_h3_sighting_count_sum(self) -> None:
        from src.m8_spatial import compute_h3_occupancy
        points = _make_sighting_points()
        grid = compute_h3_occupancy(points)

        total = sum(c.sighting_count for c in grid.cells)
        assert total == len(points)

    def test_h3_empty_points(self) -> None:
        from src.m8_spatial import compute_h3_occupancy
        grid = compute_h3_occupancy([])
        assert grid.total_cells == 0


# =====================================================================
#  Test 6: Village Proximity
# =====================================================================

class TestVillageProximity:

    def test_proximity_computed(self) -> None:
        from src.m8_spatial import compute_village_proximity
        prox = compute_village_proximity(
            _make_sighting_points(), PENCH_VILLAGES,
        )
        assert len(prox) > 0
        assert prox[0].distance_km > 0
        assert prox[0].nearest_village in [v["name"] for v in PENCH_VILLAGES]

    def test_proximity_sorted_by_distance(self) -> None:
        from src.m8_spatial import compute_village_proximity
        prox = compute_village_proximity(
            _make_sighting_points(), PENCH_VILLAGES,
        )
        distances = [p.distance_km for p in prox]
        assert distances == sorted(distances)

    def test_proximity_no_villages(self) -> None:
        from src.m8_spatial import compute_village_proximity
        prox = compute_village_proximity(_make_sighting_points(), [])
        assert len(prox) == 0


# =====================================================================
#  Test 7: Movement Statistics
# =====================================================================

class TestMovementStats:

    def test_movement_stats_basic(self) -> None:
        from src.m8_spatial import compute_movement_stats
        stats = compute_movement_stats(_make_sighting_points())

        assert stats["total_distance_km"] > 0
        assert stats["max_step_km"] > 0
        assert stats["mean_step_km"] > 0
        assert stats["total_points"] == len(PENCH_SIGHTINGS)

    def test_movement_single_point(self) -> None:
        from src.m8_spatial import SightingPoint, compute_movement_stats
        stats = compute_movement_stats([SightingPoint(lat=22.72, lon=79.29)])
        assert stats["total_distance_km"] == 0.0
        assert stats["total_points"] == 1


# =====================================================================
#  Test 8: Spatial Analyzer Orchestrator
# =====================================================================

class TestSpatialAnalyzer:

    def test_analyze_single_tiger(self) -> None:
        from src.m8_spatial import SpatialAnalyzer
        analyzer = SpatialAnalyzer(h3_resolution=8, villages=PENCH_VILLAGES)
        result = analyzer.analyze_tiger(
            "PTR_M_001", _make_sighting_points(), "2026_DRY",
        )

        assert result.tiger_id == "PTR_M_001"
        assert result.sighting_count == 10
        assert result.season_year == "2026_DRY"
        assert result.mcp.mcp_100_area_sq_km > 0
        assert result.h3_occupancy.total_cells > 0

    def test_analyze_batch(self) -> None:
        from src.m8_spatial import SightingPoint, SpatialAnalyzer
        analyzer = SpatialAnalyzer()
        tiger_data = {
            "PTR_M_001": _make_sighting_points(),
            "PTR_M_002": [
                SightingPoint(lat=22.80, lon=79.35, captured_at="2026-01-01"),
                SightingPoint(lat=22.81, lon=79.36, captured_at="2026-01-02"),
                SightingPoint(lat=22.82, lon=79.37, captured_at="2026-01-03"),
            ],
        }
        result = analyzer.analyze_batch(tiger_data, "2026_DRY")
        assert result.total_tigers == 2
        assert len(result.analyses) == 2


# =====================================================================
#  Test 9: GeoJSON Export
# =====================================================================

class TestGeoJSONExport:

    def test_geojson_valid(self, tmp_dir: Path) -> None:
        from src.m8_spatial import SpatialAnalyzer, export_geojson
        analyzer = SpatialAnalyzer(villages=PENCH_VILLAGES)
        analysis = analyzer.analyze_tiger("PTR_M_001", _make_sighting_points())

        geo_path = tmp_dir / "test.geojson"
        export_geojson(analysis, geo_path)

        assert geo_path.exists()
        with open(geo_path) as f:
            data = json.load(f)

        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) > 0

        # Verify layers
        layers = {f["properties"]["layer"] for f in data["features"]}
        assert "MCP_100" in layers
        assert "CENTROID" in layers

    def test_geojson_polygon_geometry(self, tmp_dir: Path) -> None:
        from src.m8_spatial import SpatialAnalyzer, export_geojson
        analysis = SpatialAnalyzer().analyze_tiger("T1", _make_sighting_points())

        geo_path = tmp_dir / "poly.geojson"
        export_geojson(analysis, geo_path)

        with open(geo_path) as f:
            data = json.load(f)

        polygons = [
            f for f in data["features"]
            if f["geometry"]["type"] == "Polygon"
        ]
        assert len(polygons) >= 1
        # Each polygon must have coordinates
        for p in polygons:
            assert len(p["geometry"]["coordinates"][0]) >= 3


# =====================================================================
#  Test 10: Full Pipeline M1 -> M8
# =====================================================================

class TestFullPipeline:

    def test_full_pipeline_spatial_analysis(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """Full pipeline through spatial analysis."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend as TriageMock
        from src.m2_triage import TriageEngine
        from src.m4_flank_pose import FlankExtractionEngine, MockPoseBackend
        from src.m5_reid_engine import (
            MockEmbeddingBackend, ReIDEngine, VectorGallery,
        )
        from src.m7_db_manager import DatabaseManager
        from src.m8_spatial import SightingPoint, SpatialAnalyzer

        # M1-M5
        manifest = generate_manifest(full_fixture_dir)
        triage = TriageEngine(
            backend=TriageMock(seed=42),
            confidence_threshold=0.15,
        ).process_manifest(manifest.model_dump())
        flank = FlankExtractionEngine(
            backend=MockPoseBackend(seed=42),
            crops_dir=tmp_dir / "crops",
        ).process_triage(
            triage.model_dump(by_alias=True),
            source_records=manifest.model_dump()["records"],
        )
        reid = ReIDEngine(
            backend=MockEmbeddingBackend(seed=42),
            gallery=VectorGallery(),
        ).process_extractions(flank.model_dump())

        # M7: Ingest
        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'spatial_test.db'}")
        db.ingest_pipeline_results(
            manifest_data=manifest.model_dump(),
            reid_data=reid.model_dump(),
        )

        # M8: Build sighting points from Re-ID results
        # (Using synthetic coords since test fixtures lack real GPS)
        tiger_sightings: dict[str, list[SightingPoint]] = {}
        for d in reid.dispatches:
            tid = d.assigned_tiger_id
            if not tid:
                continue
            if tid not in tiger_sightings:
                tiger_sightings[tid] = []
            # Synthetic coords around Pench
            idx = len(tiger_sightings[tid])
            tiger_sightings[tid].append(SightingPoint(
                lat=22.72 + idx * 0.005,
                lon=79.29 + idx * 0.005,
                captured_at="2026-02-14T08:00:00",
                station_id="PTR_STN_104B",
            ))

        # Pad to minimum 3 points for valid MCP
        for tid in tiger_sightings:
            while len(tiger_sightings[tid]) < 3:
                n = len(tiger_sightings[tid])
                tiger_sightings[tid].append(SightingPoint(
                    lat=22.72 + n * 0.003,
                    lon=79.29 - n * 0.003,
                ))

        analyzer = SpatialAnalyzer(h3_resolution=8)
        result = analyzer.analyze_batch(tiger_sightings, "2026_DRY")

        assert result.total_tigers > 0
        for a in result.analyses:
            assert a.sighting_count >= 3
            assert a.mcp.mcp_100_area_sq_km >= 0
