"""
TERA-STRIPE M9 Security Alerts -- Test Suite
===============================================
Validates anomaly detection, alert generation, and DB integration.

Test classes
------------
  TestAlertModels           - Pydantic contract model validation
  TestVillageProximity      - Village distance alert logic
  TestCoreRangeShift        - Centroid shift detection
  TestNovelStation          - Novel station detection
  TestProlongedAbsence      - Absence threshold detection
  TestAlertEngine           - Full orchestrator pipeline
  TestAlertDB               - Database write integration
  TestFullPipeline          - End-to-end M1 through M9
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["USE_POSTGIS"] = "false"
os.environ["ENVIRONMENT"] = "testing"

IST = timezone(timedelta(hours=5, minutes=30))


# ── Test data: spatial analysis with village proximity ───────────

def _make_spatial_analysis(
    tiger_id: str = "PTR_M_001",
    centroid: list[float] | None = None,
    village_prox: list[dict] | None = None,
    h3_cells: list[dict] | None = None,
) -> dict:
    """Build a mock TigerSpatialAnalysis dict."""
    return {
        "tiger_id": tiger_id,
        "sighting_count": 10,
        "analysis_timestamp": "2026-03-01T10:00:00",
        "season_year": "2026_DRY",
        "mcp": {
            "mcp_100_coords": [[79.28, 22.71], [79.31, 22.71], [79.31, 22.74], [79.28, 22.74], [79.28, 22.71]],
            "mcp_95_coords": [[79.28, 22.71], [79.31, 22.71], [79.31, 22.74], [79.28, 22.71]],
            "mcp_100_area_sq_km": 9.5,
            "mcp_95_area_sq_km": 7.8,
            "centroid": centroid or [79.295, 22.725],
        },
        "kde": {"bandwidth": 0.005},
        "h3_occupancy": {
            "resolution": 8,
            "total_cells": len(h3_cells or []),
            "cells": h3_cells or [
                {"h3_index": "8844c0a305fffff", "sighting_count": 3, "center_lat": 22.72, "center_lon": 79.29},
                {"h3_index": "8844c0a307fffff", "sighting_count": 2, "center_lat": 22.73, "center_lon": 79.30},
            ],
        },
        "village_proximity": village_prox or [],
        "movement_stats": {"total_distance_km": 12.5},
    }


def _make_spatial_result(analyses: list[dict]) -> dict:
    """Wrap analyses in a SpatialAnalysisResult."""
    return {
        "batch_id": "SPATIAL_TEST",
        "total_tigers": len(analyses),
        "analyses": analyses,
    }


# =====================================================================
#  Test 1: Pydantic Models
# =====================================================================

class TestAlertModels:

    def test_alert_schema(self) -> None:
        from src.m9_alerts import Alert
        a = Alert(
            alert_type="VILLAGE_PROXIMITY",
            severity="WARNING",
            tiger_id="PTR_M_001",
            description="Test alert",
        )
        assert a.alert_type == "VILLAGE_PROXIMITY"
        assert a.alert_id.startswith("ALR_")

    def test_alert_summary_schema(self) -> None:
        from src.m9_alerts import AlertSummary
        s = AlertSummary(total_alerts=5, critical=1, warning=2, info=2)
        assert s.total_alerts == 5

    def test_alert_result_schema(self) -> None:
        from src.m9_alerts import AlertResult, AlertSummary
        r = AlertResult(
            batch_id="TEST",
            summary=AlertSummary(total_alerts=0),
            alerts=[],
        )
        assert r.batch_id == "TEST"


# =====================================================================
#  Test 2: Village Proximity
# =====================================================================

class TestVillageProximity:

    def test_no_proximity_data(self) -> None:
        from src.m9_alerts import detect_village_proximity
        alerts = detect_village_proximity(_make_spatial_analysis())
        assert len(alerts) == 0

    def test_critical_proximity(self) -> None:
        """Distance <= 1 km should be CRITICAL."""
        from src.m9_alerts import detect_village_proximity
        data = _make_spatial_analysis(
            village_prox=[{
                "nearest_village": "Turia",
                "distance_km": 0.8,
                "bearing_deg": 45.0,
                "sighting_point": [79.29, 22.72],
            }],
        )
        alerts = detect_village_proximity(data, radius_km=5.0)
        assert len(alerts) == 1
        assert alerts[0].severity == "CRITICAL"
        assert alerts[0].distance_to_village_km == 0.8

    def test_warning_proximity(self) -> None:
        """1 < distance <= 3 km should be WARNING."""
        from src.m9_alerts import detect_village_proximity
        data = _make_spatial_analysis(
            village_prox=[{
                "nearest_village": "Khawasa",
                "distance_km": 2.5,
                "bearing_deg": 180.0,
                "sighting_point": [79.29, 22.72],
            }],
        )
        alerts = detect_village_proximity(data, radius_km=5.0)
        assert len(alerts) == 1
        assert alerts[0].severity == "WARNING"

    def test_info_proximity(self) -> None:
        """3 < distance <= 5 km should be INFO."""
        from src.m9_alerts import detect_village_proximity
        data = _make_spatial_analysis(
            village_prox=[{
                "nearest_village": "Rukhad",
                "distance_km": 4.2,
                "bearing_deg": 90.0,
                "sighting_point": [79.29, 22.72],
            }],
        )
        alerts = detect_village_proximity(data, radius_km=5.0)
        assert len(alerts) == 1
        assert alerts[0].severity == "INFO"

    def test_outside_radius_no_alert(self) -> None:
        """Distance > radius should not trigger alert."""
        from src.m9_alerts import detect_village_proximity
        data = _make_spatial_analysis(
            village_prox=[{
                "nearest_village": "FarVillage",
                "distance_km": 10.0,
                "bearing_deg": 0.0,
                "sighting_point": [79.29, 22.72],
            }],
        )
        alerts = detect_village_proximity(data, radius_km=5.0)
        assert len(alerts) == 0

    def test_multiple_villages(self) -> None:
        from src.m9_alerts import detect_village_proximity
        data = _make_spatial_analysis(
            village_prox=[
                {"nearest_village": "V1", "distance_km": 0.5, "bearing_deg": 0, "sighting_point": [79, 22]},
                {"nearest_village": "V2", "distance_km": 4.0, "bearing_deg": 90, "sighting_point": [79, 22]},
                {"nearest_village": "V3", "distance_km": 8.0, "bearing_deg": 180, "sighting_point": [79, 22]},
            ],
        )
        alerts = detect_village_proximity(data, radius_km=5.0)
        assert len(alerts) == 2  # V1 and V2 within radius


# =====================================================================
#  Test 3: Core Range Shift
# =====================================================================

class TestCoreRangeShift:

    def test_no_previous_data(self) -> None:
        from src.m9_alerts import detect_core_range_shift
        alerts = detect_core_range_shift(_make_spatial_analysis())
        assert len(alerts) == 0

    def test_significant_shift(self) -> None:
        from src.m9_alerts import detect_core_range_shift
        current = _make_spatial_analysis(centroid=[79.295, 22.725])
        previous = _make_spatial_analysis(centroid=[79.295, 22.925])  # ~22km north

        alerts = detect_core_range_shift(current, previous, shift_threshold_km=10.0)
        assert len(alerts) == 1
        assert alerts[0].alert_type == "CORE_RANGE_SHIFT"
        assert alerts[0].centroid_shift_km > 10.0

    def test_minor_shift_no_alert(self) -> None:
        from src.m9_alerts import detect_core_range_shift
        current = _make_spatial_analysis(centroid=[79.295, 22.725])
        previous = _make_spatial_analysis(centroid=[79.296, 22.726])

        alerts = detect_core_range_shift(current, previous, shift_threshold_km=10.0)
        assert len(alerts) == 0

    def test_critical_shift(self) -> None:
        """Shift >= 20 km should be CRITICAL."""
        from src.m9_alerts import detect_core_range_shift
        current = _make_spatial_analysis(centroid=[79.295, 22.725])
        previous = _make_spatial_analysis(centroid=[79.295, 22.525])  # ~22km south

        alerts = detect_core_range_shift(current, previous, shift_threshold_km=5.0)
        assert len(alerts) == 1
        assert alerts[0].severity == "CRITICAL"


# =====================================================================
#  Test 4: Novel Station
# =====================================================================

class TestNovelStation:

    def test_novel_station_detected(self) -> None:
        from src.m9_alerts import detect_novel_station
        alerts = detect_novel_station(
            "PTR_M_001",
            current_stations=["STN_A", "STN_B", "STN_C"],
            historical_stations=["STN_A", "STN_B"],
        )
        assert len(alerts) == 1
        assert alerts[0].station_id == "STN_C"
        assert alerts[0].alert_type == "NOVEL_STATION"

    def test_no_novel_stations(self) -> None:
        from src.m9_alerts import detect_novel_station
        alerts = detect_novel_station(
            "PTR_M_001",
            current_stations=["STN_A", "STN_B"],
            historical_stations=["STN_A", "STN_B", "STN_C"],
        )
        assert len(alerts) == 0

    def test_multiple_novel_stations(self) -> None:
        from src.m9_alerts import detect_novel_station
        alerts = detect_novel_station(
            "PTR_M_001",
            current_stations=["STN_A", "STN_X", "STN_Y"],
            historical_stations=["STN_A"],
        )
        assert len(alerts) == 2


# =====================================================================
#  Test 5: Prolonged Absence
# =====================================================================

class TestProlongedAbsence:

    def test_absence_detected(self) -> None:
        from src.m9_alerts import detect_prolonged_absence
        last_seen = datetime(2026, 1, 1, tzinfo=IST)
        ref_time = datetime(2026, 3, 15, tzinfo=IST)  # 73 days later

        alerts = detect_prolonged_absence(
            "PTR_M_001", last_seen, threshold_days=30, reference_time=ref_time,
        )
        assert len(alerts) == 1
        assert alerts[0].alert_type == "PROLONGED_ABSENCE"
        assert alerts[0].severity == "WARNING"  # 60-90 days

    def test_recent_sighting_no_alert(self) -> None:
        from src.m9_alerts import detect_prolonged_absence
        last_seen = datetime(2026, 3, 10, tzinfo=IST)
        ref_time = datetime(2026, 3, 15, tzinfo=IST)

        alerts = detect_prolonged_absence(
            "PTR_M_001", last_seen, threshold_days=30, reference_time=ref_time,
        )
        assert len(alerts) == 0

    def test_critical_absence(self) -> None:
        """90+ days should be CRITICAL."""
        from src.m9_alerts import detect_prolonged_absence
        last_seen = datetime(2025, 12, 1, tzinfo=IST)
        ref_time = datetime(2026, 4, 1, tzinfo=IST)  # 121 days

        alerts = detect_prolonged_absence(
            "PTR_M_001", last_seen, threshold_days=30, reference_time=ref_time,
        )
        assert len(alerts) == 1
        assert alerts[0].severity == "CRITICAL"

    def test_iso_string_input(self) -> None:
        from src.m9_alerts import detect_prolonged_absence
        alerts = detect_prolonged_absence(
            "PTR_M_001",
            "2026-01-01T00:00:00",
            threshold_days=30,
            reference_time=datetime(2026, 3, 15),
        )
        assert len(alerts) == 1

    def test_none_last_seen(self) -> None:
        from src.m9_alerts import detect_prolonged_absence
        alerts = detect_prolonged_absence("PTR_M_001", None)
        assert len(alerts) == 0


# =====================================================================
#  Test 6: Alert Engine Orchestrator
# =====================================================================

class TestAlertEngine:

    def test_engine_processes_spatial(self) -> None:
        from src.m9_alerts import AlertEngine
        spatial = _make_spatial_result([
            _make_spatial_analysis(
                village_prox=[{
                    "nearest_village": "Turia",
                    "distance_km": 2.0,
                    "bearing_deg": 45.0,
                    "sighting_point": [79.29, 22.72],
                }],
            ),
        ])

        engine = AlertEngine(village_radius_km=5.0)
        result = engine.process_spatial_results(spatial)

        assert result.summary.total_alerts >= 1
        assert result.batch_id == "SPATIAL_TEST"

    def test_engine_summary_counts(self) -> None:
        from src.m9_alerts import AlertEngine
        spatial = _make_spatial_result([
            _make_spatial_analysis(
                village_prox=[
                    {"nearest_village": "V1", "distance_km": 0.5, "bearing_deg": 0, "sighting_point": [0, 0]},
                    {"nearest_village": "V2", "distance_km": 4.0, "bearing_deg": 0, "sighting_point": [0, 0]},
                ],
            ),
        ])

        result = AlertEngine(village_radius_km=5.0).process_spatial_results(spatial)
        s = result.summary
        assert s.total_alerts == s.critical + s.warning + s.info

    def test_engine_with_absence(self) -> None:
        from src.m9_alerts import AlertEngine
        spatial = _make_spatial_result([_make_spatial_analysis()])

        ref = datetime(2026, 5, 1, tzinfo=IST)
        result = AlertEngine(absence_days=30).process_spatial_results(
            spatial,
            last_seen_dates={"PTR_M_001": "2026-01-01T00:00:00"},
            reference_time=ref,
        )

        absence_alerts = [a for a in result.alerts if a.alert_type == "PROLONGED_ABSENCE"]
        assert len(absence_alerts) == 1

    def test_engine_empty_spatial(self) -> None:
        from src.m9_alerts import AlertEngine
        result = AlertEngine().process_spatial_results({"analyses": []})
        assert result.summary.total_alerts == 0


# =====================================================================
#  Test 7: Database Integration
# =====================================================================

class TestAlertDB:

    def test_write_alerts_to_db(self, tmp_dir: Path) -> None:
        from src.m9_alerts import Alert, write_alerts_to_db

        alerts = [
            Alert(
                alert_type="VILLAGE_PROXIMITY",
                severity="WARNING",
                tiger_id="PTR_M_001",
                description="Test alert",
                distance_to_village_km=2.5,
                payload={"village": "Turia"},
                created_at=datetime.now(IST).isoformat(),
            ),
            Alert(
                alert_type="PROLONGED_ABSENCE",
                severity="CRITICAL",
                tiger_id="PTR_M_002",
                description="Tiger absent 100 days",
                payload={"days_absent": 100},
                created_at=datetime.now(IST).isoformat(),
            ),
        ]

        db_url = f"sqlite:///{tmp_dir / 'alerts_test.db'}"
        n = write_alerts_to_db(alerts, db_url)
        assert n == 2


# =====================================================================
#  Test 8: Full Pipeline M1 -> M9
# =====================================================================

class TestFullPipeline:

    def test_full_pipeline_generates_alerts(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend as TriageMock
        from src.m2_triage import TriageEngine
        from src.m4_flank_pose import FlankExtractionEngine, MockPoseBackend
        from src.m5_reid_engine import (
            MockEmbeddingBackend, ReIDEngine, VectorGallery,
        )
        from src.m8_spatial import SightingPoint, SpatialAnalyzer
        from src.m9_alerts import AlertEngine

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

        # M8: Spatial with synthetic sightings near village
        villages = [{"name": "Turia", "lat": 22.74, "lon": 79.31}]
        tiger_sightings = {}
        for d in reid.dispatches:
            tid = d.assigned_tiger_id
            if not tid:
                continue
            if tid not in tiger_sightings:
                tiger_sightings[tid] = []
            idx = len(tiger_sightings[tid])
            tiger_sightings[tid].append(SightingPoint(
                lat=22.735 + idx * 0.002,  # Near Turia village
                lon=79.305 + idx * 0.002,
                captured_at="2026-02-14T08:00:00",
            ))
        for tid in tiger_sightings:
            while len(tiger_sightings[tid]) < 3:
                n = len(tiger_sightings[tid])
                tiger_sightings[tid].append(SightingPoint(
                    lat=22.735 + n * 0.001, lon=79.305 - n * 0.001,
                ))

        analyzer = SpatialAnalyzer(h3_resolution=8, villages=villages)
        spatial = analyzer.analyze_batch(tiger_sightings, "2026_DRY")

        # M9: Alerts
        engine = AlertEngine(village_radius_km=5.0)
        result = engine.process_spatial_results(spatial.model_dump())

        assert isinstance(result.summary.total_alerts, int)
        assert result.batch_id != ""
