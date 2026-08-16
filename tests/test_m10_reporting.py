"""
TERA-STRIPE M10 Reporting -- Test Suite
=========================================
Validates reporting engine, census exports, and dashboard feeds.

Test classes
------------
  TestReportModels        - Pydantic contract model validation
  TestKPIMetrics          - KPI computation (mock and DB)
  TestStorageROI          - Storage/labor savings calculation
  TestCensusExport        - NTCA census CSV export
  TestTigerDossier        - Per-tiger dossier generation
  TestDashboardFeed       - Dashboard data feed assembly
  TestFullPipeline        - End-to-end M1 through M10
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["USE_POSTGIS"] = "false"
os.environ["ENVIRONMENT"] = "testing"


# =====================================================================
#  Test 1: Pydantic Models
# =====================================================================

class TestReportModels:

    def test_kpi_schema(self) -> None:
        from src.m10_reporting import KPIMetrics
        k = KPIMetrics(active_stations=104, tracked_tigers=48)
        assert k.active_stations == 104

    def test_census_record_schema(self) -> None:
        from src.m10_reporting import CensusRecord
        r = CensusRecord(
            tiger_id="PTR_M_001",
            total_sightings=25,
            home_range_mcp95_sq_km=42.5,
        )
        assert r.reserve == "Pench Tiger Reserve"

    def test_storage_roi_schema(self) -> None:
        from src.m10_reporting import StorageROI
        r = StorageROI(storage_saved_gb=142.6, estimated_review_hours_saved=45.2)
        assert r.storage_saved_gb == 142.6

    def test_dashboard_feed_schema(self) -> None:
        from src.m10_reporting import DashboardFeed
        f = DashboardFeed(generated_at="2026-08-16T10:00:00")
        assert f.tigers == []

    def test_tiger_dossier_schema(self) -> None:
        from src.m10_reporting import TigerDossier
        d = TigerDossier(tiger_id="PTR_M_001", home_range_sq_km=42.5)
        assert d.tiger_id == "PTR_M_001"


# =====================================================================
#  Test 2: KPI Metrics
# =====================================================================

class TestKPIMetrics:

    def test_mock_kpis(self) -> None:
        from src.m10_reporting import ReportGenerator
        gen = ReportGenerator()
        kpi = gen.mock_kpis()

        assert kpi.active_stations == 104
        assert kpi.tracked_tigers == 48
        assert kpi.storage_saved_gb > 0

    def test_compute_kpis_no_db(self) -> None:
        from src.m10_reporting import ReportGenerator
        gen = ReportGenerator()  # No DB
        kpi = gen.compute_kpis()
        assert kpi.active_stations > 0  # Falls back to mock

    def test_compute_kpis_with_db(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager
        from src.m10_reporting import ReportGenerator

        db_url = f"sqlite:///{tmp_dir / 'kpi_test.db'}"
        db = DatabaseManager(db_url=db_url)
        db.create_or_update_tiger("PTR_M_001")
        db.create_or_update_tiger("PTR_M_002")

        gen = ReportGenerator(db_url=db_url)
        kpi = gen.compute_kpis()
        assert kpi.tracked_tigers == 2


# =====================================================================
#  Test 3: Storage ROI
# =====================================================================

class TestStorageROI:

    def test_roi_calculation(self) -> None:
        from src.m10_reporting import ReportGenerator
        gen = ReportGenerator()
        roi = gen.compute_storage_roi(
            total_images=5000,
            blanks_quarantined=3842,
            avg_image_size_mb=4.5,
            review_seconds_per_image=45.0,
            auto_matches=910,
            review_queue=12,
            new_individuals=48,
        )

        assert roi.storage_saved_gb > 0
        assert roi.estimated_review_hours_saved > 0
        assert roi.auto_match_rate_pct > 0
        assert roi.total_images_processed == 5000

    def test_roi_zero_images(self) -> None:
        from src.m10_reporting import ReportGenerator
        roi = ReportGenerator().compute_storage_roi()
        assert roi.storage_saved_gb == 0.0

    def test_roi_auto_rate_calculation(self) -> None:
        from src.m10_reporting import ReportGenerator
        roi = ReportGenerator().compute_storage_roi(
            auto_matches=80, review_queue=10, new_individuals=10,
        )
        assert roi.auto_match_rate_pct == 80.0


# =====================================================================
#  Test 4: NTCA Census Export
# =====================================================================

class TestCensusExport:

    def test_mock_census_export(self, tmp_dir: Path) -> None:
        from src.m10_reporting import ReportGenerator
        gen = ReportGenerator()
        out = tmp_dir / "census.csv"
        n = gen.export_ntca_census(out)

        assert n > 0
        assert out.exists()

        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == n
        assert "tiger_id" in rows[0]
        assert "home_range_mcp95_sq_km" in rows[0]

    def test_census_with_db(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager
        from src.m10_reporting import ReportGenerator

        db_url = f"sqlite:///{tmp_dir / 'census_db.db'}"
        db = DatabaseManager(db_url=db_url)
        db.create_or_update_tiger("PTR_M_001")
        db.create_or_update_tiger("PTR_M_002")

        gen = ReportGenerator(db_url=db_url)
        out = tmp_dir / "census_db.csv"
        n = gen.export_ntca_census(out)
        assert n == 2

    def test_census_csv_structure(self, tmp_dir: Path) -> None:
        from src.m10_reporting import ReportGenerator
        out = tmp_dir / "struct.csv"
        ReportGenerator().export_ntca_census(out)

        with open(out) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        expected = ["tiger_id", "common_name", "sex", "status",
                     "first_detected", "last_detected", "total_sightings",
                     "stations_visited", "home_range_mcp95_sq_km",
                     "core_territory_kde50_sq_km", "reserve"]
        assert headers == expected


# =====================================================================
#  Test 5: Tiger Dossier
# =====================================================================

class TestTigerDossier:

    def test_dossier_no_db(self) -> None:
        from src.m10_reporting import ReportGenerator
        dossier = ReportGenerator().generate_tiger_dossier("PTR_M_001")
        assert dossier.tiger_id == "PTR_M_001"

    def test_dossier_with_db(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager
        from src.m10_reporting import ReportGenerator

        db_url = f"sqlite:///{tmp_dir / 'dossier.db'}"
        db = DatabaseManager(db_url=db_url)
        db.create_or_update_tiger("PTR_M_001", common_name="Bajrang", sex="MALE")

        gen = ReportGenerator(db_url=db_url)
        dossier = gen.generate_tiger_dossier("PTR_M_001")
        assert dossier.common_name == "Bajrang"


# =====================================================================
#  Test 6: Dashboard Feed
# =====================================================================

class TestDashboardFeed:

    def test_feed_generation(self) -> None:
        from src.m10_reporting import ReportGenerator
        gen = ReportGenerator()
        feed = gen.generate_dashboard_feed()

        assert feed.generated_at != ""
        assert feed.kpi.active_stations > 0
        assert len(feed.stations) > 0
        assert len(feed.alerts) > 0

    def test_feed_json_roundtrip(self, tmp_dir: Path) -> None:
        from src.m10_reporting import DashboardFeed, ReportGenerator
        feed = ReportGenerator().generate_dashboard_feed()

        out = tmp_dir / "feed.json"
        with open(out, "w") as f:
            json.dump(feed.model_dump(), f, indent=2)
        with open(out) as f:
            restored = DashboardFeed(**json.load(f))

        assert restored.kpi.tracked_tigers == feed.kpi.tracked_tigers


# =====================================================================
#  Test 7: Full Pipeline M1 -> M10
# =====================================================================

class TestFullPipeline:

    def test_full_pipeline_reporting(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend as TriageMock
        from src.m2_triage import TriageEngine
        from src.m4_flank_pose import FlankExtractionEngine, MockPoseBackend
        from src.m5_reid_engine import (
            MockEmbeddingBackend, ReIDEngine, VectorGallery,
        )
        from src.m7_db_manager import DatabaseManager
        from src.m10_reporting import ReportGenerator

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

        # M7: DB
        db_url = f"sqlite:///{tmp_dir / 'report_test.db'}"
        db = DatabaseManager(db_url=db_url)
        stats = db.ingest_pipeline_results(
            manifest_data=manifest.model_dump(),
            reid_data=reid.model_dump(),
        )

        # M10: Reporting
        gen = ReportGenerator(db_url=db_url)

        # KPI
        kpi = gen.compute_kpis()
        assert kpi.tracked_tigers > 0

        # Census
        census_path = tmp_dir / "census.csv"
        n = gen.export_ntca_census(census_path)
        assert n > 0

        # ROI
        roi = gen.compute_storage_roi(
            total_images=10, blanks_quarantined=5,
            auto_matches=stats.tigers_updated,
        )
        assert roi.total_images_processed == 10

        # Feed
        feed = gen.generate_dashboard_feed()
        assert feed.generated_at != ""
