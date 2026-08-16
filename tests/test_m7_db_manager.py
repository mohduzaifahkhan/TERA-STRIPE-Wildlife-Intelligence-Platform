"""
TERA-STRIPE M7 Database Manager -- Test Suite
================================================
Validates database ingestion, query, and export capabilities.

Test classes
------------
  TestDatabaseManagerInit   - Engine/session creation and table init
  TestStationRegistration   - Station CRUD from manifests
  TestTigerProfiles         - Tiger create/update lifecycle
  TestSightingLogging       - Sighting INSERT and retrieval
  TestPipelineIngestion     - Full batch ingestion from pipeline artifacts
  TestQueryAPIs             - Tiger profile, station stats, recent sightings
  TestExport                - CSV and JSON export
  TestFullPipeline          - End-to-end M1 through M7
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["USE_POSTGIS"] = "false"
os.environ["ENVIRONMENT"] = "testing"


# ── Helper ───────────────────────────────────────────────────────

def _make_manifest() -> dict:
    """Build a minimal manifest for testing."""
    return {
        "batch_id": "BATCH_DB_TEST",
        "source_directory": "/raw/STN_104B",
        "station": {
            "station_id": "PTR_STN_104B",
            "zone_type": "CORE",
            "range_name": "Karmajhiri",
            "geom_wkt": "POINT(79.2 22.5)",
            "elevation_m": 550.0,
            "h3_res8": "8844c0a305fffff",
            "h3_res9": "8944c0a305bffff",
        },
        "records": [
            {
                "image_id": "IMG_00001",
                "absolute_path": "/raw/STN_104B/IMG_00001.JPG",
                "station_id": "PTR_STN_104B",
                "geom_wkt": "POINT(79.2 22.5)",
                "exif": {
                    "datetime_original": "2026-02-14T08:30:00+05:30",
                    "ambient_temp_c": 18.5,
                    "flash_fired": False,
                },
            },
            {
                "image_id": "IMG_00002",
                "absolute_path": "/raw/STN_104B/IMG_00002.JPG",
                "station_id": "PTR_STN_104B",
                "geom_wkt": "POINT(79.2 22.5)",
                "exif": {
                    "datetime_original": "2026-02-14T09:15:00+05:30",
                },
            },
        ],
    }


def _make_reid_data() -> dict:
    """Build a minimal reid_result for testing."""
    return {
        "batch_id": "BATCH_DB_TEST",
        "dispatches": [
            {
                "image_id": "IMG_00001",
                "crop_path": "/crops/IMG_00001_LEFT.jpg",
                "flank_side": "LEFT",
                "status": "AUTO_MATCH",
                "assigned_tiger_id": "PTR_M_001",
                "confidence": 0.92,
                "top_k_matches": [],
            },
            {
                "image_id": "IMG_00002",
                "crop_path": "/crops/IMG_00002_RIGHT.jpg",
                "flank_side": "RIGHT",
                "status": "NEW_INDIVIDUAL",
                "assigned_tiger_id": "PTR_NEW_001",
                "confidence": 0.30,
                "top_k_matches": [],
            },
        ],
    }


# =====================================================================
#  Test 1: Database Initialization
# =====================================================================

class TestDatabaseManagerInit:
    """Validate engine/session creation."""

    def test_creates_tables(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db_url = f"sqlite:///{tmp_dir / 'test.db'}"
        db = DatabaseManager(db_url=db_url)
        summary = db.get_database_summary()
        assert summary["camera_stations"] == 0
        assert summary["tigers"] == 0

    def test_idempotent_init(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db_url = f"sqlite:///{tmp_dir / 'test.db'}"
        db1 = DatabaseManager(db_url=db_url)
        db2 = DatabaseManager(db_url=db_url)  # Should not fail
        assert db2.get_database_summary()["tigers"] == 0


# =====================================================================
#  Test 2: Station Registration
# =====================================================================

class TestStationRegistration:
    """Validate station CRUD from manifests."""

    def test_register_new_station(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        manifest = _make_manifest()

        n = db.register_stations_from_manifest(manifest)
        assert n == 1
        assert db.get_database_summary()["camera_stations"] == 1

    def test_duplicate_station_updates(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        manifest = _make_manifest()

        db.register_stations_from_manifest(manifest)
        n = db.register_stations_from_manifest(manifest)  # Duplicate
        assert n == 0  # Updated, not created
        assert db.get_database_summary()["camera_stations"] == 1

    def test_empty_manifest_station(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        assert db.register_stations_from_manifest({}) == 0


# =====================================================================
#  Test 3: Tiger Profile Management
# =====================================================================

class TestTigerProfiles:
    """Validate tiger create/update lifecycle."""

    def test_create_tiger(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        tiger = db.create_or_update_tiger(
            tiger_id="PTR_M_001",
            common_name="Collarwali",
            sex="FEMALE",
        )
        assert tiger.tiger_id == "PTR_M_001"
        assert tiger.common_name == "Collarwali"

    def test_update_tiger_timestamp(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 6, 1, tzinfo=timezone.utc)

        db.create_or_update_tiger("PTR_M_001", timestamp=t1)
        tiger = db.create_or_update_tiger("PTR_M_001", timestamp=t2)

        # SQLite strips tzinfo; compare naive values
        expected = t2.replace(tzinfo=None)
        actual = tiger.last_detected_at
        if actual and actual.tzinfo:
            actual = actual.replace(tzinfo=None)
        assert actual == expected

    def test_get_all_tigers(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        db.create_or_update_tiger("PTR_M_001")
        db.create_or_update_tiger("PTR_M_002")
        db.create_or_update_tiger("PTR_M_003")

        tigers = db.get_all_tigers()
        assert len(tigers) == 3


# =====================================================================
#  Test 4: Sighting Logging
# =====================================================================

class TestSightingLogging:
    """Validate sighting INSERT and retrieval."""

    def test_log_sighting(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        db.create_or_update_tiger("PTR_M_001")
        db.register_stations_from_manifest(_make_manifest())

        sid = db.log_sighting(
            tiger_id="PTR_M_001",
            station_id="PTR_STN_104B",
            captured_at=datetime.now(timezone.utc),
            flank_orientation="LEFT_FLANK",
            reid_confidence=0.92,
            verification_status="AUTO_COMMITTED",
            raw_image_path="/raw/IMG_00001.JPG",
            flank_crop_path="/crops/IMG_00001_LEFT.jpg",
        )
        assert sid  # UUID string returned
        assert db.get_database_summary()["sightings"] == 1

    def test_multiple_sightings(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        db.create_or_update_tiger("PTR_M_001")

        for i in range(5):
            db.log_sighting(
                tiger_id="PTR_M_001",
                station_id="PTR_STN_104B",
                captured_at=datetime.now(timezone.utc),
                flank_orientation="LEFT_FLANK",
                reid_confidence=0.85,
                verification_status="AUTO_COMMITTED",
                raw_image_path=f"/raw/IMG_{i:05d}.JPG",
                flank_crop_path=f"/crops/IMG_{i:05d}_LEFT.jpg",
            )

        assert db.get_database_summary()["sightings"] == 5


# =====================================================================
#  Test 5: Batch Pipeline Ingestion
# =====================================================================

class TestPipelineIngestion:
    """Validate full batch ingestion from pipeline artifacts."""

    def test_ingest_manifest_and_reid(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        stats = db.ingest_pipeline_results(
            manifest_data=_make_manifest(),
            reid_data=_make_reid_data(),
        )

        assert stats.batch_id == "BATCH_DB_TEST"
        assert stats.stations_created == 1
        assert stats.tigers_created >= 1  # NEW_INDIVIDUAL
        assert stats.sightings_created == 2
        assert len(stats.errors) == 0

    def test_ingestion_creates_tiger_profiles(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        db.ingest_pipeline_results(
            manifest_data=_make_manifest(),
            reid_data=_make_reid_data(),
        )

        tigers = db.get_all_tigers()
        ids = [t.tiger_id for t in tigers]
        assert "PTR_M_001" in ids
        assert "PTR_NEW_001" in ids

    def test_ingestion_with_no_data(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        stats = db.ingest_pipeline_results()
        assert stats.sightings_created == 0


# =====================================================================
#  Test 6: Query APIs
# =====================================================================

class TestQueryAPIs:
    """Validate tiger profile and station queries."""

    def test_tiger_profile_query(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        db.ingest_pipeline_results(
            manifest_data=_make_manifest(),
            reid_data=_make_reid_data(),
        )

        profile = db.get_tiger_profile("PTR_M_001")
        assert profile is not None
        assert profile.tiger_id == "PTR_M_001"
        assert profile.total_sightings >= 1

    def test_nonexistent_tiger(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        assert db.get_tiger_profile("NONEXISTENT") is None

    def test_station_stats_query(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        db.ingest_pipeline_results(
            manifest_data=_make_manifest(),
            reid_data=_make_reid_data(),
        )

        stats = db.get_station_stats("PTR_STN_104B")
        assert stats is not None
        assert stats.station_id == "PTR_STN_104B"
        assert stats.total_sightings >= 1

    def test_recent_sightings(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        db.ingest_pipeline_results(
            manifest_data=_make_manifest(),
            reid_data=_make_reid_data(),
        )

        recent = db.get_recent_sightings(limit=10)
        assert len(recent) >= 1
        assert "tiger_id" in recent[0]
        assert "station_id" in recent[0]


# =====================================================================
#  Test 7: Export
# =====================================================================

class TestExport:
    """Validate CSV and JSON export."""

    def test_export_sightings_csv(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        db.ingest_pipeline_results(
            manifest_data=_make_manifest(),
            reid_data=_make_reid_data(),
        )

        csv_path = tmp_dir / "sightings.csv"
        n = db.export_sightings_csv(csv_path)
        assert n >= 1
        assert csv_path.exists()

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 1
        assert "tiger_id" in rows[0]

    def test_export_tigers_json(self, tmp_dir: Path) -> None:
        from src.m7_db_manager import DatabaseManager

        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'test.db'}")
        db.ingest_pipeline_results(
            manifest_data=_make_manifest(),
            reid_data=_make_reid_data(),
        )

        json_path = tmp_dir / "tigers.json"
        n = db.export_tigers_json(json_path)
        assert n >= 1
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert len(data) >= 1
        assert data[0]["tiger_id"]


# =====================================================================
#  Test 8: Full Pipeline M1 -> M7
# =====================================================================

class TestFullPipeline:
    """End-to-end integration test."""

    def test_full_pipeline_populates_database(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """Full pipeline must populate the database correctly."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend as TriageMock
        from src.m2_triage import TriageEngine
        from src.m4_flank_pose import FlankExtractionEngine, MockPoseBackend
        from src.m5_reid_engine import (
            MockEmbeddingBackend, ReIDEngine, VectorGallery,
        )
        from src.m7_db_manager import DatabaseManager

        # M1
        manifest = generate_manifest(full_fixture_dir)

        # M2
        triage = TriageEngine(
            backend=TriageMock(seed=42),
            confidence_threshold=0.15,
        ).process_manifest(manifest.model_dump())

        # M4
        flank = FlankExtractionEngine(
            backend=MockPoseBackend(seed=42),
            crops_dir=tmp_dir / "crops",
        ).process_triage(
            triage.model_dump(by_alias=True),
            source_records=manifest.model_dump()["records"],
        )

        # M5
        reid = ReIDEngine(
            backend=MockEmbeddingBackend(seed=42),
            gallery=VectorGallery(),
        ).process_extractions(flank.model_dump())

        # M7: Ingest into database
        db = DatabaseManager(db_url=f"sqlite:///{tmp_dir / 'pipeline.db'}")
        stats = db.ingest_pipeline_results(
            manifest_data=manifest.model_dump(),
            reid_data=reid.model_dump(),
        )

        assert stats.sightings_created > 0
        assert stats.tigers_created > 0
        assert len(stats.errors) == 0

        # Verify database
        summary = db.get_database_summary()
        assert summary["tigers"] > 0
        assert summary["sightings"] > 0

        # Export test
        csv_path = tmp_dir / "export.csv"
        n = db.export_sightings_csv(csv_path)
        assert n == stats.sightings_created
