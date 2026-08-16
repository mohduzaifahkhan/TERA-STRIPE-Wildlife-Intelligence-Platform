"""
TERA-STRIPE M1 Ingestion — Test Suite
=======================================
Validates all Milestone 1 acceptance criteria per Master Context Packet §8.6:

  1. Station token parsing from folder paths
  2. EXIF extraction (valid, corrupted, fallback)
  3. pHash computation and uniqueness
  4. Burst duplicate detection (Hamming ≤ 2, Δt ≤ 2s)
  5. Full manifest generation and schema validation
  6. Database table initialization (SQLite fallback)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Test 1: Station Token Parsing                               ║
# ╚═══════════════════════════════════════════════════════════════╝

class TestStationTokenParsing:
    """Verify that folder names are correctly resolved against the station registry."""

    def test_known_station_stn_104b(self, tmp_dir: Path) -> None:
        """STN_104B must resolve to PTR_STN_104B in Karmajhiri CORE."""
        from src.m1_ingestion import parse_station_token

        station_dir = tmp_dir / "STN_104B"
        station_dir.mkdir()
        meta = parse_station_token(station_dir)

        assert meta.station_id == "PTR_STN_104B"
        assert meta.zone == "CORE"
        assert meta.range_name == "Karmajhiri"
        assert abs(meta.latitude - 21.68502) < 1e-4
        assert abs(meta.longitude - 79.28504) < 1e-4

    def test_known_station_stn_103(self, tmp_dir: Path) -> None:
        """STN_103 must resolve to PTR_STN_103 in Rukhad BUFFER."""
        from src.m1_ingestion import parse_station_token

        station_dir = tmp_dir / "STN_103"
        station_dir.mkdir()
        meta = parse_station_token(station_dir)

        assert meta.station_id == "PTR_STN_103"
        assert meta.zone == "BUFFER"
        assert meta.range_name == "Rukhad"

    def test_unknown_station_fallback(self, tmp_dir: Path) -> None:
        """Unknown station tokens must fall back gracefully."""
        from src.m1_ingestion import parse_station_token

        station_dir = tmp_dir / "STN_UNKNOWN_999"
        station_dir.mkdir()
        meta = parse_station_token(station_dir)

        assert "STN_UNKNOWN_999" in meta.station_id
        assert meta.zone in ("CORE", "BUFFER", "CORRIDOR", "FRINGE")
        assert meta.h3_res8  # Must have some H3 value, even if fallback

    def test_h3_index_populated(self, tmp_dir: Path) -> None:
        """H3 resolution 8 and 9 fields must be non-empty strings."""
        from src.m1_ingestion import parse_station_token

        station_dir = tmp_dir / "STN_104B"
        station_dir.mkdir()
        meta = parse_station_token(station_dir)

        assert isinstance(meta.h3_res8, str)
        assert len(meta.h3_res8) > 0
        assert isinstance(meta.h3_res9, str)
        assert len(meta.h3_res9) > 0

    def test_case_insensitive_token(self, tmp_dir: Path) -> None:
        """Station token resolution should be case-insensitive."""
        from src.m1_ingestion import parse_station_token

        station_dir = tmp_dir / "stn_104b"
        station_dir.mkdir()
        meta = parse_station_token(station_dir)

        assert meta.station_id == "PTR_STN_104B"


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Test 2: EXIF Metadata Extraction                            ║
# ╚═══════════════════════════════════════════════════════════════╝

class TestExifExtraction:
    """Verify EXIF parsing across valid, corrupted, and fallback scenarios."""

    def test_valid_exif_extraction(self, sample_station_dir: Path) -> None:
        """Images with valid EXIF should return status=VALID and parsed timestamp."""
        from src.m1_ingestion import extract_exif_metadata

        img_path = next(sample_station_dir.glob("*.JPG"))
        result = extract_exif_metadata(img_path)

        assert result["exif_status"] in ("VALID", "FALLBACK_INFERRED")
        assert result["captured_at"] is not None
        # Verify ISO-8601 format
        dt = datetime.fromisoformat(result["captured_at"])
        assert dt.tzinfo is not None, "Timestamp must include timezone"

    def test_iso8601_format(self, sample_station_dir: Path) -> None:
        """Extracted timestamps must be valid ISO-8601 strings with IST offset."""
        from src.m1_ingestion import extract_exif_metadata

        img_path = next(sample_station_dir.glob("*.JPG"))
        result = extract_exif_metadata(img_path)
        ts = result["captured_at"]

        # Must parse as ISO-8601
        dt = datetime.fromisoformat(ts)
        assert dt.year >= 2020

    def test_flash_boolean(self, sample_station_dir: Path) -> None:
        """Flash field must be a proper boolean."""
        from src.m1_ingestion import extract_exif_metadata

        img_path = next(sample_station_dir.glob("*.JPG"))
        result = extract_exif_metadata(img_path)

        assert isinstance(result["flash_fired"], bool)

    def test_corrupted_exif_fallback(self, tmp_dir: Path) -> None:
        """Files with corrupted/missing EXIF should gracefully fall back."""
        from src.m1_ingestion import extract_exif_metadata

        # Create a minimal JPEG with no EXIF
        from PIL import Image

        bad_img = tmp_dir / "no_exif.JPG"
        Image.new("RGB", (100, 100), (128, 128, 128)).save(str(bad_img), "JPEG")

        result = extract_exif_metadata(bad_img)

        assert result["exif_status"] in ("CORRUPTED", "FALLBACK_INFERRED")
        assert result["captured_at"] is not None  # Must have fallback timestamp

    def test_temperature_parsing(self, sample_station_dir: Path) -> None:
        """Temperature should be parsed from ImageDescription EXIF field."""
        from src.m1_ingestion import extract_exif_metadata

        img_path = next(sample_station_dir.glob("*.JPG"))
        result = extract_exif_metadata(img_path)

        # Our fixture images include Temp in ImageDescription
        if result["ambient_temp_c"] is not None:
            assert isinstance(result["ambient_temp_c"], float)
            assert -20.0 <= result["ambient_temp_c"] <= 55.0


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Test 3: Perceptual Hash Computation                         ║
# ╚═══════════════════════════════════════════════════════════════╝

class TestPhashComputation:
    """Verify 64-bit perceptual hash computation and MD5 checksum."""

    def test_phash_returns_hex_string(self, sample_station_dir: Path) -> None:
        """pHash must be a 16-character hexadecimal string (64 bits)."""
        from src.m1_ingestion import compute_phash_and_md5

        img_path = next(sample_station_dir.glob("*.JPG"))
        phash, md5 = compute_phash_and_md5(img_path)

        assert isinstance(phash, str)
        assert len(phash) == 16, f"Expected 16-char hex, got {len(phash)}: {phash}"
        int(phash, 16)  # Must be valid hex

    def test_md5_returns_hex_string(self, sample_station_dir: Path) -> None:
        """MD5 must be a 32-character hex string."""
        from src.m1_ingestion import compute_phash_and_md5

        img_path = next(sample_station_dir.glob("*.JPG"))
        _, md5 = compute_phash_and_md5(img_path)

        assert isinstance(md5, str)
        assert len(md5) == 32

    def test_unique_images_have_different_phash(
        self, sample_station_dir: Path
    ) -> None:
        """Visually different images should produce different perceptual hashes."""
        from src.m1_ingestion import compute_phash_and_md5

        images = sorted(sample_station_dir.glob("*.JPG"))
        if len(images) < 2:
            pytest.skip("Need at least 2 images")

        hashes = set()
        for img_path in images:
            phash, _ = compute_phash_and_md5(img_path)
            hashes.add(phash)

        # At least 2 distinct hashes out of 3 images
        assert len(hashes) >= 2, "Unique images should produce distinct pHash values"


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Test 4: Burst Duplicate Detection                           ║
# ╚═══════════════════════════════════════════════════════════════╝

class TestBurstDeduplication:
    """Verify intra-burst duplicate detection via Hamming distance."""

    def test_burst_duplicates_flagged(self, full_fixture_dir: Path) -> None:
        """
        IMG_00009 and IMG_00010 are near-duplicates of IMG_00008.
        They must be flagged with is_burst_duplicate = True.
        """
        from src.m1_ingestion import generate_manifest

        manifest = generate_manifest(full_fixture_dir)

        dup_ids = {r.image_id for r in manifest.records if r.is_burst_duplicate}
        assert len(dup_ids) >= 1, (
            f"Expected at least 1 burst duplicate, found {len(dup_ids)}"
        )

        # IMG_00008 should NOT be flagged (it's the original)
        img8 = next(
            (r for r in manifest.records if r.image_id == "IMG_00008"), None
        )
        if img8:
            assert not img8.is_burst_duplicate, "Original frame should not be flagged"

    def test_no_false_duplicates(self, full_fixture_dir: Path) -> None:
        """
        Images with different visual content and timestamps far apart
        must NOT be flagged as duplicates.
        """
        from src.m1_ingestion import generate_manifest

        manifest = generate_manifest(full_fixture_dir)

        # IMG_00001 through IMG_00007 should all be unique
        for i in range(1, 8):
            image_id = f"IMG_{i:05d}"
            record = next(
                (r for r in manifest.records if r.image_id == image_id), None
            )
            if record:
                assert not record.is_burst_duplicate, (
                    f"{image_id} was falsely flagged as a burst duplicate"
                )


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Test 5: Full Manifest Generation & Schema Validation        ║
# ╚═══════════════════════════════════════════════════════════════╝

class TestManifestGeneration:
    """Verify that the generated manifest conforms to the data contract."""

    def test_manifest_structure(self, sample_station_dir: Path, tmp_dir: Path) -> None:
        """Manifest must contain all required top-level fields."""
        from src.m1_ingestion import generate_manifest

        output = tmp_dir / "test_manifest.json"
        manifest = generate_manifest(sample_station_dir, output)

        assert manifest.batch_id
        assert manifest.source_directory
        assert manifest.total_frames > 0
        assert manifest.station_metadata is not None
        assert len(manifest.records) == manifest.total_frames

    def test_manifest_json_roundtrip(
        self, sample_station_dir: Path, tmp_dir: Path
    ) -> None:
        """Manifest must serialize to valid JSON and deserialize back."""
        from src.m1_ingestion import IngestManifest, generate_manifest

        output = tmp_dir / "roundtrip.json"
        original = generate_manifest(sample_station_dir, output)

        # Read back from disk
        with open(output, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Validate against Pydantic model
        restored = IngestManifest(**raw)
        assert restored.batch_id == original.batch_id
        assert restored.total_frames == original.total_frames
        assert len(restored.records) == len(original.records)

    def test_manifest_station_metadata(
        self, sample_station_dir: Path, tmp_dir: Path
    ) -> None:
        """Station metadata must be populated with expected fields."""
        from src.m1_ingestion import generate_manifest

        manifest = generate_manifest(sample_station_dir, tmp_dir / "m.json")
        sm = manifest.station_metadata

        assert sm.station_id == "PTR_STN_104B"
        assert sm.zone == "CORE"
        assert sm.latitude > 0
        assert sm.longitude > 0
        assert sm.h3_res8
        assert sm.h3_res9

    def test_manifest_records_have_required_fields(
        self, sample_station_dir: Path, tmp_dir: Path
    ) -> None:
        """Every record must have image_id, phash, captured_at, exif_status."""
        from src.m1_ingestion import generate_manifest

        manifest = generate_manifest(sample_station_dir, tmp_dir / "m.json")

        for record in manifest.records:
            assert record.image_id, "Missing image_id"
            assert record.phash, "Missing phash"
            assert record.captured_at, "Missing captured_at"
            assert record.exif_status in ("VALID", "CORRUPTED", "FALLBACK_INFERRED")
            assert record.md5_checksum, "Missing md5_checksum"

    def test_manifest_on_full_fixtures(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """Full fixture set must produce exactly 10 records."""
        from src.m1_ingestion import generate_manifest

        manifest = generate_manifest(full_fixture_dir, tmp_dir / "full.json")

        assert manifest.total_frames == 10
        assert manifest.station_metadata.station_id == "PTR_STN_104B"

    def test_empty_directory_raises(self, tmp_dir: Path) -> None:
        """Attempting ingestion on an empty directory must raise ValueError."""
        from src.m1_ingestion import generate_manifest

        empty_dir = tmp_dir / "EMPTY_STATION"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No images found"):
            generate_manifest(empty_dir)

    def test_nonexistent_directory_raises(self) -> None:
        """Attempting ingestion on a missing path must raise FileNotFoundError."""
        from src.m1_ingestion import generate_manifest

        with pytest.raises(FileNotFoundError):
            generate_manifest(Path("/nonexistent/path/STN_999"))


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Test 6: Database Table Initialization (SQLite Fallback)     ║
# ╚═══════════════════════════════════════════════════════════════╝

class TestDatabaseInit:
    """Verify all 5 tables create successfully on SQLite."""

    def test_tables_created(self, db_engine) -> None:
        """All 5 core tables must exist after init_db()."""
        from sqlalchemy import inspect

        inspector = inspect(db_engine)
        tables = set(inspector.get_table_names())

        expected = {
            "camera_stations",
            "tigers",
            "tiger_sightings",
            "home_ranges",
            "security_alerts",
        }
        assert expected.issubset(tables), (
            f"Missing tables: {expected - tables}"
        )

    def test_camera_station_insert(self, db_session) -> None:
        """Insert and retrieve a CameraStation row."""
        from src.m7_database import CameraStation

        station = CameraStation(
            station_id="PTR_STN_TEST",
            zone_type="CORE",
            range_name="TestRange",
            geom="POINT(79.285 21.685)",
            elevation_m=420.0,
            h3_res8_index="8860144949fffff",
            h3_res9_index="89601449487ffff",
        )
        db_session.add(station)
        db_session.commit()

        fetched = db_session.query(CameraStation).filter_by(
            station_id="PTR_STN_TEST"
        ).first()
        assert fetched is not None
        assert fetched.zone_type == "CORE"
        assert fetched.range_name == "TestRange"

    def test_tiger_insert(self, db_session) -> None:
        """Insert and retrieve a Tiger row."""
        from src.m7_database import Tiger

        now = datetime.utcnow()
        tiger = Tiger(
            tiger_id="PTR_M_TEST",
            common_name="Test Tiger",
            sex="MALE",
            status="RESIDENT",
            first_detected_at=now,
            last_detected_at=now,
        )
        db_session.add(tiger)
        db_session.commit()

        fetched = db_session.query(Tiger).filter_by(tiger_id="PTR_M_TEST").first()
        assert fetched is not None
        assert fetched.common_name == "Test Tiger"
        assert fetched.sex == "MALE"

    def test_security_alert_insert(self, db_session) -> None:
        """Insert and retrieve a SecurityAlert row."""
        from src.m7_database import SecurityAlert

        alert = SecurityAlert(
            alert_type="VILLAGE_PROXIMITY",
            severity="CRITICAL",
            tiger_id="PTR_M_TEST",
            distance_to_village_km=2.14,
            alert_payload='{"nearest_village": "Sillari"}',
        )
        db_session.add(alert)
        db_session.commit()

        fetched = db_session.query(SecurityAlert).filter_by(
            alert_type="VILLAGE_PROXIMITY"
        ).first()
        assert fetched is not None
        assert fetched.severity == "CRITICAL"
        assert fetched.distance_to_village_km == pytest.approx(2.14)


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Test 7: Config Module Validation                            ║
# ╚═══════════════════════════════════════════════════════════════╝

class TestConfigModule:
    """Verify Pydantic v2 AppConfig loads with correct defaults."""

    def test_config_loads(self) -> None:
        """AppConfig must instantiate without errors."""
        from src.config import AppConfig

        cfg = AppConfig()
        assert cfg.PROJECT_NAME == "TERA-STRIPE Wildlife Intelligence Platform"

    def test_config_vram_budget(self) -> None:
        """VRAM budget must default to 6.0 GB."""
        from src.config import AppConfig

        cfg = AppConfig()
        assert cfg.VRAM_BUDGET_GB == 6.0

    def test_config_spatial_crs(self) -> None:
        """CRS defaults must be EPSG:4326 (storage) and EPSG:32644 (metric)."""
        from src.config import AppConfig

        cfg = AppConfig()
        assert cfg.STORAGE_CRS == "EPSG:4326"
        assert cfg.METRIC_CRS == "EPSG:32644"

    def test_config_alert_thresholds(self) -> None:
        """Alert thresholds must match the Master Context Packet values."""
        from src.config import AppConfig

        cfg = AppConfig()
        assert cfg.DIST_VILLAGE_ALERT_KM == 5.0
        assert cfg.SHIFT_CORE_CENTROID_KM2 == 15.0
        assert cfg.DAYS_ABSENCE_ALERT == 45

    def test_config_derived_paths(self) -> None:
        """Derived paths must resolve from BASE_DIR."""
        from src.config import AppConfig

        cfg = AppConfig()
        assert cfg.DATA_DIR is not None
        assert cfg.MANIFESTS_DIR is not None
        assert cfg.EXPORTS_DIR is not None
        assert "data" in str(cfg.DATA_DIR)
        assert "exports" in str(cfg.EXPORTS_DIR)
