"""
TERA-STRIPE M3 Quarantine -- Test Suite
=========================================
Validates the staged quarantine lifecycle per Master Context Packet.

Test classes
------------
  TestQuarantineModels     - Pydantic ledger and ROI model validation
  TestQuarantineOperation  - File move, ledger creation, idempotency
  TestRollbackOperation    - Restore quarantined files to original paths
  TestPurgeExpired         - Retention window enforcement and deletion
  TestROITelemetry         - Storage-saved and hours-saved calculations
  TestQuarantineManager    - Batch listing, status queries
  TestCLIIntegration       - End-to-end triage -> quarantine flow
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["USE_POSTGIS"] = "false"
os.environ["ENVIRONMENT"] = "testing"

IST = timezone(timedelta(hours=5, minutes=30))


# ── Helpers ──────────────────────────────────────────────────────

def _create_dummy_images(directory: Path, names: list[str]) -> list[Path]:
    """Create minimal dummy files simulating camera-trap images."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        p = directory / name
        p.write_bytes(b"\xff\xd8\xff\xe0" + os.urandom(1024))
        paths.append(p)
    return paths


def _build_triage_data(
    batch_id: str,
    source_dir: str,
    image_files: list[Path],
    blank_ids: set[str],
    quarantine_dir: Path,
) -> dict:
    """Build a triage_result.json-compatible dict for testing."""
    dispatches = []
    for img in image_files:
        image_id = img.stem
        if image_id in blank_ids:
            dispatches.append({
                "image_id": image_id,
                "status": "QUARANTINED_BLANK",
                "max_confidence": 0.0,
                "detections": [],
                "target_quarantine_path": str(
                    quarantine_dir / batch_id / img.name
                ),
            })
        else:
            dispatches.append({
                "image_id": image_id,
                "status": "FAUNA_DETECTED",
                "max_confidence": 0.85,
                "detections": [],
            })

    source_records = [
        {"image_id": img.stem, "absolute_path": str(img)}
        for img in image_files
    ]

    return {
        "batch_id": batch_id,
        "triage_summary": {
            "processed_frames": len(image_files),
            "blank_frames": len(blank_ids),
            "fauna_frames": len(image_files) - len(blank_ids),
            "storage_saved_gb": 0.001,
            "manual_hours_saved": 0.01,
        },
        "dispatches": dispatches,
        "_source_records": source_records,
        "_source_directory": source_dir,
    }


# =====================================================================
#  Test 1: Pydantic Ledger Models
# =====================================================================

class TestQuarantineModels:
    """Validate quarantine ledger Pydantic schemas."""

    def test_quarantine_entry_schema(self) -> None:
        from src.m3_quarantine import QuarantineEntry

        entry = QuarantineEntry(
            image_id="IMG_00001",
            original_path="/data/raw/IMG_00001.JPG",
            quarantine_path="/data/quarantine/BATCH_01/IMG_00001.JPG",
            batch_id="BATCH_01",
            file_size_bytes=4096,
            quarantined_at="2026-02-14T10:00:00+05:30",
            retention_expires_at="2026-03-16T10:00:00+05:30",
        )
        assert entry.status == "QUARANTINED"
        assert entry.file_size_bytes == 4096

    def test_roi_summary_schema(self) -> None:
        from src.m3_quarantine import ROISummary

        roi = ROISummary(
            total_quarantined=80,
            storage_saved_bytes=83886080,
            storage_saved_gb=0.0781,
            manual_hours_saved=0.10,
        )
        assert roi.total_quarantined == 80
        assert roi.retention_days == 30

    def test_quarantine_ledger_schema(self) -> None:
        from src.m3_quarantine import QuarantineEntry, QuarantineLedger, ROISummary

        ledger = QuarantineLedger(
            batch_id="BATCH_TEST",
            quarantine_dir="/data/quarantine",
            created_at="2026-02-14T10:00:00+05:30",
            entries=[
                QuarantineEntry(
                    image_id="IMG_00001",
                    original_path="/data/raw/IMG_00001.JPG",
                    quarantine_path="/data/quarantine/BATCH_TEST/IMG_00001.JPG",
                    batch_id="BATCH_TEST",
                    file_size_bytes=1024,
                    quarantined_at="2026-02-14T10:00:00+05:30",
                    retention_expires_at="2026-03-16T10:00:00+05:30",
                )
            ],
            roi_summary=ROISummary(
                total_quarantined=1,
                storage_saved_bytes=1024,
                storage_saved_gb=0.000001,
                manual_hours_saved=0.00125,
            ),
        )
        assert ledger.batch_id == "BATCH_TEST"
        assert len(ledger.entries) == 1


# =====================================================================
#  Test 2: Quarantine File Operations
# =====================================================================

class TestQuarantineOperation:
    """Test file move, ledger creation, and source path resolution."""

    def test_quarantine_moves_blank_files(self, tmp_dir: Path) -> None:
        """Blank files must be moved from source to quarantine dir."""
        from src.m3_quarantine import QuarantineManager

        source_dir = tmp_dir / "raw" / "STN_104B"
        quarantine_dir = tmp_dir / "quarantine"
        images = _create_dummy_images(source_dir, [
            "IMG_00001.JPG", "IMG_00002.JPG", "IMG_00003.JPG",
        ])

        triage_data = _build_triage_data(
            batch_id="BATCH_TEST_01",
            source_dir=str(source_dir),
            image_files=images,
            blank_ids={"IMG_00001", "IMG_00003"},
            quarantine_dir=quarantine_dir,
        )

        manager = QuarantineManager(quarantine_dir)
        ledger = manager.quarantine_batch(triage_data)

        # 2 files should be quarantined
        assert len(ledger.entries) == 2
        assert ledger.roi_summary.total_quarantined == 2

        # Source files should no longer exist
        assert not (source_dir / "IMG_00001.JPG").exists()
        assert not (source_dir / "IMG_00003.JPG").exists()

        # Fauna file should still exist
        assert (source_dir / "IMG_00002.JPG").exists()

        # Quarantined files should exist in quarantine dir
        q_batch = quarantine_dir / "BATCH_TEST_01"
        assert (q_batch / "IMG_00001.JPG").exists()
        assert (q_batch / "IMG_00003.JPG").exists()

    def test_copy_mode_preserves_originals(self, tmp_dir: Path) -> None:
        """copy_mode=True must copy files instead of moving."""
        from src.m3_quarantine import QuarantineManager

        source_dir = tmp_dir / "raw" / "STN_COPY"
        quarantine_dir = tmp_dir / "quarantine"
        images = _create_dummy_images(source_dir, ["IMG_C01.JPG"])

        triage_data = _build_triage_data(
            batch_id="BATCH_COPY",
            source_dir=str(source_dir),
            image_files=images,
            blank_ids={"IMG_C01"},
            quarantine_dir=quarantine_dir,
        )

        manager = QuarantineManager(quarantine_dir)
        manager.quarantine_batch(triage_data, copy_mode=True)

        # Original should still exist
        assert (source_dir / "IMG_C01.JPG").exists()
        # Copy should exist in quarantine
        assert (quarantine_dir / "BATCH_COPY" / "IMG_C01.JPG").exists()

    def test_ledger_persisted_to_disk(self, tmp_dir: Path) -> None:
        """Quarantine must persist a ledger JSON file."""
        from src.m3_quarantine import QuarantineManager

        source_dir = tmp_dir / "raw" / "STN_LED"
        quarantine_dir = tmp_dir / "quarantine"
        images = _create_dummy_images(source_dir, ["IMG_L01.JPG"])

        triage_data = _build_triage_data(
            batch_id="BATCH_LEDGER",
            source_dir=str(source_dir),
            image_files=images,
            blank_ids={"IMG_L01"},
            quarantine_dir=quarantine_dir,
        )

        manager = QuarantineManager(quarantine_dir)
        manager.quarantine_batch(triage_data)

        ledger_path = quarantine_dir / "ledger_BATCH_LEDGER.json"
        assert ledger_path.exists()

        with open(ledger_path, "r") as f:
            data = json.load(f)
        assert data["batch_id"] == "BATCH_LEDGER"
        assert len(data["entries"]) == 1

    def test_no_blanks_produces_empty_ledger(self, tmp_dir: Path) -> None:
        """Batch with no blank frames should produce an empty ledger."""
        from src.m3_quarantine import QuarantineManager

        source_dir = tmp_dir / "raw" / "STN_NB"
        quarantine_dir = tmp_dir / "quarantine"
        images = _create_dummy_images(source_dir, ["IMG_NB01.JPG"])

        triage_data = _build_triage_data(
            batch_id="BATCH_NO_BLANK",
            source_dir=str(source_dir),
            image_files=images,
            blank_ids=set(),  # No blanks
            quarantine_dir=quarantine_dir,
        )

        manager = QuarantineManager(quarantine_dir)
        ledger = manager.quarantine_batch(triage_data)

        assert ledger.roi_summary.total_quarantined == 0
        assert len(ledger.entries) == 0


# =====================================================================
#  Test 3: Rollback Operation
# =====================================================================

class TestRollbackOperation:
    """Test restoring quarantined files to original paths."""

    def test_rollback_restores_files(self, tmp_dir: Path) -> None:
        """Rollback must move files back to original paths."""
        from src.m3_quarantine import QuarantineManager

        source_dir = tmp_dir / "raw" / "STN_RB"
        quarantine_dir = tmp_dir / "quarantine"
        images = _create_dummy_images(source_dir, [
            "IMG_R01.JPG", "IMG_R02.JPG",
        ])

        triage_data = _build_triage_data(
            batch_id="BATCH_ROLLBACK",
            source_dir=str(source_dir),
            image_files=images,
            blank_ids={"IMG_R01", "IMG_R02"},
            quarantine_dir=quarantine_dir,
        )

        manager = QuarantineManager(quarantine_dir)
        manager.quarantine_batch(triage_data)

        # Files should be in quarantine, not source
        assert not (source_dir / "IMG_R01.JPG").exists()
        assert not (source_dir / "IMG_R02.JPG").exists()

        # Rollback
        restored = manager.rollback_batch("BATCH_ROLLBACK")
        assert restored == 2

        # Files should be back in source
        assert (source_dir / "IMG_R01.JPG").exists()
        assert (source_dir / "IMG_R02.JPG").exists()

    def test_rollback_updates_ledger_status(self, tmp_dir: Path) -> None:
        """Rollback must update entry status to RESTORED in ledger."""
        from src.m3_quarantine import QuarantineManager

        source_dir = tmp_dir / "raw" / "STN_RBS"
        quarantine_dir = tmp_dir / "quarantine"
        images = _create_dummy_images(source_dir, ["IMG_RS01.JPG"])

        triage_data = _build_triage_data(
            batch_id="BATCH_RB_STATUS",
            source_dir=str(source_dir),
            image_files=images,
            blank_ids={"IMG_RS01"},
            quarantine_dir=quarantine_dir,
        )

        manager = QuarantineManager(quarantine_dir)
        manager.quarantine_batch(triage_data)
        manager.rollback_batch("BATCH_RB_STATUS")

        # Check ledger
        ledger = manager._load_ledger("BATCH_RB_STATUS")
        assert ledger is not None
        assert ledger.entries[0].status == "RESTORED"
        assert ledger.roi_summary.total_restored == 1

    def test_rollback_nonexistent_batch_returns_zero(
        self, tmp_dir: Path
    ) -> None:
        """Rollback of unknown batch must return 0."""
        from src.m3_quarantine import QuarantineManager

        manager = QuarantineManager(tmp_dir / "quarantine")
        assert manager.rollback_batch("NONEXISTENT") == 0


# =====================================================================
#  Test 4: Purge Expired
# =====================================================================

class TestPurgeExpired:
    """Test retention window enforcement."""

    def test_purge_removes_expired_files(self, tmp_dir: Path) -> None:
        """Files past retention must be deleted and status set to PURGED."""
        from src.m3_quarantine import QuarantineEntry, QuarantineLedger, QuarantineManager, ROISummary

        quarantine_dir = tmp_dir / "quarantine"
        batch_dir = quarantine_dir / "BATCH_EXP"
        batch_dir.mkdir(parents=True)

        # Create a quarantined file
        expired_file = batch_dir / "IMG_EXP.JPG"
        expired_file.write_bytes(b"\xff\xd8" + b"\x00" * 100)

        # Build a ledger where retention already expired
        past = datetime.now(IST) - timedelta(days=35)
        expired_at = past + timedelta(days=30)

        ledger = QuarantineLedger(
            batch_id="BATCH_EXP",
            quarantine_dir=str(quarantine_dir),
            created_at=past.isoformat(),
            entries=[
                QuarantineEntry(
                    image_id="IMG_EXP",
                    original_path="/original/IMG_EXP.JPG",
                    quarantine_path=str(expired_file),
                    batch_id="BATCH_EXP",
                    file_size_bytes=102,
                    quarantined_at=past.isoformat(),
                    retention_expires_at=expired_at.isoformat(),
                    status="QUARANTINED",
                )
            ],
            roi_summary=ROISummary(
                total_quarantined=1,
                storage_saved_bytes=102,
                storage_saved_gb=0.0,
                manual_hours_saved=0.00125,
            ),
        )

        manager = QuarantineManager(quarantine_dir)
        manager._save_ledger(ledger)

        # Purge
        purged = manager.purge_expired()
        assert purged == 1
        assert not expired_file.exists()

        # Verify ledger updated
        updated = manager._load_ledger("BATCH_EXP")
        assert updated.entries[0].status == "PURGED"

    def test_purge_skips_unexpired(self, tmp_dir: Path) -> None:
        """Files within retention window must NOT be purged."""
        from src.m3_quarantine import QuarantineEntry, QuarantineLedger, QuarantineManager, ROISummary

        quarantine_dir = tmp_dir / "quarantine"
        batch_dir = quarantine_dir / "BATCH_FRESH"
        batch_dir.mkdir(parents=True)

        fresh_file = batch_dir / "IMG_FRESH.JPG"
        fresh_file.write_bytes(b"\xff\xd8" + b"\x00" * 100)

        now = datetime.now(IST)
        expires = now + timedelta(days=25)  # Still 25 days left

        ledger = QuarantineLedger(
            batch_id="BATCH_FRESH",
            quarantine_dir=str(quarantine_dir),
            created_at=now.isoformat(),
            entries=[
                QuarantineEntry(
                    image_id="IMG_FRESH",
                    original_path="/original/IMG_FRESH.JPG",
                    quarantine_path=str(fresh_file),
                    batch_id="BATCH_FRESH",
                    file_size_bytes=102,
                    quarantined_at=now.isoformat(),
                    retention_expires_at=expires.isoformat(),
                    status="QUARANTINED",
                )
            ],
            roi_summary=ROISummary(
                total_quarantined=1,
                storage_saved_bytes=102,
                storage_saved_gb=0.0,
                manual_hours_saved=0.00125,
            ),
        )

        manager = QuarantineManager(quarantine_dir)
        manager._save_ledger(ledger)

        purged = manager.purge_expired()
        assert purged == 0
        assert fresh_file.exists()


# =====================================================================
#  Test 5: ROI Telemetry
# =====================================================================

class TestROITelemetry:
    """Validate storage and time savings calculations."""

    def test_storage_saved_matches_file_sizes(self, tmp_dir: Path) -> None:
        """Storage saved must equal the sum of quarantined file sizes."""
        from src.m3_quarantine import QuarantineManager

        source_dir = tmp_dir / "raw" / "STN_ROI"
        quarantine_dir = tmp_dir / "quarantine"
        images = _create_dummy_images(source_dir, [
            "IMG_ROI1.JPG", "IMG_ROI2.JPG", "IMG_ROI3.JPG",
        ])

        # Record total size before quarantine
        total_blank_size = sum(
            img.stat().st_size
            for img in images
            if img.stem in {"IMG_ROI1", "IMG_ROI3"}
        )

        triage_data = _build_triage_data(
            batch_id="BATCH_ROI",
            source_dir=str(source_dir),
            image_files=images,
            blank_ids={"IMG_ROI1", "IMG_ROI3"},
            quarantine_dir=quarantine_dir,
        )

        manager = QuarantineManager(quarantine_dir)
        ledger = manager.quarantine_batch(triage_data)

        assert ledger.roi_summary.storage_saved_bytes == total_blank_size

    def test_manual_hours_calculation(self, tmp_dir: Path) -> None:
        """Hours saved = quarantined_count * 4.5s / 3600."""
        from src.m3_quarantine import QuarantineManager

        source_dir = tmp_dir / "raw" / "STN_MH"
        quarantine_dir = tmp_dir / "quarantine"
        images = _create_dummy_images(source_dir, [
            "IMG_MH1.JPG", "IMG_MH2.JPG",
        ])

        triage_data = _build_triage_data(
            batch_id="BATCH_MH",
            source_dir=str(source_dir),
            image_files=images,
            blank_ids={"IMG_MH1", "IMG_MH2"},
            quarantine_dir=quarantine_dir,
        )

        manager = QuarantineManager(quarantine_dir)
        ledger = manager.quarantine_batch(triage_data)

        expected = round(2 * 4.5 / 3600.0, 4)
        assert ledger.roi_summary.manual_hours_saved == pytest.approx(
            expected, abs=0.001
        )


# =====================================================================
#  Test 6: Manager Status & Listing
# =====================================================================

class TestQuarantineManager:
    """Test batch listing and status queries."""

    def test_list_batches(self, tmp_dir: Path) -> None:
        """list_batches must return all ledger batch IDs."""
        from src.m3_quarantine import QuarantineManager

        quarantine_dir = tmp_dir / "quarantine"
        source_dir = tmp_dir / "raw"

        manager = QuarantineManager(quarantine_dir)

        for batch_num in range(3):
            sdir = source_dir / f"STN_L{batch_num}"
            images = _create_dummy_images(sdir, [f"IMG_L{batch_num}.JPG"])
            triage_data = _build_triage_data(
                batch_id=f"BATCH_LIST_{batch_num}",
                source_dir=str(sdir),
                image_files=images,
                blank_ids={f"IMG_L{batch_num}"},
                quarantine_dir=quarantine_dir,
            )
            manager.quarantine_batch(triage_data)

        batches = manager.list_batches()
        assert len(batches) == 3
        assert "BATCH_LIST_0" in batches
        assert "BATCH_LIST_2" in batches

    def test_get_batch_status(self, tmp_dir: Path) -> None:
        """get_batch_status must return active/restored/purged counts."""
        from src.m3_quarantine import QuarantineManager

        source_dir = tmp_dir / "raw" / "STN_ST"
        quarantine_dir = tmp_dir / "quarantine"
        images = _create_dummy_images(source_dir, ["IMG_ST1.JPG"])

        triage_data = _build_triage_data(
            batch_id="BATCH_STATUS",
            source_dir=str(source_dir),
            image_files=images,
            blank_ids={"IMG_ST1"},
            quarantine_dir=quarantine_dir,
        )

        manager = QuarantineManager(quarantine_dir)
        manager.quarantine_batch(triage_data)

        status = manager.get_batch_status("BATCH_STATUS")
        assert status is not None
        assert status["total_entries"] == 1
        assert status["active_quarantined"] == 1
        assert status["restored"] == 0

    def test_status_nonexistent_returns_none(self, tmp_dir: Path) -> None:
        from src.m3_quarantine import QuarantineManager

        manager = QuarantineManager(tmp_dir / "quarantine")
        assert manager.get_batch_status("NONEXISTENT") is None


# =====================================================================
#  Test 7: End-to-End Triage -> Quarantine Flow
# =====================================================================

class TestCLIIntegration:
    """Test the full M2 -> M3 pipeline integration."""

    def test_triage_to_quarantine_pipeline(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """Full pipeline: M1 manifest -> M2 triage -> M3 quarantine."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend, TriageEngine
        from src.m3_quarantine import QuarantineManager

        # M1: Generate manifest
        manifest = generate_manifest(full_fixture_dir)
        manifest_data = manifest.model_dump()

        # M2: Run triage
        backend = HeuristicMockBackend(seed=42, detection_rate=0.4)
        triage_engine = TriageEngine(
            backend=backend,
            confidence_threshold=0.15,
            quarantine_dir=tmp_dir / "quarantine",
        )
        triage_result = triage_engine.process_manifest(manifest_data)

        # Enrich triage data with source info for M3
        triage_data = triage_result.model_dump(by_alias=True)
        triage_data["_source_records"] = manifest_data["records"]
        triage_data["_source_directory"] = manifest_data["source_directory"]

        # M3: Quarantine blank frames (copy mode to preserve fixtures)
        manager = QuarantineManager(tmp_dir / "quarantine")
        ledger = manager.quarantine_batch(triage_data, copy_mode=True)

        # Verify
        blank_count = triage_result.triage_summary.blank_frames
        assert ledger.roi_summary.total_quarantined == blank_count
        assert ledger.batch_id == triage_result.batch_id

        # Verify ledger file exists
        ledger_path = manager._ledger_path(ledger.batch_id)
        assert ledger_path.exists()
