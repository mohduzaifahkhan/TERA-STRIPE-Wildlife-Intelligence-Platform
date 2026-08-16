"""
TERA-STRIPE M2 Triage -- Test Suite
=====================================
Validates the MegaDetector blank triage engine per Master Context Packet.

Test classes
------------
  TestTriageModels          - Pydantic contract model validation
  TestHeuristicMockBackend  - Mock detector determinism and API contract
  TestTriageClassification  - FAUNA / BLANK / HUMAN_VEHICLE decision logic
  TestTriageEngine          - End-to-end pipeline on fixture images
  TestTriageROITelemetry    - Storage saved / hours saved calculations
  TestTriageCLIOutput       - JSON output schema and roundtrip validation
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["USE_POSTGIS"] = "false"
os.environ["ENVIRONMENT"] = "testing"


# =====================================================================
#  Test 1: Pydantic Contract Models
# =====================================================================

class TestTriageModels:
    """Validate triage_result.json Pydantic schemas."""

    def test_raw_detection_schema(self) -> None:
        """RawDetection must accept 'class' alias and bbox with 4 elements."""
        from src.m2_triage import RawDetection

        det = RawDetection(
            **{"class": "animal"},
            confidence=0.942,
            bbox_normalized=[0.245, 0.312, 0.782, 0.891],
        )
        assert det.class_name == "animal"
        assert det.confidence == 0.942
        assert len(det.bbox_normalized) == 4

    def test_raw_detection_serialisation_alias(self) -> None:
        """Serialised JSON must use 'class' not 'class_name'."""
        from src.m2_triage import RawDetection

        det = RawDetection(
            **{"class": "person"},
            confidence=0.75,
            bbox_normalized=[0.1, 0.2, 0.8, 0.9],
        )
        dumped = det.model_dump(by_alias=True)
        assert "class" in dumped
        assert dumped["class"] == "person"

    def test_triage_dispatch_schema(self) -> None:
        """TriageDispatch must hold status, max_confidence, and optional quarantine path."""
        from src.m2_triage import TriageDispatch

        dispatch = TriageDispatch(
            image_id="IMG_00001",
            status="QUARANTINED_BLANK",
            max_confidence=0.0,
            detections=[],
            target_quarantine_path="/data/quarantine/BATCH_001/IMG_00001.JPG",
        )
        assert dispatch.status == "QUARANTINED_BLANK"
        assert dispatch.target_quarantine_path is not None

    def test_triage_summary_schema(self) -> None:
        """TriageSummary must have all required telemetry fields."""
        from src.m2_triage import TriageSummary

        summary = TriageSummary(
            processed_frames=100,
            blank_frames=80,
            fauna_frames=18,
            human_vehicle_frames=2,
            storage_saved_gb=0.32,
            manual_hours_saved=0.1,
        )
        assert summary.processed_frames == 100
        assert summary.blank_frames == 80

    def test_triage_result_schema(self) -> None:
        """TriageResult must compose summary + dispatches."""
        from src.m2_triage import TriageDispatch, TriageResult, TriageSummary

        result = TriageResult(
            batch_id="BATCH_TEST",
            triage_summary=TriageSummary(
                processed_frames=1,
                blank_frames=1,
                fauna_frames=0,
                storage_saved_gb=0.001,
                manual_hours_saved=0.001,
            ),
            dispatches=[
                TriageDispatch(
                    image_id="IMG_00001",
                    status="QUARANTINED_BLANK",
                    max_confidence=0.0,
                )
            ],
        )
        assert result.batch_id == "BATCH_TEST"
        assert len(result.dispatches) == 1


# =====================================================================
#  Test 2: Heuristic Mock Backend
# =====================================================================

class TestHeuristicMockBackend:
    """Validate the deterministic mock detector backend."""

    def test_mock_loads_and_unloads(self) -> None:
        """Mock backend must support load/unload lifecycle."""
        from src.m2_triage import HeuristicMockBackend

        mock = HeuristicMockBackend()
        mock.load()
        assert mock._loaded is True
        mock.unload()
        assert mock._loaded is False

    def test_mock_requires_load_before_predict(self) -> None:
        """Calling predict_batch before load must raise RuntimeError."""
        from src.m2_triage import HeuristicMockBackend

        mock = HeuristicMockBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            mock.predict_batch([Path("dummy.jpg")])

    def test_mock_deterministic_results(self, sample_station_dir: Path) -> None:
        """Same inputs with same seed must produce identical results."""
        from src.m2_triage import HeuristicMockBackend

        images = sorted(sample_station_dir.glob("*.JPG"))
        if not images:
            pytest.skip("No fixture images")

        mock1 = HeuristicMockBackend(seed=42)
        mock1.load()
        result1 = mock1.predict_batch(images)
        mock1.unload()

        mock2 = HeuristicMockBackend(seed=42)
        mock2.load()
        result2 = mock2.predict_batch(images)
        mock2.unload()

        assert len(result1) == len(result2)
        for r1, r2 in zip(result1, result2):
            assert len(r1) == len(r2)
            for d1, d2 in zip(r1, r2):
                assert d1.class_name == d2.class_name
                assert d1.confidence == d2.confidence

    def test_mock_returns_valid_detections(
        self, sample_station_dir: Path
    ) -> None:
        """Every detection must have valid class_name and normalised bbox."""
        from src.m2_triage import HeuristicMockBackend

        images = sorted(sample_station_dir.glob("*.JPG"))
        if not images:
            pytest.skip("No fixture images")

        mock = HeuristicMockBackend(seed=42, detection_rate=0.9)
        mock.load()
        results = mock.predict_batch(images)
        mock.unload()

        for frame_dets in results:
            for det in frame_dets:
                assert det.class_name in ("animal", "person", "vehicle")
                assert 0.0 <= det.confidence <= 1.0
                assert len(det.bbox_normalized) == 4
                assert all(0.0 <= v <= 1.0 for v in det.bbox_normalized)

    def test_mock_backend_name(self) -> None:
        """Mock backend must identify itself correctly."""
        from src.m2_triage import HeuristicMockBackend

        mock = HeuristicMockBackend()
        assert "Mock" in mock.backend_name


# =====================================================================
#  Test 3: Triage Classification Logic
# =====================================================================

class TestTriageClassification:
    """Test the FAUNA / BLANK / HUMAN_VEHICLE classification rules."""

    def test_no_detections_is_blank(self) -> None:
        """Zero detections must classify as QUARANTINED_BLANK."""
        from src.m2_triage import TriageEngine, HeuristicMockBackend

        engine = TriageEngine(
            backend=HeuristicMockBackend(),
            confidence_threshold=0.15,
        )
        result = engine._classify_detections([])
        assert result == "QUARANTINED_BLANK"

    def test_animal_above_threshold_is_fauna(self) -> None:
        """Animal detection above threshold must classify as FAUNA_DETECTED."""
        from src.m2_triage import RawDetection, TriageEngine, HeuristicMockBackend

        det = RawDetection(
            **{"class": "animal"},
            confidence=0.85,
            bbox_normalized=[0.1, 0.2, 0.8, 0.9],
        )
        engine = TriageEngine(
            backend=HeuristicMockBackend(),
            confidence_threshold=0.15,
        )
        result = engine._classify_detections([det])
        assert result == "FAUNA_DETECTED"

    def test_animal_below_threshold_is_blank(self) -> None:
        """Animal detection below threshold must classify as QUARANTINED_BLANK."""
        from src.m2_triage import RawDetection, TriageEngine, HeuristicMockBackend

        det = RawDetection(
            **{"class": "animal"},
            confidence=0.10,
            bbox_normalized=[0.1, 0.2, 0.8, 0.9],
        )
        engine = TriageEngine(
            backend=HeuristicMockBackend(),
            confidence_threshold=0.15,
        )
        result = engine._classify_detections([det])
        assert result == "QUARANTINED_BLANK"

    def test_person_detection_is_human_vehicle_flag(self) -> None:
        """Person detection must always flag as HUMAN_VEHICLE_FLAG."""
        from src.m2_triage import RawDetection, TriageEngine, HeuristicMockBackend

        det = RawDetection(
            **{"class": "person"},
            confidence=0.60,
            bbox_normalized=[0.2, 0.3, 0.7, 0.8],
        )
        engine = TriageEngine(
            backend=HeuristicMockBackend(),
            confidence_threshold=0.15,
        )
        result = engine._classify_detections([det])
        assert result == "HUMAN_VEHICLE_FLAG"

    def test_vehicle_detection_is_human_vehicle_flag(self) -> None:
        """Vehicle detection must always flag as HUMAN_VEHICLE_FLAG."""
        from src.m2_triage import RawDetection, TriageEngine, HeuristicMockBackend

        det = RawDetection(
            **{"class": "vehicle"},
            confidence=0.45,
            bbox_normalized=[0.1, 0.1, 0.9, 0.6],
        )
        engine = TriageEngine(
            backend=HeuristicMockBackend(),
            confidence_threshold=0.15,
        )
        result = engine._classify_detections([det])
        assert result == "HUMAN_VEHICLE_FLAG"

    def test_person_takes_priority_over_animal(self) -> None:
        """Mixed person+animal detections must classify as HUMAN_VEHICLE_FLAG."""
        from src.m2_triage import RawDetection, TriageEngine, HeuristicMockBackend

        dets = [
            RawDetection(
                **{"class": "animal"}, confidence=0.95,
                bbox_normalized=[0.1, 0.2, 0.5, 0.8],
            ),
            RawDetection(
                **{"class": "person"}, confidence=0.40,
                bbox_normalized=[0.5, 0.3, 0.9, 0.9],
            ),
        ]
        engine = TriageEngine(
            backend=HeuristicMockBackend(),
            confidence_threshold=0.15,
        )
        result = engine._classify_detections(dets)
        assert result == "HUMAN_VEHICLE_FLAG"


# =====================================================================
#  Test 4: End-to-End Triage Pipeline
# =====================================================================

class TestTriageEngine:
    """Validate the full triage pipeline on fixture images."""

    def test_process_manifest_returns_triage_result(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """Pipeline must return a valid TriageResult for real fixture images."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import (
            HeuristicMockBackend,
            TriageEngine,
            TriageResult,
        )

        # Generate manifest from M1
        manifest = generate_manifest(full_fixture_dir)
        manifest_data = manifest.model_dump()

        # Run triage with mock backend
        backend = HeuristicMockBackend(seed=42)
        engine = TriageEngine(
            backend=backend,
            confidence_threshold=0.15,
            quarantine_dir=tmp_dir / "quarantine",
        )
        result = engine.process_manifest(manifest_data)

        assert isinstance(result, TriageResult)
        assert result.batch_id == manifest.batch_id
        assert result.triage_summary.processed_frames == 10
        assert len(result.dispatches) == 10

    def test_all_dispatches_have_valid_status(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """Every dispatch must have a valid triage status."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend, TriageEngine

        manifest = generate_manifest(full_fixture_dir)
        backend = HeuristicMockBackend(seed=42)
        engine = TriageEngine(backend=backend, confidence_threshold=0.15)
        result = engine.process_manifest(manifest.model_dump())

        valid_statuses = {"FAUNA_DETECTED", "QUARANTINED_BLANK", "HUMAN_VEHICLE_FLAG"}
        for dispatch in result.dispatches:
            assert dispatch.status in valid_statuses
            assert isinstance(dispatch.max_confidence, float)
            assert dispatch.max_confidence >= 0.0

    def test_blank_frames_have_quarantine_path(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """Blank frames must have a target_quarantine_path when quarantine_dir is set."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend, TriageEngine

        manifest = generate_manifest(full_fixture_dir)
        engine = TriageEngine(
            backend=HeuristicMockBackend(seed=42, detection_rate=0.3),
            confidence_threshold=0.15,
            quarantine_dir=tmp_dir / "quarantine",
        )
        result = engine.process_manifest(manifest.model_dump())

        blanks = [d for d in result.dispatches if d.status == "QUARANTINED_BLANK"]
        if blanks:
            for b in blanks:
                assert b.target_quarantine_path is not None
                assert "quarantine" in b.target_quarantine_path.lower()

    def test_summary_counts_are_consistent(
        self, full_fixture_dir: Path
    ) -> None:
        """Summary counts must sum to total processed frames."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend, TriageEngine

        manifest = generate_manifest(full_fixture_dir)
        engine = TriageEngine(
            backend=HeuristicMockBackend(seed=42),
            confidence_threshold=0.15,
        )
        result = engine.process_manifest(manifest.model_dump())
        s = result.triage_summary

        assert s.processed_frames == s.blank_frames + s.fauna_frames + s.human_vehicle_frames

    def test_vram_discipline_unload(self) -> None:
        """Backend must be unloaded after process_manifest completes."""
        from src.m2_triage import HeuristicMockBackend, TriageEngine

        backend = HeuristicMockBackend(seed=42)
        engine = TriageEngine(backend=backend, confidence_threshold=0.15)

        # Create minimal manifest
        manifest_data = {
            "batch_id": "TEST_BATCH",
            "records": [],
        }
        engine.process_manifest(manifest_data)

        # Backend should be unloaded
        assert backend._loaded is False


# =====================================================================
#  Test 5: ROI Telemetry Calculations
# =====================================================================

class TestTriageROITelemetry:
    """Validate storage-saved and manual-hours-saved computations."""

    def test_manual_hours_calculation(
        self, full_fixture_dir: Path
    ) -> None:
        """Manual hours saved must equal blank_count * 4.5s / 3600."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend, TriageEngine

        manifest = generate_manifest(full_fixture_dir)
        engine = TriageEngine(
            backend=HeuristicMockBackend(seed=42),
            confidence_threshold=0.15,
        )
        result = engine.process_manifest(manifest.model_dump())
        s = result.triage_summary

        expected_hours = round(s.blank_frames * 4.5 / 3600.0, 4)
        assert s.manual_hours_saved == pytest.approx(expected_hours, abs=0.001)

    def test_storage_saved_nonnegative(
        self, full_fixture_dir: Path
    ) -> None:
        """Storage saved must be >= 0."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend, TriageEngine

        manifest = generate_manifest(full_fixture_dir)
        engine = TriageEngine(
            backend=HeuristicMockBackend(seed=42),
            confidence_threshold=0.15,
        )
        result = engine.process_manifest(manifest.model_dump())
        assert result.triage_summary.storage_saved_gb >= 0.0


# =====================================================================
#  Test 6: JSON Output Roundtrip
# =====================================================================

class TestTriageCLIOutput:
    """Validate triage result JSON serialisation and schema conformance."""

    def test_json_roundtrip(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """TriageResult must serialise to JSON and deserialise back."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import (
            HeuristicMockBackend,
            TriageEngine,
            TriageResult,
        )

        manifest = generate_manifest(full_fixture_dir)
        engine = TriageEngine(
            backend=HeuristicMockBackend(seed=42),
            confidence_threshold=0.15,
        )
        result = engine.process_manifest(manifest.model_dump())

        # Serialise
        output_path = tmp_dir / "triage_test.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(by_alias=True), f, indent=2)

        # Deserialise
        with open(output_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        assert raw["batch_id"] == result.batch_id
        assert raw["triage_summary"]["processed_frames"] == 10
        assert len(raw["dispatches"]) == 10

        # Each dispatch must use 'class' alias
        for dispatch in raw["dispatches"]:
            for det in dispatch.get("detections", []):
                assert "class" in det, "Detection must use 'class' alias"

    def test_backend_factory_mock_fallback(self) -> None:
        """create_backend with no weights must return mock backend."""
        from src.m2_triage import HeuristicMockBackend, create_backend

        backend = create_backend(
            weights_path=Path("/nonexistent/model.pt"),
            force_mock=False,
        )
        assert isinstance(backend, HeuristicMockBackend)

    def test_backend_factory_force_mock(self) -> None:
        """create_backend with force_mock=True must always return mock."""
        from src.m2_triage import HeuristicMockBackend, create_backend

        backend = create_backend(
            weights_path=None,
            force_mock=True,
        )
        assert isinstance(backend, HeuristicMockBackend)
