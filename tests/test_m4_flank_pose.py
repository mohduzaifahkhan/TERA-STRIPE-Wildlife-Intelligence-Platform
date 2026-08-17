"""
TERA-STRIPE M4 Flank Extraction -- Test Suite
================================================
Validates YOLO11-Pose keypoint detection, affine warp, quality scoring,
and crop generation per Master Context Packet.

Test classes
------------
  TestFlankModels           - Pydantic contract model validation
  TestKeypoints             - Keypoint generation and structure
  TestFlankSide             - LEFT/RIGHT/AMBIGUOUS determination
  TestQualityScoring        - Composite quality score calculation
  TestAffineWarp            - Crop generation and affine transform
  TestMockPoseBackend       - Mock backend determinism and API
  TestFlankExtractionEngine - End-to-end pipeline on fixtures
"""

from __future__ import annotations

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
#  Test 1: Pydantic Contract Models
# =====================================================================

class TestFlankModels:
    """Validate flank_extraction.json Pydantic schemas."""

    def test_keypoint_schema(self) -> None:
        from src.m4_flank_pose import Keypoint

        kp = Keypoint(name="left_shoulder", x=0.35, y=0.42, confidence=0.92)
        assert kp.name == "left_shoulder"
        assert 0 <= kp.x <= 1
        assert 0 <= kp.confidence <= 1

    def test_flank_extraction_schema(self) -> None:
        from src.m4_flank_pose import FlankExtraction, Keypoint

        ext = FlankExtraction(
            image_id="IMG_00001",
            source_path="/raw/IMG_00001.JPG",
            crop_path="/crops/IMG_00001_LEFT.jpg",
            flank_side="LEFT",
            keypoints=[
                Keypoint(name="left_shoulder", x=0.3, y=0.4, confidence=0.9),
            ],
            quality_score=0.82,
            bbox_normalized=[0.1, 0.2, 0.8, 0.9],
            affine_warp_applied=True,
        )
        assert ext.flank_side == "LEFT"
        assert ext.crop_size_px == 224
        assert ext.affine_warp_applied is True

    def test_extraction_result_schema(self) -> None:
        from src.m4_flank_pose import (
            FlankExtraction,
            FlankExtractionResult,
            FlankExtractionSummary,
            Keypoint,
        )

        result = FlankExtractionResult(
            batch_id="BATCH_TEST",
            crops_directory="/crops/BATCH_TEST",
            summary=FlankExtractionSummary(
                total_fauna_frames=10,
                total_extractions=8,
                high_quality_count=5,
                left_flanks=4,
                right_flanks=3,
                ambiguous_flanks=1,
                mean_quality_score=0.78,
            ),
            extractions=[],
        )
        assert result.batch_id == "BATCH_TEST"
        assert result.summary.total_extractions == 8

    def test_flank_side_enum_values(self) -> None:
        from src.m4_flank_pose import FlankExtraction, Keypoint

        for side in ("LEFT", "RIGHT", "AMBIGUOUS"):
            ext = FlankExtraction(
                image_id="test",
                source_path="/test.jpg",
                crop_path="/crop.jpg",
                flank_side=side,
                keypoints=[],
                quality_score=0.5,
                bbox_normalized=[0.1, 0.1, 0.9, 0.9],
            )
            assert ext.flank_side == side


# =====================================================================
#  Test 2: Keypoint Structure
# =====================================================================

class TestKeypoints:
    """Validate keypoint naming and structure."""

    def test_tiger_keypoint_names_count(self) -> None:
        """Must have 18 keypoint names for tiger pose."""
        from src.m4_flank_pose import TIGER_KEYPOINT_NAMES

        assert len(TIGER_KEYPOINT_NAMES) == 18

    def test_required_keypoints_present(self) -> None:
        """Core structural keypoints must be in the list."""
        from src.m4_flank_pose import TIGER_KEYPOINT_NAMES

        required = {
            "nose", "left_shoulder", "right_shoulder",
            "left_hip", "right_hip", "tail_base",
        }
        assert required.issubset(set(TIGER_KEYPOINT_NAMES))

    def test_bilateral_symmetry(self) -> None:
        """Every 'left_X' must have a matching 'right_X'."""
        from src.m4_flank_pose import TIGER_KEYPOINT_NAMES

        lefts = {n for n in TIGER_KEYPOINT_NAMES if n.startswith("left_")}
        rights = {n for n in TIGER_KEYPOINT_NAMES if n.startswith("right_")}
        left_base = {n.replace("left_", "") for n in lefts}
        right_base = {n.replace("right_", "") for n in rights}
        assert left_base == right_base


# =====================================================================
#  Test 3: Flank Side Determination
# =====================================================================

class TestFlankSide:
    """Validate LEFT/RIGHT/AMBIGUOUS determination logic."""

    def test_left_visible_returns_right_flank(self) -> None:
        """High left keypoint confidence -> RIGHT flank visible."""
        from src.m4_flank_pose import Keypoint, determine_flank_side

        kps = [
            Keypoint(name="left_shoulder", x=0.3, y=0.4, confidence=0.95),
            Keypoint(name="left_hip", x=0.35, y=0.7, confidence=0.90),
            Keypoint(name="left_knee", x=0.3, y=0.8, confidence=0.85),
            Keypoint(name="right_shoulder", x=0.7, y=0.4, confidence=0.20),
            Keypoint(name="right_hip", x=0.65, y=0.7, confidence=0.15),
            Keypoint(name="right_knee", x=0.7, y=0.8, confidence=0.10),
        ]
        assert determine_flank_side(kps) == "RIGHT"

    def test_right_visible_returns_left_flank(self) -> None:
        """High right keypoint confidence -> LEFT flank visible."""
        from src.m4_flank_pose import Keypoint, determine_flank_side

        kps = [
            Keypoint(name="left_shoulder", x=0.3, y=0.4, confidence=0.10),
            Keypoint(name="left_hip", x=0.35, y=0.7, confidence=0.15),
            Keypoint(name="left_knee", x=0.3, y=0.8, confidence=0.12),
            Keypoint(name="right_shoulder", x=0.7, y=0.4, confidence=0.92),
            Keypoint(name="right_hip", x=0.65, y=0.7, confidence=0.88),
            Keypoint(name="right_knee", x=0.7, y=0.8, confidence=0.85),
        ]
        assert determine_flank_side(kps) == "LEFT"

    def test_equal_visibility_returns_ambiguous(self) -> None:
        """Equal left/right confidence -> AMBIGUOUS."""
        from src.m4_flank_pose import Keypoint, determine_flank_side

        kps = [
            Keypoint(name="left_shoulder", x=0.3, y=0.4, confidence=0.80),
            Keypoint(name="left_hip", x=0.35, y=0.7, confidence=0.80),
            Keypoint(name="right_shoulder", x=0.7, y=0.4, confidence=0.80),
            Keypoint(name="right_hip", x=0.65, y=0.7, confidence=0.80),
        ]
        assert determine_flank_side(kps) == "AMBIGUOUS"


# =====================================================================
#  Test 4: Quality Scoring
# =====================================================================

class TestQualityScoring:
    """Validate quality score computation."""

    def test_quality_score_range(self, sample_station_dir: Path) -> None:
        """Quality score must be between 0 and 1."""
        from PIL import Image
        from src.m4_flank_pose import Keypoint, compute_quality_score

        img = Image.open(next(sample_station_dir.glob("*.JPG"))).convert("RGB")
        kps = [
            Keypoint(name="left_shoulder", x=0.3, y=0.4, confidence=0.9),
            Keypoint(name="right_hip", x=0.7, y=0.7, confidence=0.8),
        ]
        score = compute_quality_score(kps, img, "LEFT")
        assert 0.0 <= score <= 1.0

    def test_high_confidence_improves_score(self) -> None:
        """Higher keypoint confidence should yield a higher quality score."""
        from PIL import Image
        from src.m4_flank_pose import Keypoint, compute_quality_score

        img = Image.new("RGB", (224, 224), (128, 128, 128))

        kps_low = [Keypoint(name="left_shoulder", x=0.3, y=0.4, confidence=0.2)]
        kps_high = [Keypoint(name="left_shoulder", x=0.3, y=0.4, confidence=0.95)]

        score_low = compute_quality_score(kps_low, img, "LEFT")
        score_high = compute_quality_score(kps_high, img, "LEFT")

        assert score_high > score_low

    def test_sharpness_calculation(self) -> None:
        """Sharper images should have higher sharpness scores."""
        from PIL import Image, ImageDraw
        from src.m4_flank_pose import compute_sharpness

        # Sharp image with edges
        sharp = Image.new("RGB", (224, 224), (128, 128, 128))
        draw = ImageDraw.Draw(sharp)
        for i in range(0, 224, 10):
            draw.line([(i, 0), (i, 223)], fill=(255, 255, 255), width=1)

        # Blurry image (uniform)
        blurry = Image.new("RGB", (224, 224), (128, 128, 128))

        s_sharp = compute_sharpness(sharp)
        s_blurry = compute_sharpness(blurry)

        assert s_sharp > s_blurry

    def test_laterality_affects_score(self) -> None:
        """LEFT/RIGHT should score higher than AMBIGUOUS."""
        from PIL import Image
        from src.m4_flank_pose import Keypoint, compute_quality_score

        img = Image.new("RGB", (224, 224), (128, 128, 128))
        kps = [Keypoint(name="nose", x=0.5, y=0.5, confidence=0.8)]

        score_lateral = compute_quality_score(kps, img, "LEFT")
        score_ambig = compute_quality_score(kps, img, "AMBIGUOUS")

        assert score_lateral > score_ambig


# =====================================================================
#  Test 5: Affine Warp & Crop
# =====================================================================

class TestAffineWarp:
    """Validate crop generation and affine transform."""

    def test_simple_crop_produces_correct_size(
        self, sample_station_dir: Path
    ) -> None:
        """Crop output must be 224x224."""
        from PIL import Image
        from src.m4_flank_pose import Keypoint, _simple_crop

        img = Image.open(next(sample_station_dir.glob("*.JPG"))).convert("RGB")
        kps = [Keypoint(name="nose", x=0.5, y=0.5, confidence=0.9)]
        crop = _simple_crop(img, kps, 224)

        assert crop.size == (224, 224)

    def test_affine_warp_produces_correct_size(
        self, sample_station_dir: Path
    ) -> None:
        """Affine warp output must be 224x224."""
        from PIL import Image
        from src.m4_flank_pose import Keypoint, affine_warp_flank

        img = Image.open(next(sample_station_dir.glob("*.JPG"))).convert("RGB")
        kps = [
            Keypoint(name="left_shoulder", x=0.3, y=0.3, confidence=0.9),
            Keypoint(name="left_hip", x=0.35, y=0.7, confidence=0.85),
        ]
        crop, warp_applied = affine_warp_flank(img, kps, 224)

        assert crop.size == (224, 224)
        assert warp_applied is True

    def test_no_keypoints_uses_simple_crop(self) -> None:
        """No valid keypoints should fallback to simple crop."""
        from PIL import Image
        from src.m4_flank_pose import Keypoint, affine_warp_flank

        img = Image.new("RGB", (640, 480), (100, 100, 100))
        kps = [
            Keypoint(name="nose", x=0.5, y=0.5, confidence=0.1),
        ]
        crop, warp_applied = affine_warp_flank(img, kps, 224)

        assert crop.size == (224, 224)
        assert warp_applied is False


# =====================================================================
#  Test 6: Mock Pose Backend
# =====================================================================

class TestMockPoseBackend:
    """Validate mock pose backend."""

    def test_mock_lifecycle(self) -> None:
        from src.m4_flank_pose import MockPoseBackend

        mock = MockPoseBackend()
        mock.load()
        assert mock._loaded
        mock.unload()
        assert not mock._loaded

    def test_mock_requires_load(self) -> None:
        from src.m4_flank_pose import MockPoseBackend

        mock = MockPoseBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            mock.predict_batch([Path("dummy.jpg")])

    def test_mock_deterministic(self, sample_station_dir: Path) -> None:
        from src.m4_flank_pose import MockPoseBackend

        images = sorted(sample_station_dir.glob("*.JPG"))
        if not images:
            pytest.skip("No fixtures")

        m1 = MockPoseBackend(seed=42)
        m1.load()
        r1 = m1.predict_batch(images)
        m1.unload()

        m2 = MockPoseBackend(seed=42)
        m2.load()
        r2 = m2.predict_batch(images)
        m2.unload()

        for d1, d2 in zip(r1, r2):
            assert len(d1) == len(d2)
            for det1, det2 in zip(d1, d2):
                assert det1["bbox"] == det2["bbox"]
                assert len(det1["keypoints"]) == len(det2["keypoints"])

    def test_mock_returns_18_keypoints(self, sample_station_dir: Path) -> None:
        """Mock must return exactly 18 keypoints per detection."""
        from src.m4_flank_pose import MockPoseBackend, TIGER_KEYPOINT_NAMES

        images = sorted(sample_station_dir.glob("*.JPG"))
        if not images:
            pytest.skip("No fixtures")

        mock = MockPoseBackend(seed=42)
        mock.load()
        results = mock.predict_batch(images[:1])
        mock.unload()

        assert len(results) == 1
        assert len(results[0]) == 1  # One animal per image
        kps = results[0][0]["keypoints"]
        assert len(kps) == len(TIGER_KEYPOINT_NAMES)

    def test_mock_valid_keypoint_coordinates(
        self, sample_station_dir: Path
    ) -> None:
        """All keypoint coords must be normalised 0..1."""
        from src.m4_flank_pose import MockPoseBackend

        images = sorted(sample_station_dir.glob("*.JPG"))
        if not images:
            pytest.skip("No fixtures")

        mock = MockPoseBackend(seed=42)
        mock.load()
        results = mock.predict_batch(images[:1])
        mock.unload()

        for kp in results[0][0]["keypoints"]:
            assert 0 <= kp["x"] <= 1, f"{kp['name']} x={kp['x']}"
            assert 0 <= kp["y"] <= 1, f"{kp['name']} y={kp['y']}"
            assert 0 <= kp["confidence"] <= 1


# =====================================================================
#  Test 7: End-to-End Flank Extraction Pipeline
# =====================================================================

class TestFlankExtractionEngine:
    """Full pipeline test: M1 -> M2 -> M4."""

    def test_pipeline_produces_crops(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """Pipeline must produce crop files and valid extraction result."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend, TriageEngine
        from src.m4_flank_pose import FlankExtractionEngine, MockPoseBackend

        # M1
        manifest = generate_manifest(full_fixture_dir)

        # M2
        triage_engine = TriageEngine(
            backend=HeuristicMockBackend(seed=42),
            confidence_threshold=0.15,
        )
        triage_result = triage_engine.process_manifest(manifest.model_dump())
        triage_data = triage_result.model_dump(by_alias=True)

        # M4
        crops_dir = tmp_dir / "crops"
        pose_backend = MockPoseBackend(seed=42)
        flank_engine = FlankExtractionEngine(
            backend=pose_backend,
            crops_dir=crops_dir,
            crop_size=224,
        )
        result = flank_engine.process_triage(
            triage_data,
            source_records=manifest.model_dump()["records"],
        )

        # Verify
        assert result.batch_id == manifest.batch_id
        assert result.summary.total_fauna_frames == triage_result.triage_summary.fauna_frames
        assert result.summary.total_extractions > 0

        # Verify crops exist on disk
        for ext in result.extractions:
            crop_path = Path(ext.crop_path)
            assert crop_path.exists(), f"Crop missing: {crop_path}"

            from PIL import Image
            crop = Image.open(crop_path)
            assert crop.size == (224, 224)

    def test_extraction_has_valid_fields(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """Each extraction must have all required fields populated."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend, TriageEngine
        from src.m4_flank_pose import FlankExtractionEngine, MockPoseBackend

        manifest = generate_manifest(full_fixture_dir)
        triage = TriageEngine(
            backend=HeuristicMockBackend(seed=42),
            confidence_threshold=0.15,
        ).process_manifest(manifest.model_dump())

        result = FlankExtractionEngine(
            backend=MockPoseBackend(seed=42),
            crops_dir=tmp_dir / "crops",
        ).process_triage(
            triage.model_dump(by_alias=True),
            source_records=manifest.model_dump()["records"],
        )

        for ext in result.extractions:
            assert ext.image_id
            assert ext.flank_side in ("LEFT", "RIGHT", "AMBIGUOUS")
            assert 0.0 <= ext.quality_score <= 1.0
            assert len(ext.keypoints) == 18
            assert len(ext.bbox_normalized) == 4

    def test_json_roundtrip(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """Result must serialise to JSON and parse back."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend, TriageEngine
        from src.m4_flank_pose import (
            FlankExtractionEngine,
            FlankExtractionResult,
            MockPoseBackend,
        )

        manifest = generate_manifest(full_fixture_dir)
        triage = TriageEngine(
            backend=HeuristicMockBackend(seed=42),
            confidence_threshold=0.15,
        ).process_manifest(manifest.model_dump())

        result = FlankExtractionEngine(
            backend=MockPoseBackend(seed=42),
            crops_dir=tmp_dir / "crops",
        ).process_triage(
            triage.model_dump(by_alias=True),
            source_records=manifest.model_dump()["records"],
        )

        out_path = tmp_dir / "flank_test.json"
        with open(out_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2)

        with open(out_path, "r") as f:
            raw = json.load(f)

        restored = FlankExtractionResult(**raw)
        assert restored.batch_id == result.batch_id
        assert len(restored.extractions) == len(result.extractions)

    def test_vram_discipline(self) -> None:
        """Backend must be unloaded after processing."""
        from src.m4_flank_pose import FlankExtractionEngine, MockPoseBackend

        backend = MockPoseBackend()
        engine = FlankExtractionEngine(
            backend=backend,
            crops_dir=Path("/tmp/test"),
        )
        engine.process_triage({"batch_id": "TEST", "dispatches": []})
        assert not backend._loaded

    def test_summary_counts_consistent(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """left + right + ambiguous must equal total_extractions."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend, TriageEngine
        from src.m4_flank_pose import FlankExtractionEngine, MockPoseBackend

        manifest = generate_manifest(full_fixture_dir)
        triage = TriageEngine(
            backend=HeuristicMockBackend(seed=42),
            confidence_threshold=0.15,
        ).process_manifest(manifest.model_dump())

        result = FlankExtractionEngine(
            backend=MockPoseBackend(seed=42),
            crops_dir=tmp_dir / "crops",
        ).process_triage(
            triage.model_dump(by_alias=True),
            source_records=manifest.model_dump()["records"],
        )

        s = result.summary
        assert s.left_flanks + s.right_flanks + s.ambiguous_flanks == s.total_extractions


# =====================================================================
#  Test 8: TERA-STRIPE 6-Landmark Schema & Real Model Integration
# =====================================================================

class TestTeraStripe6Landmarks:
    """Validate 6-landmark anatomical schema and YOLO11-Pose integration."""

    def test_6_landmark_names(self) -> None:
        from src.m4_flank_pose import TERA_STRIPE_6_KEYPOINT_NAMES
        assert len(TERA_STRIPE_6_KEYPOINT_NAMES) == 6
        assert "shoulder_scapula" in TERA_STRIPE_6_KEYPOINT_NAMES
        assert "hip_pelvis_root" in TERA_STRIPE_6_KEYPOINT_NAMES
        assert "spine_midpoint" in TERA_STRIPE_6_KEYPOINT_NAMES
        assert "ventral_belly_contour" in TERA_STRIPE_6_KEYPOINT_NAMES
        assert "foreleg_root" in TERA_STRIPE_6_KEYPOINT_NAMES
        assert "hindleg_root" in TERA_STRIPE_6_KEYPOINT_NAMES

    def test_6_landmark_laterality_right(self) -> None:
        from src.m4_flank_pose import Keypoint, determine_flank_side
        kps = [
            Keypoint(name="shoulder_scapula", x=0.70, y=0.40, confidence=0.95),
            Keypoint(name="hip_pelvis_root", x=0.30, y=0.40, confidence=0.95),
            Keypoint(name="spine_midpoint", x=0.50, y=0.30, confidence=0.95),
        ]
        # Shoulder is to the right of hip -> Tiger facing right -> RIGHT flank visible
        assert determine_flank_side(kps) == "RIGHT"

    def test_6_landmark_laterality_left(self) -> None:
        from src.m4_flank_pose import Keypoint, determine_flank_side
        kps = [
            Keypoint(name="shoulder_scapula", x=0.25, y=0.40, confidence=0.95),
            Keypoint(name="hip_pelvis_root", x=0.75, y=0.40, confidence=0.95),
            Keypoint(name="spine_midpoint", x=0.50, y=0.30, confidence=0.95),
        ]
        # Shoulder is to the left of hip -> Tiger facing left -> LEFT flank visible
        assert determine_flank_side(kps) == "LEFT"

    def test_6_landmark_affine_warp(self) -> None:
        from PIL import Image
        from src.m4_flank_pose import Keypoint, affine_warp_flank
        img = Image.new("RGB", (640, 480), color=(100, 150, 200))
        kps = [
            Keypoint(name="shoulder_scapula", x=0.30, y=0.50, confidence=0.95),
            Keypoint(name="hip_pelvis_root", x=0.70, y=0.50, confidence=0.95),
        ]
        crop, warp_applied = affine_warp_flank(img, kps, crop_size=224)
        assert crop.size == (224, 224)
        assert warp_applied is True

    def test_real_yolo_weights_inference(self) -> None:
        weights_path = PROJECT_ROOT / "weights" / "yolo11_pose_tiger.pt"
        if not weights_path.exists():
            pytest.skip("Trained weights not found")

        from PIL import Image
        from src.m4_flank_pose import YOLOPoseBackend

        backend = YOLOPoseBackend(
            weights_path=weights_path,
            device="cuda:0" if os.environ.get("USE_CUDA", "1") == "1" else "cpu",
        )
        backend.load()
        assert backend._model is not None
        backend.unload()

