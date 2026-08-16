"""
TERA-STRIPE M5 Re-ID Engine -- Test Suite
============================================
Validates DINOv2 + ArcFace re-identification pipeline.

Test classes
------------
  TestReIDModels          - Pydantic contract model validation
  TestVectorGallery       - Gallery add/search/save/load operations
  TestMockEmbedding       - Mock backend determinism and API
  TestReIDClassification  - AUTO_MATCH / REVIEW / NEW_INDIVIDUAL logic
  TestReIDEngine          - End-to-end pipeline M1->M2->M4->M5
  TestGalleryPersistence  - Save/load roundtrip on disk
"""

from __future__ import annotations

import json
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


# =====================================================================
#  Test 1: Pydantic Contract Models
# =====================================================================

class TestReIDModels:
    """Validate reid_result.json Pydantic schemas."""

    def test_reid_match_schema(self) -> None:
        from src.m5_reid_engine import ReIDMatch

        m = ReIDMatch(tiger_id="PTR_M_001", similarity=0.92, rank=1)
        assert m.tiger_id == "PTR_M_001"
        assert m.similarity == 0.92

    def test_reid_dispatch_schema(self) -> None:
        from src.m5_reid_engine import ReIDDispatch

        d = ReIDDispatch(
            image_id="IMG_00001",
            crop_path="/crops/IMG_00001_LEFT.jpg",
            flank_side="LEFT",
            status="AUTO_MATCH",
            assigned_tiger_id="PTR_M_001",
            confidence=0.92,
        )
        assert d.status == "AUTO_MATCH"
        assert d.embedding_dim == 1024

    def test_reid_summary_schema(self) -> None:
        from src.m5_reid_engine import ReIDSummary

        s = ReIDSummary(
            total_queries=10,
            auto_matches=3,
            review_queue=4,
            new_individuals=3,
            gallery_size_before=20,
            gallery_size_after=23,
        )
        assert s.total_queries == 10
        assert s.gallery_size_after == 23

    def test_reid_result_schema(self) -> None:
        from src.m5_reid_engine import ReIDResult, ReIDSummary

        r = ReIDResult(
            batch_id="BATCH_TEST",
            summary=ReIDSummary(
                total_queries=0,
                gallery_size_before=0,
                gallery_size_after=0,
            ),
            dispatches=[],
        )
        assert r.batch_id == "BATCH_TEST"
        assert r.similarity_thresholds["auto_match"] == 0.85

    def test_all_status_values(self) -> None:
        from src.m5_reid_engine import ReIDDispatch

        for status in ("AUTO_MATCH", "REVIEW", "NEW_INDIVIDUAL"):
            d = ReIDDispatch(
                image_id="test",
                crop_path="/test.jpg",
                flank_side="LEFT",
                status=status,
                confidence=0.5,
            )
            assert d.status == status


# =====================================================================
#  Test 2: Vector Gallery
# =====================================================================

class TestVectorGallery:
    """Validate gallery CRUD and search operations."""

    def test_empty_gallery(self) -> None:
        from src.m5_reid_engine import VectorGallery

        g = VectorGallery()
        assert g.size == 0
        assert g.search(np.random.randn(1024).astype(np.float32)) == []

    def test_add_and_search(self) -> None:
        from src.m5_reid_engine import VectorGallery

        g = VectorGallery(dimension=128)
        emb1 = np.random.randn(128).astype(np.float32)
        emb2 = np.random.randn(128).astype(np.float32)

        g.add("TIGER_001", emb1)
        g.add("TIGER_002", emb2)
        assert g.size == 2

        # Search with emb1 should return TIGER_001 as top match
        matches = g.search(emb1, top_k=2)
        assert len(matches) == 2
        assert matches[0].tiger_id == "TIGER_001"
        assert matches[0].rank == 1
        assert matches[0].similarity > matches[1].similarity

    def test_search_returns_correct_top_k(self) -> None:
        from src.m5_reid_engine import VectorGallery

        g = VectorGallery(dimension=64)
        for i in range(10):
            g.add(f"T_{i:02d}", np.random.randn(64).astype(np.float32))

        matches = g.search(np.random.randn(64).astype(np.float32), top_k=5)
        assert len(matches) == 5
        assert all(m.rank == i + 1 for i, m in enumerate(matches))

    def test_top_k_capped_by_gallery_size(self) -> None:
        from src.m5_reid_engine import VectorGallery

        g = VectorGallery(dimension=64)
        g.add("T_01", np.random.randn(64).astype(np.float32))
        g.add("T_02", np.random.randn(64).astype(np.float32))

        matches = g.search(np.random.randn(64).astype(np.float32), top_k=10)
        assert len(matches) == 2

    def test_update_existing_identity(self) -> None:
        from src.m5_reid_engine import VectorGallery

        g = VectorGallery(dimension=64)
        emb1 = np.ones(64, dtype=np.float32)
        emb2 = np.ones(64, dtype=np.float32) * 2

        g.add("TIGER_001", emb1)
        assert g.size == 1
        g.add("TIGER_001", emb2)  # Update, not duplicate
        assert g.size == 1

    def test_contains(self) -> None:
        from src.m5_reid_engine import VectorGallery

        g = VectorGallery(dimension=64)
        g.add("TIGER_001", np.random.randn(64).astype(np.float32))

        assert g.contains("TIGER_001")
        assert not g.contains("TIGER_999")

    def test_get_embedding(self) -> None:
        from src.m5_reid_engine import VectorGallery

        g = VectorGallery(dimension=64)
        emb = np.random.randn(64).astype(np.float32)
        g.add("TIGER_001", emb)

        retrieved = g.get_embedding("TIGER_001")
        assert retrieved is not None
        assert retrieved.shape == (64,)
        assert g.get_embedding("NONEXISTENT") is None

    def test_cosine_similarity_range(self) -> None:
        """All similarity values must be clamped to [0, 1]."""
        from src.m5_reid_engine import VectorGallery

        g = VectorGallery(dimension=64)
        for i in range(5):
            g.add(f"T_{i}", np.random.randn(64).astype(np.float32))

        query = np.random.randn(64).astype(np.float32)
        matches = g.search(query, top_k=5)
        for m in matches:
            assert 0.0 <= m.similarity <= 1.0


# =====================================================================
#  Test 3: Mock Embedding Backend
# =====================================================================

class TestMockEmbedding:
    """Validate mock embedding backend."""

    def test_mock_lifecycle(self) -> None:
        from src.m5_reid_engine import MockEmbeddingBackend

        mock = MockEmbeddingBackend()
        mock.load()
        assert mock._loaded
        mock.unload()
        assert not mock._loaded

    def test_mock_requires_load(self) -> None:
        from src.m5_reid_engine import MockEmbeddingBackend

        mock = MockEmbeddingBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            mock.extract_batch([Path("dummy.jpg")])

    def test_mock_returns_correct_dimension(
        self, sample_station_dir: Path
    ) -> None:
        """Embeddings must be 1024-dimensional."""
        from src.m5_reid_engine import MockEmbeddingBackend, EMBEDDING_DIM

        images = sorted(sample_station_dir.glob("*.JPG"))
        if not images:
            pytest.skip("No fixtures")

        mock = MockEmbeddingBackend(seed=42)
        mock.load()
        embeddings = mock.extract_batch(images)
        mock.unload()

        assert len(embeddings) == len(images)
        for emb in embeddings:
            assert emb.shape == (EMBEDDING_DIM,)

    def test_mock_embeddings_normalised(
        self, sample_station_dir: Path
    ) -> None:
        """All embeddings must be L2-normalised (norm ~= 1.0)."""
        from src.m5_reid_engine import MockEmbeddingBackend

        images = sorted(sample_station_dir.glob("*.JPG"))
        if not images:
            pytest.skip("No fixtures")

        mock = MockEmbeddingBackend(seed=42)
        mock.load()
        embeddings = mock.extract_batch(images)
        mock.unload()

        for emb in embeddings:
            assert np.linalg.norm(emb) == pytest.approx(1.0, abs=0.01)

    def test_mock_deterministic(self, sample_station_dir: Path) -> None:
        """Same inputs with same seed must produce identical embeddings."""
        from src.m5_reid_engine import MockEmbeddingBackend

        images = sorted(sample_station_dir.glob("*.JPG"))
        if not images:
            pytest.skip("No fixtures")

        m1 = MockEmbeddingBackend(seed=42)
        m1.load()
        e1 = m1.extract_batch(images)
        m1.unload()

        m2 = MockEmbeddingBackend(seed=42)
        m2.load()
        e2 = m2.extract_batch(images)
        m2.unload()

        for a, b in zip(e1, e2):
            assert np.allclose(a, b)


# =====================================================================
#  Test 4: Re-ID Classification Logic
# =====================================================================

class TestReIDClassification:
    """Validate AUTO_MATCH / REVIEW / NEW_INDIVIDUAL thresholds."""

    def test_no_matches_is_new_individual(self) -> None:
        from src.m5_reid_engine import (
            MockEmbeddingBackend, ReIDEngine, VectorGallery,
        )

        engine = ReIDEngine(
            backend=MockEmbeddingBackend(),
            gallery=VectorGallery(),
        )
        status, tiger_id, conf = engine._classify_match([])
        assert status == "NEW_INDIVIDUAL"
        assert tiger_id is None

    def test_high_similarity_is_auto_match(self) -> None:
        from src.m5_reid_engine import (
            MockEmbeddingBackend, ReIDEngine, ReIDMatch, VectorGallery,
        )

        engine = ReIDEngine(
            backend=MockEmbeddingBackend(),
            gallery=VectorGallery(),
        )
        matches = [ReIDMatch(tiger_id="PTR_M_001", similarity=0.92, rank=1)]
        status, tiger_id, conf = engine._classify_match(matches)
        assert status == "AUTO_MATCH"
        assert tiger_id == "PTR_M_001"

    def test_medium_similarity_is_review(self) -> None:
        from src.m5_reid_engine import (
            MockEmbeddingBackend, ReIDEngine, ReIDMatch, VectorGallery,
        )

        engine = ReIDEngine(
            backend=MockEmbeddingBackend(),
            gallery=VectorGallery(),
        )
        matches = [ReIDMatch(tiger_id="PTR_M_002", similarity=0.72, rank=1)]
        status, tiger_id, conf = engine._classify_match(matches)
        assert status == "REVIEW"
        assert tiger_id == "PTR_M_002"

    def test_low_similarity_is_new_individual(self) -> None:
        from src.m5_reid_engine import (
            MockEmbeddingBackend, ReIDEngine, ReIDMatch, VectorGallery,
        )

        engine = ReIDEngine(
            backend=MockEmbeddingBackend(),
            gallery=VectorGallery(),
        )
        matches = [ReIDMatch(tiger_id="PTR_M_003", similarity=0.40, rank=1)]
        status, tiger_id, conf = engine._classify_match(matches)
        assert status == "NEW_INDIVIDUAL"
        assert tiger_id is None

    def test_threshold_boundary_auto(self) -> None:
        """Exactly 0.85 must be AUTO_MATCH."""
        from src.m5_reid_engine import (
            MockEmbeddingBackend, ReIDEngine, ReIDMatch, VectorGallery,
        )

        engine = ReIDEngine(
            backend=MockEmbeddingBackend(),
            gallery=VectorGallery(),
        )
        matches = [ReIDMatch(tiger_id="PTR_M_004", similarity=0.85, rank=1)]
        status, _, _ = engine._classify_match(matches)
        assert status == "AUTO_MATCH"

    def test_threshold_boundary_review(self) -> None:
        """Exactly 0.60 must be REVIEW."""
        from src.m5_reid_engine import (
            MockEmbeddingBackend, ReIDEngine, ReIDMatch, VectorGallery,
        )

        engine = ReIDEngine(
            backend=MockEmbeddingBackend(),
            gallery=VectorGallery(),
        )
        matches = [ReIDMatch(tiger_id="PTR_M_005", similarity=0.60, rank=1)]
        status, _, _ = engine._classify_match(matches)
        assert status == "REVIEW"


# =====================================================================
#  Test 5: Gallery Persistence
# =====================================================================

class TestGalleryPersistence:
    """Validate gallery save/load roundtrip."""

    def test_save_and_load(self, tmp_dir: Path) -> None:
        from src.m5_reid_engine import VectorGallery

        g = VectorGallery(dimension=128)
        e1 = np.random.randn(128).astype(np.float32)
        e2 = np.random.randn(128).astype(np.float32)
        g.add("TIGER_001", e1, metadata={"sex": "MALE"})
        g.add("TIGER_002", e2)

        gallery_dir = tmp_dir / "gallery"
        g.save(gallery_dir)

        # Load into new gallery
        g2 = VectorGallery(dimension=128)
        g2.load(gallery_dir)

        assert g2.size == 2
        assert g2.contains("TIGER_001")
        assert g2.contains("TIGER_002")

        # Search should still work
        matches = g2.search(e1, top_k=2)
        assert matches[0].tiger_id == "TIGER_001"

    def test_load_nonexistent_is_empty(self, tmp_dir: Path) -> None:
        from src.m5_reid_engine import VectorGallery

        g = VectorGallery()
        g.load(tmp_dir / "nonexistent_gallery")
        assert g.size == 0

    def test_files_created_on_save(self, tmp_dir: Path) -> None:
        from src.m5_reid_engine import VectorGallery

        g = VectorGallery(dimension=64)
        g.add("T1", np.random.randn(64).astype(np.float32))
        g.save(tmp_dir / "gal")

        assert (tmp_dir / "gal" / "gallery_index.json").exists()
        assert (tmp_dir / "gal" / "gallery_embeddings.npy").exists()


# =====================================================================
#  Test 6: End-to-End Re-ID Pipeline
# =====================================================================

class TestReIDEngine:
    """Full pipeline test: M1 -> M2 -> M4 -> M5."""

    def test_full_pipeline_produces_reid_result(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """End-to-end pipeline must produce valid ReIDResult."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend as TriageMock
        from src.m2_triage import TriageEngine
        from src.m4_flank_pose import FlankExtractionEngine, MockPoseBackend
        from src.m5_reid_engine import (
            MockEmbeddingBackend,
            ReIDEngine,
            ReIDResult,
            VectorGallery,
        )

        # M1
        manifest = generate_manifest(full_fixture_dir)

        # M2
        triage = TriageEngine(
            backend=TriageMock(seed=42),
            confidence_threshold=0.15,
        ).process_manifest(manifest.model_dump())

        # M4
        crops_dir = tmp_dir / "crops"
        flank_result = FlankExtractionEngine(
            backend=MockPoseBackend(seed=42),
            crops_dir=crops_dir,
        ).process_triage(
            triage.model_dump(by_alias=True),
            source_records=manifest.model_dump()["records"],
        )

        # M5
        gallery = VectorGallery()
        engine = ReIDEngine(
            backend=MockEmbeddingBackend(seed=42),
            gallery=gallery,
        )
        result = engine.process_extractions(
            flank_result.model_dump(),
            gallery_dir=tmp_dir / "gallery",
        )

        assert isinstance(result, ReIDResult)
        assert result.batch_id == manifest.batch_id
        assert result.summary.total_queries > 0
        assert len(result.dispatches) == result.summary.total_queries

    def test_new_individuals_added_to_gallery(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """First-time crops must create new gallery entries."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend as TriageMock
        from src.m2_triage import TriageEngine
        from src.m4_flank_pose import FlankExtractionEngine, MockPoseBackend
        from src.m5_reid_engine import (
            MockEmbeddingBackend, ReIDEngine, VectorGallery,
        )

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

        gallery = VectorGallery()
        engine = ReIDEngine(
            backend=MockEmbeddingBackend(seed=42),
            gallery=gallery,
        )
        result = engine.process_extractions(flank.model_dump())

        # Empty gallery -> all should be NEW_INDIVIDUAL
        assert result.summary.new_individuals == result.summary.total_queries
        assert result.summary.gallery_size_after == result.summary.total_queries

    def test_summary_counts_consistent(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """auto + review + new must equal total_queries."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend as TriageMock
        from src.m2_triage import TriageEngine
        from src.m4_flank_pose import FlankExtractionEngine, MockPoseBackend
        from src.m5_reid_engine import (
            MockEmbeddingBackend, ReIDEngine, VectorGallery,
        )

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

        result = ReIDEngine(
            backend=MockEmbeddingBackend(seed=42),
            gallery=VectorGallery(),
        ).process_extractions(flank.model_dump())

        s = result.summary
        assert s.auto_matches + s.review_queue + s.new_individuals == s.total_queries

    def test_vram_discipline(self) -> None:
        """Backend must be unloaded after processing."""
        from src.m5_reid_engine import (
            MockEmbeddingBackend, ReIDEngine, VectorGallery,
        )

        backend = MockEmbeddingBackend()
        engine = ReIDEngine(backend=backend, gallery=VectorGallery())
        engine.process_extractions({"batch_id": "TEST", "extractions": []})
        assert not backend._loaded

    def test_json_roundtrip(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """ReIDResult must serialise and deserialise correctly."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend as TriageMock
        from src.m2_triage import TriageEngine
        from src.m4_flank_pose import FlankExtractionEngine, MockPoseBackend
        from src.m5_reid_engine import (
            MockEmbeddingBackend, ReIDEngine, ReIDResult, VectorGallery,
        )

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

        result = ReIDEngine(
            backend=MockEmbeddingBackend(seed=42),
            gallery=VectorGallery(),
        ).process_extractions(flank.model_dump())

        out = tmp_dir / "reid_test.json"
        with open(out, "w") as f:
            json.dump(result.model_dump(), f, indent=2)
        with open(out) as f:
            restored = ReIDResult(**json.load(f))

        assert restored.batch_id == result.batch_id
        assert len(restored.dispatches) == len(result.dispatches)
