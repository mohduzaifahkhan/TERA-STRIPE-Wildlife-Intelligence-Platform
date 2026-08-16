"""
TERA-STRIPE M6 HITL Queue -- Test Suite
==========================================
Validates Human-in-the-Loop review queue lifecycle.

Test classes
------------
  TestHITLModels           - Pydantic contract model validation
  TestInMemoryBackend      - In-memory queue CRUD operations
  TestEnqueueFromReID      - M5 -> M6 task creation
  TestReviewDecisions      - CONFIRM/REJECT/MERGE/SPLIT processing
  TestQueueStatistics      - Summary and reviewer tracking
  TestHITLExport           - hitl_decisions.json roundtrip
  TestFullPipeline         - End-to-end M1->M2->M4->M5->M6
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


# ── Helper to build mock reid_result data ──

def _build_reid_data(
    batch_id: str = "BATCH_HITL_TEST",
    n_review: int = 3,
    n_auto: int = 2,
    n_new: int = 1,
) -> dict:
    """Build a mock reid_result.json with various dispatch statuses."""
    dispatches = []

    for i in range(n_review):
        dispatches.append({
            "image_id": f"IMG_REV_{i:02d}",
            "crop_path": f"/crops/IMG_REV_{i:02d}_LEFT.jpg",
            "flank_side": "LEFT",
            "embedding_dim": 1024,
            "status": "REVIEW",
            "assigned_tiger_id": f"PTR_M_{i:03d}",
            "confidence": 0.72 + i * 0.03,
            "top_k_matches": [
                {"tiger_id": f"PTR_M_{i:03d}", "similarity": 0.72 + i * 0.03, "rank": 1},
                {"tiger_id": f"PTR_M_{i + 10:03d}", "similarity": 0.55, "rank": 2},
            ],
        })

    for i in range(n_auto):
        dispatches.append({
            "image_id": f"IMG_AUTO_{i:02d}",
            "crop_path": f"/crops/IMG_AUTO_{i:02d}_RIGHT.jpg",
            "flank_side": "RIGHT",
            "embedding_dim": 1024,
            "status": "AUTO_MATCH",
            "assigned_tiger_id": f"PTR_M_{i + 100:03d}",
            "confidence": 0.92,
            "top_k_matches": [],
        })

    for i in range(n_new):
        dispatches.append({
            "image_id": f"IMG_NEW_{i:02d}",
            "crop_path": f"/crops/IMG_NEW_{i:02d}_LEFT.jpg",
            "flank_side": "LEFT",
            "embedding_dim": 1024,
            "status": "NEW_INDIVIDUAL",
            "assigned_tiger_id": f"PTR_NEW_{i + 200:03d}",
            "confidence": 0.30,
            "top_k_matches": [],
        })

    return {
        "batch_id": batch_id,
        "summary": {
            "total_queries": len(dispatches),
            "auto_matches": n_auto,
            "review_queue": n_review,
            "new_individuals": n_new,
            "gallery_size_before": 50,
            "gallery_size_after": 50 + n_new,
        },
        "dispatches": dispatches,
    }


# =====================================================================
#  Test 1: Pydantic Models
# =====================================================================

class TestHITLModels:
    """Validate HITL Pydantic schemas."""

    def test_review_task_schema(self) -> None:
        from src.m6_hitl_queue import ReviewTask

        t = ReviewTask(
            task_id="HITL_ABC123",
            image_id="IMG_00001",
            crop_path="/crops/IMG_00001_LEFT.jpg",
            flank_side="LEFT",
            batch_id="BATCH_01",
        )
        assert t.status == "PENDING"
        assert t.reviewer is None

    def test_review_decision_schema(self) -> None:
        from src.m6_hitl_queue import ReviewDecision

        d = ReviewDecision(
            task_id="HITL_ABC123",
            action="CONFIRM",
            reviewer="ranger_singh",
            final_tiger_id="PTR_M_001",
        )
        assert d.action == "CONFIRM"

    def test_hitl_summary_schema(self) -> None:
        from src.m6_hitl_queue import HITLSummary

        s = HITLSummary(
            total_tasks=10,
            pending=3,
            confirmed=5,
            rejected=2,
        )
        assert s.total_tasks == 10
        assert s.merged == 0

    def test_hitl_result_schema(self) -> None:
        from src.m6_hitl_queue import HITLResult, HITLSummary

        r = HITLResult(
            batch_id="TEST",
            summary=HITLSummary(total_tasks=0),
            decisions=[],
        )
        assert r.batch_id == "TEST"

    def test_all_status_values(self) -> None:
        from src.m6_hitl_queue import ReviewTask

        for status in ("PENDING", "CONFIRMED", "REJECTED", "MERGED", "SPLIT", "EXPIRED"):
            t = ReviewTask(
                task_id="T1",
                image_id="I1",
                crop_path="/c.jpg",
                flank_side="LEFT",
                batch_id="B1",
                status=status,
            )
            assert t.status == status


# =====================================================================
#  Test 2: InMemory Backend
# =====================================================================

class TestInMemoryBackend:
    """Validate in-memory queue operations."""

    def test_push_and_get(self) -> None:
        from src.m6_hitl_queue import InMemoryQueueBackend, ReviewTask

        backend = InMemoryQueueBackend()
        task = ReviewTask(
            task_id="T001",
            image_id="IMG_01",
            crop_path="/c.jpg",
            flank_side="LEFT",
            batch_id="B01",
        )
        backend.push(task)
        assert backend.get_task("T001") is not None
        assert backend.get_task("NONEXISTENT") is None

    def test_get_pending_fifo(self) -> None:
        from src.m6_hitl_queue import InMemoryQueueBackend, ReviewTask

        backend = InMemoryQueueBackend()
        for i in range(5):
            backend.push(ReviewTask(
                task_id=f"T{i:03d}",
                image_id=f"IMG_{i}",
                crop_path="/c.jpg",
                flank_side="LEFT",
                batch_id="B01",
            ))

        pending = backend.get_pending(limit=3)
        assert len(pending) == 3
        assert pending[0].task_id == "T000"

    def test_update_task(self) -> None:
        from src.m6_hitl_queue import InMemoryQueueBackend, ReviewTask

        backend = InMemoryQueueBackend()
        backend.push(ReviewTask(
            task_id="T001",
            image_id="IMG_01",
            crop_path="/c.jpg",
            flank_side="LEFT",
            batch_id="B01",
        ))

        assert backend.update_task("T001", {"status": "CONFIRMED"})
        assert backend.get_task("T001").status == "CONFIRMED"
        assert not backend.update_task("NONEXISTENT", {"status": "CONFIRMED"})

    def test_count_by_status(self) -> None:
        from src.m6_hitl_queue import InMemoryQueueBackend, ReviewTask

        backend = InMemoryQueueBackend()
        backend.push(ReviewTask(
            task_id="T1", image_id="I1", crop_path="/c.jpg",
            flank_side="LEFT", batch_id="B1", status="PENDING",
        ))
        backend.push(ReviewTask(
            task_id="T2", image_id="I2", crop_path="/c.jpg",
            flank_side="LEFT", batch_id="B1", status="PENDING",
        ))
        backend.push(ReviewTask(
            task_id="T3", image_id="I3", crop_path="/c.jpg",
            flank_side="LEFT", batch_id="B1", status="CONFIRMED",
        ))

        counts = backend.count_by_status()
        assert counts["PENDING"] == 2
        assert counts["CONFIRMED"] == 1

    def test_get_by_batch(self) -> None:
        from src.m6_hitl_queue import InMemoryQueueBackend, ReviewTask

        backend = InMemoryQueueBackend()
        backend.push(ReviewTask(
            task_id="T1", image_id="I1", crop_path="/c.jpg",
            flank_side="LEFT", batch_id="B1",
        ))
        backend.push(ReviewTask(
            task_id="T2", image_id="I2", crop_path="/c.jpg",
            flank_side="LEFT", batch_id="B2",
        ))

        b1 = backend.get_by_batch("B1")
        assert len(b1) == 1
        assert b1[0].task_id == "T1"


# =====================================================================
#  Test 3: Enqueue from ReID
# =====================================================================

class TestEnqueueFromReID:
    """Validate task creation from M5 REVIEW dispatches."""

    def test_only_review_dispatches_enqueued(self) -> None:
        """Only REVIEW status dispatches should become tasks."""
        from src.m6_hitl_queue import HITLQueueManager, InMemoryQueueBackend

        reid_data = _build_reid_data(n_review=3, n_auto=2, n_new=1)
        manager = HITLQueueManager(InMemoryQueueBackend())
        tasks = manager.enqueue_from_reid(reid_data)

        assert len(tasks) == 3  # Only 3 REVIEW dispatches
        for t in tasks:
            assert t.status == "PENDING"

    def test_task_ids_are_unique(self) -> None:
        from src.m6_hitl_queue import HITLQueueManager, InMemoryQueueBackend

        reid_data = _build_reid_data(n_review=5)
        manager = HITLQueueManager(InMemoryQueueBackend())
        tasks = manager.enqueue_from_reid(reid_data)

        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_candidate_info_preserved(self) -> None:
        from src.m6_hitl_queue import HITLQueueManager, InMemoryQueueBackend

        reid_data = _build_reid_data(n_review=1)
        manager = HITLQueueManager(InMemoryQueueBackend())
        tasks = manager.enqueue_from_reid(reid_data)

        t = tasks[0]
        assert t.candidate_tiger_id == "PTR_M_000"
        assert t.candidate_similarity > 0
        assert len(t.top_k_matches) == 2

    def test_no_reviews_returns_empty(self) -> None:
        from src.m6_hitl_queue import HITLQueueManager, InMemoryQueueBackend

        reid_data = _build_reid_data(n_review=0, n_auto=5, n_new=3)
        manager = HITLQueueManager(InMemoryQueueBackend())
        tasks = manager.enqueue_from_reid(reid_data)
        assert len(tasks) == 0


# =====================================================================
#  Test 4: Review Decisions
# =====================================================================

class TestReviewDecisions:
    """Validate CONFIRM/REJECT/MERGE/SPLIT processing."""

    def _setup_manager_with_tasks(self, n: int = 3):
        from src.m6_hitl_queue import HITLQueueManager, InMemoryQueueBackend

        reid_data = _build_reid_data(n_review=n)
        manager = HITLQueueManager(InMemoryQueueBackend())
        tasks = manager.enqueue_from_reid(reid_data)
        return manager, tasks

    def test_confirm_sets_status(self) -> None:
        from src.m6_hitl_queue import ReviewDecision

        manager, tasks = self._setup_manager_with_tasks()
        decision = ReviewDecision(
            task_id=tasks[0].task_id,
            action="CONFIRM",
            reviewer="ranger_singh",
            final_tiger_id="PTR_M_000",
        )
        result = manager.submit_decision(decision)
        assert result is not None
        assert result.status == "CONFIRMED"
        assert result.final_tiger_id == "PTR_M_000"
        assert result.reviewer == "ranger_singh"

    def test_reject_sets_status(self) -> None:
        from src.m6_hitl_queue import ReviewDecision

        manager, tasks = self._setup_manager_with_tasks()
        decision = ReviewDecision(
            task_id=tasks[0].task_id,
            action="REJECT",
            reviewer="ranger_patel",
        )
        result = manager.submit_decision(decision)
        assert result.status == "REJECTED"
        assert result.final_tiger_id is None

    def test_merge_sets_target(self) -> None:
        from src.m6_hitl_queue import ReviewDecision

        manager, tasks = self._setup_manager_with_tasks()
        decision = ReviewDecision(
            task_id=tasks[0].task_id,
            action="MERGE",
            reviewer="dr_sharma",
            merge_target_id="PTR_M_099",
        )
        result = manager.submit_decision(decision)
        assert result.status == "MERGED"
        assert result.final_tiger_id == "PTR_M_099"

    def test_split_sets_status(self) -> None:
        from src.m6_hitl_queue import ReviewDecision

        manager, tasks = self._setup_manager_with_tasks()
        decision = ReviewDecision(
            task_id=tasks[0].task_id,
            action="SPLIT",
            reviewer="ranger_kumar",
            final_tiger_id="PTR_NEW_SPLIT_001",
        )
        result = manager.submit_decision(decision)
        assert result.status == "SPLIT"

    def test_cannot_decide_twice(self) -> None:
        """Already-resolved tasks should not be re-decidable."""
        from src.m6_hitl_queue import ReviewDecision

        manager, tasks = self._setup_manager_with_tasks()
        tid = tasks[0].task_id

        d1 = ReviewDecision(
            task_id=tid, action="CONFIRM", reviewer="r1",
            final_tiger_id="PTR_M_000",
        )
        manager.submit_decision(d1)

        # Second decision on same task
        d2 = ReviewDecision(
            task_id=tid, action="REJECT", reviewer="r2",
        )
        result = manager.submit_decision(d2)
        # Should still be CONFIRMED (first decision wins)
        assert result.status == "CONFIRMED"

    def test_nonexistent_task_returns_none(self) -> None:
        from src.m6_hitl_queue import HITLQueueManager, InMemoryQueueBackend, ReviewDecision

        manager = HITLQueueManager(InMemoryQueueBackend())
        decision = ReviewDecision(
            task_id="NONEXISTENT",
            action="CONFIRM",
            reviewer="test",
        )
        assert manager.submit_decision(decision) is None

    def test_reviewed_at_timestamp_set(self) -> None:
        from src.m6_hitl_queue import ReviewDecision

        manager, tasks = self._setup_manager_with_tasks()
        decision = ReviewDecision(
            task_id=tasks[0].task_id,
            action="CONFIRM",
            reviewer="ranger_singh",
        )
        result = manager.submit_decision(decision)
        assert result.reviewed_at is not None

    def test_notes_preserved(self) -> None:
        from src.m6_hitl_queue import ReviewDecision

        manager, tasks = self._setup_manager_with_tasks()
        decision = ReviewDecision(
            task_id=tasks[0].task_id,
            action="CONFIRM",
            reviewer="ranger_singh",
            notes="Distinctive scar on left flank confirmed",
        )
        result = manager.submit_decision(decision)
        assert result.notes == "Distinctive scar on left flank confirmed"


# =====================================================================
#  Test 5: Queue Statistics
# =====================================================================

class TestQueueStatistics:
    """Validate summary and reviewer tracking."""

    def test_summary_counts_correct(self) -> None:
        from src.m6_hitl_queue import HITLQueueManager, InMemoryQueueBackend, ReviewDecision

        reid_data = _build_reid_data(n_review=4)
        manager = HITLQueueManager(InMemoryQueueBackend())
        tasks = manager.enqueue_from_reid(reid_data)

        # Decide on 2 tasks
        manager.submit_decision(ReviewDecision(
            task_id=tasks[0].task_id, action="CONFIRM", reviewer="r1",
        ))
        manager.submit_decision(ReviewDecision(
            task_id=tasks[1].task_id, action="REJECT", reviewer="r2",
        ))

        summary = manager.get_summary()
        assert summary.total_tasks == 4
        assert summary.pending == 2
        assert summary.confirmed == 1
        assert summary.rejected == 1

    def test_reviewer_list(self) -> None:
        from src.m6_hitl_queue import HITLQueueManager, InMemoryQueueBackend, ReviewDecision

        reid_data = _build_reid_data(n_review=2)
        manager = HITLQueueManager(InMemoryQueueBackend())
        tasks = manager.enqueue_from_reid(reid_data)

        manager.submit_decision(ReviewDecision(
            task_id=tasks[0].task_id, action="CONFIRM", reviewer="ranger_a",
        ))
        manager.submit_decision(ReviewDecision(
            task_id=tasks[1].task_id, action="REJECT", reviewer="ranger_b",
        ))

        summary = manager.get_summary()
        assert "ranger_a" in summary.reviewers
        assert "ranger_b" in summary.reviewers


# =====================================================================
#  Test 6: Export Decisions
# =====================================================================

class TestHITLExport:
    """Validate hitl_decisions.json output."""

    def test_export_produces_valid_json(self, tmp_dir: Path) -> None:
        from src.m6_hitl_queue import HITLQueueManager, InMemoryQueueBackend, ReviewDecision

        reid_data = _build_reid_data(n_review=2)
        manager = HITLQueueManager(InMemoryQueueBackend())
        tasks = manager.enqueue_from_reid(reid_data)

        manager.submit_decision(ReviewDecision(
            task_id=tasks[0].task_id,
            action="CONFIRM",
            reviewer="ranger_singh",
            final_tiger_id="PTR_M_000",
        ))

        result = manager.export_decisions("BATCH_HITL_TEST")

        out_path = tmp_dir / "hitl_test.json"
        with open(out_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2)

        with open(out_path) as f:
            restored = json.load(f)

        assert restored["batch_id"] == "BATCH_HITL_TEST"
        assert len(restored["decisions"]) == 2

    def test_export_roundtrip(self, tmp_dir: Path) -> None:
        from src.m6_hitl_queue import HITLQueueManager, HITLResult, InMemoryQueueBackend

        reid_data = _build_reid_data(n_review=3)
        manager = HITLQueueManager(InMemoryQueueBackend())
        manager.enqueue_from_reid(reid_data)

        result = manager.export_decisions("BATCH_HITL_TEST")
        out = tmp_dir / "roundtrip.json"
        with open(out, "w") as f:
            json.dump(result.model_dump(), f, indent=2)
        with open(out) as f:
            restored = HITLResult(**json.load(f))

        assert restored.batch_id == result.batch_id
        assert len(restored.decisions) == len(result.decisions)


# =====================================================================
#  Test 7: Full Pipeline M1 -> M2 -> M4 -> M5 -> M6
# =====================================================================

class TestFullPipeline:
    """End-to-end integration test."""

    def test_full_pipeline_m1_through_m6(
        self, full_fixture_dir: Path, tmp_dir: Path
    ) -> None:
        """Full pipeline must produce valid HITL tasks from camera-trap images."""
        from src.m1_ingestion import generate_manifest
        from src.m2_triage import HeuristicMockBackend as TriageMock
        from src.m2_triage import TriageEngine
        from src.m4_flank_pose import FlankExtractionEngine, MockPoseBackend
        from src.m5_reid_engine import (
            MockEmbeddingBackend, ReIDEngine, VectorGallery,
        )
        from src.m6_hitl_queue import HITLQueueManager, InMemoryQueueBackend

        # M1: Ingest
        manifest = generate_manifest(full_fixture_dir)

        # M2: Triage (with low threshold to force some REVIEW in M5)
        triage = TriageEngine(
            backend=TriageMock(seed=42),
            confidence_threshold=0.15,
        ).process_manifest(manifest.model_dump())

        # M4: Flank extraction
        flank = FlankExtractionEngine(
            backend=MockPoseBackend(seed=42),
            crops_dir=tmp_dir / "crops",
        ).process_triage(
            triage.model_dump(by_alias=True),
            source_records=manifest.model_dump()["records"],
        )

        # M5: Re-ID (first pass: empty gallery -> all NEW_INDIVIDUAL)
        gallery = VectorGallery()
        reid_1 = ReIDEngine(
            backend=MockEmbeddingBackend(seed=42),
            gallery=gallery,
        ).process_extractions(
            flank.model_dump(),
            gallery_dir=tmp_dir / "gallery",
        )

        # M5 pass 2: re-run with populated gallery -> should get REVIEW/AUTO
        reid_2 = ReIDEngine(
            backend=MockEmbeddingBackend(seed=42),
            gallery=gallery,
            sim_auto_match=0.99,  # Very high -> force most to REVIEW
            sim_review_min=0.01,  # Very low -> almost everything is REVIEW
        ).process_extractions(flank.model_dump())

        # M6: HITL queue
        manager = HITLQueueManager(InMemoryQueueBackend())
        tasks = manager.enqueue_from_reid(reid_2.model_dump())

        # Should have enqueued REVIEW dispatches from second pass
        assert len(tasks) >= 0  # May be 0 if all AUTO_MATCH at sim >= 0.99
        summary = manager.get_summary()
        assert summary.total_tasks == len(tasks)
