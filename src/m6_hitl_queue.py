"""
TERA-STRIPE Module 6 -- Human-in-the-Loop (HITL) Review Queue
===============================================================
Manages uncertain re-identification matches (REVIEW status from M5)
through a structured review workflow. Supports CONFIRM, REJECT, MERGE,
and SPLIT actions with full audit trail.

Backends:
  InMemoryQueueBackend  -- Testing and local development
  RedisQueueBackend     -- Production (Redis 7+)

Data contract:
  Input  : reid_result.json  (from M5, REVIEW dispatches only)
  Output : hitl_decisions.json

CLI Usage
---------
  # Enqueue review tasks from Re-ID result
  python -m src.m6_hitl_queue \\
      --enqueue --reid-result ./data/manifests/reid_BATCH.json

  # List pending reviews
  python -m src.m6_hitl_queue --list-pending --limit 10

  # Submit a decision
  python -m src.m6_hitl_queue \\
      --decide --task-id TASK_001 --action CONFIRM \\
      --tiger-id PTR_M_001 --reviewer ranger_singh

  # Show queue statistics
  python -m src.m6_hitl_queue --stats

Reference: Master Context Packet -- HITL Queue, Reviewer Workflow
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("tera_stripe.m6_hitl")

# ── IST timezone ─────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))


# =====================================================================
#  Pydantic Contract Models
# =====================================================================

class MatchCandidate(BaseModel):
    """A candidate match shown to the reviewer."""
    tiger_id: str
    similarity: float
    rank: int


class ReviewTask(BaseModel):
    """A single HITL review task."""
    task_id: str
    image_id: str
    crop_path: str
    flank_side: str
    batch_id: str
    candidate_tiger_id: str | None = None
    candidate_similarity: float = 0.0
    top_k_matches: list[MatchCandidate] = []
    status: Literal[
        "PENDING", "CONFIRMED", "REJECTED", "MERGED", "SPLIT", "EXPIRED"
    ] = "PENDING"
    created_at: str = ""
    reviewed_at: str | None = None
    reviewer: str | None = None
    action: str | None = None
    final_tiger_id: str | None = None
    notes: str | None = None


class ReviewDecision(BaseModel):
    """A reviewer's decision on a task."""
    task_id: str
    action: Literal["CONFIRM", "REJECT", "MERGE", "SPLIT"]
    reviewer: str
    final_tiger_id: str | None = None
    merge_target_id: str | None = None
    notes: str = ""


class HITLSummary(BaseModel):
    """Queue statistics."""
    total_tasks: int = 0
    pending: int = 0
    confirmed: int = 0
    rejected: int = 0
    merged: int = 0
    split: int = 0
    expired: int = 0
    avg_review_time_seconds: float | None = None
    reviewers: list[str] = []


class HITLResult(BaseModel):
    """Output contract for hitl_decisions.json."""
    batch_id: str
    summary: HITLSummary
    decisions: list[ReviewTask]


# =====================================================================
#  Queue Backend Interface
# =====================================================================

class QueueBackend(ABC):
    """Abstract storage backend for review tasks."""

    @abstractmethod
    def push(self, task: ReviewTask) -> None:
        """Add a task to the queue."""

    @abstractmethod
    def get_pending(self, limit: int = 20) -> list[ReviewTask]:
        """Retrieve pending tasks (FIFO order)."""

    @abstractmethod
    def get_task(self, task_id: str) -> ReviewTask | None:
        """Retrieve a specific task by ID."""

    @abstractmethod
    def update_task(self, task_id: str, updates: dict) -> bool:
        """Update task fields. Returns True if successful."""

    @abstractmethod
    def get_all(self) -> list[ReviewTask]:
        """Retrieve all tasks regardless of status."""

    @abstractmethod
    def get_by_batch(self, batch_id: str) -> list[ReviewTask]:
        """Retrieve all tasks for a specific batch."""

    @abstractmethod
    def count_by_status(self) -> dict[str, int]:
        """Count tasks grouped by status."""


# =====================================================================
#  In-Memory Backend (Testing / Local Dev)
# =====================================================================

class InMemoryQueueBackend(QueueBackend):
    """
    In-memory queue backend for testing and local development.

    Thread-safe for single-process use. Tasks stored in a dict
    keyed by task_id.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, ReviewTask] = {}
        self._order: list[str] = []  # FIFO order tracking

    def push(self, task: ReviewTask) -> None:
        self._tasks[task.task_id] = task
        if task.task_id not in self._order:
            self._order.append(task.task_id)

    def get_pending(self, limit: int = 20) -> list[ReviewTask]:
        pending = []
        for tid in self._order:
            if tid in self._tasks and self._tasks[tid].status == "PENDING":
                pending.append(self._tasks[tid])
                if len(pending) >= limit:
                    break
        return pending

    def get_task(self, task_id: str) -> ReviewTask | None:
        return self._tasks.get(task_id)

    def update_task(self, task_id: str, updates: dict) -> bool:
        if task_id not in self._tasks:
            return False
        task = self._tasks[task_id]
        updated_data = task.model_dump()
        updated_data.update(updates)
        self._tasks[task_id] = ReviewTask(**updated_data)
        return True

    def get_all(self) -> list[ReviewTask]:
        return [self._tasks[tid] for tid in self._order if tid in self._tasks]

    def get_by_batch(self, batch_id: str) -> list[ReviewTask]:
        return [
            t for t in self._tasks.values()
            if t.batch_id == batch_id
        ]

    def count_by_status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in self._tasks.values():
            counts[task.status] = counts.get(task.status, 0) + 1
        return counts


# =====================================================================
#  Redis Backend (Production)
# =====================================================================

class RedisQueueBackend(QueueBackend):
    """
    Redis-backed queue for production deployment.

    Uses Redis hashes for task storage and a sorted set for FIFO ordering.
    Requires: pip install redis
    """

    QUEUE_KEY = "tera_stripe:hitl:queue"
    TASK_PREFIX = "tera_stripe:hitl:task:"

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self.redis_url = redis_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(self.redis_url)
                self._client.ping()
            except Exception as exc:
                raise ConnectionError(
                    f"Cannot connect to Redis at {self.redis_url}: {exc}"
                ) from exc
        return self._client

    def push(self, task: ReviewTask) -> None:
        client = self._get_client()
        key = f"{self.TASK_PREFIX}{task.task_id}"
        client.set(key, task.model_dump_json())
        client.rpush(self.QUEUE_KEY, task.task_id)

    def get_pending(self, limit: int = 20) -> list[ReviewTask]:
        client = self._get_client()
        all_ids = client.lrange(self.QUEUE_KEY, 0, -1)
        pending = []
        for tid_bytes in all_ids:
            tid = tid_bytes.decode() if isinstance(tid_bytes, bytes) else tid_bytes
            task = self.get_task(tid)
            if task and task.status == "PENDING":
                pending.append(task)
                if len(pending) >= limit:
                    break
        return pending

    def get_task(self, task_id: str) -> ReviewTask | None:
        client = self._get_client()
        key = f"{self.TASK_PREFIX}{task_id}"
        data = client.get(key)
        if data is None:
            return None
        return ReviewTask(**json.loads(data))

    def update_task(self, task_id: str, updates: dict) -> bool:
        task = self.get_task(task_id)
        if task is None:
            return False
        updated_data = task.model_dump()
        updated_data.update(updates)
        updated_task = ReviewTask(**updated_data)
        client = self._get_client()
        key = f"{self.TASK_PREFIX}{task_id}"
        client.set(key, updated_task.model_dump_json())
        return True

    def get_all(self) -> list[ReviewTask]:
        client = self._get_client()
        all_ids = client.lrange(self.QUEUE_KEY, 0, -1)
        tasks = []
        for tid_bytes in all_ids:
            tid = tid_bytes.decode() if isinstance(tid_bytes, bytes) else tid_bytes
            task = self.get_task(tid)
            if task:
                tasks.append(task)
        return tasks

    def get_by_batch(self, batch_id: str) -> list[ReviewTask]:
        return [t for t in self.get_all() if t.batch_id == batch_id]

    def count_by_status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in self.get_all():
            counts[task.status] = counts.get(task.status, 0) + 1
        return counts


# =====================================================================
#  HITL Queue Manager
# =====================================================================

class HITLQueueManager:
    """
    Manages the Human-in-the-Loop review workflow.

    Lifecycle:
      1. enqueue_from_reid()  -- Create tasks from M5 REVIEW dispatches
      2. get_pending()        -- Present tasks to reviewers
      3. submit_decision()    -- Process reviewer action
      4. get_summary()        -- Report queue statistics
      5. export_decisions()   -- Generate hitl_decisions.json
    """

    def __init__(self, backend: QueueBackend) -> None:
        self.backend = backend

    def enqueue_from_reid(
        self,
        reid_data: dict,
    ) -> list[ReviewTask]:
        """
        Create review tasks from M5 REVIEW dispatches.

        Parameters
        ----------
        reid_data : dict
            Parsed reid_result.json (M5 output).

        Returns
        -------
        list[ReviewTask]
            Created review tasks.
        """
        batch_id = reid_data.get("batch_id", "UNKNOWN")
        dispatches = reid_data.get("dispatches", [])

        # Filter to REVIEW status only
        reviews = [d for d in dispatches if d.get("status") == "REVIEW"]

        if not reviews:
            logger.info("No REVIEW dispatches to enqueue for batch %s.", batch_id)
            return []

        now = datetime.now(IST).isoformat()
        created_tasks: list[ReviewTask] = []

        for dispatch in reviews:
            task_id = f"HITL_{uuid.uuid4().hex[:12].upper()}"

            # Extract top match candidate
            top_matches = dispatch.get("top_k_matches", [])
            candidates = [
                MatchCandidate(**m) if isinstance(m, dict) else m
                for m in top_matches
            ]

            candidate_id = dispatch.get("assigned_tiger_id")
            candidate_sim = dispatch.get("confidence", 0.0)

            task = ReviewTask(
                task_id=task_id,
                image_id=dispatch["image_id"],
                crop_path=dispatch.get("crop_path", ""),
                flank_side=dispatch.get("flank_side", "AMBIGUOUS"),
                batch_id=batch_id,
                candidate_tiger_id=candidate_id,
                candidate_similarity=candidate_sim,
                top_k_matches=candidates,
                status="PENDING",
                created_at=now,
            )

            self.backend.push(task)
            created_tasks.append(task)

        logger.info(
            "Enqueued %d review tasks for batch %s.",
            len(created_tasks),
            batch_id,
        )
        return created_tasks

    def get_pending(self, limit: int = 20) -> list[ReviewTask]:
        """Get pending review tasks (FIFO order)."""
        return self.backend.get_pending(limit)

    def get_task(self, task_id: str) -> ReviewTask | None:
        """Get a specific task."""
        return self.backend.get_task(task_id)

    def submit_decision(
        self,
        decision: ReviewDecision,
    ) -> ReviewTask | None:
        """
        Process a reviewer's decision.

        Actions:
          CONFIRM -- Accept candidate match, assign tiger_id
          REJECT  -- Reject match, mark for new identity creation
          MERGE   -- Merge this identity into merge_target_id
          SPLIT   -- Split: this crop is a different tiger

        Returns the updated task, or None if task not found.
        """
        task = self.backend.get_task(decision.task_id)
        if task is None:
            logger.error("Task not found: %s", decision.task_id)
            return None

        if task.status != "PENDING":
            logger.warning(
                "Task %s already resolved (status=%s).",
                decision.task_id,
                task.status,
            )
            return task

        now = datetime.now(IST).isoformat()

        # Determine final status and tiger_id
        status_map = {
            "CONFIRM": "CONFIRMED",
            "REJECT": "REJECTED",
            "MERGE": "MERGED",
            "SPLIT": "SPLIT",
        }

        new_status = status_map.get(decision.action, "PENDING")
        final_id = decision.final_tiger_id

        if decision.action == "CONFIRM":
            final_id = final_id or task.candidate_tiger_id
        elif decision.action == "MERGE":
            final_id = decision.merge_target_id or final_id
        elif decision.action == "REJECT":
            final_id = None  # Will be assigned a new ID by M5/M7

        updates = {
            "status": new_status,
            "reviewed_at": now,
            "reviewer": decision.reviewer,
            "action": decision.action,
            "final_tiger_id": final_id,
            "notes": decision.notes or None,
        }

        success = self.backend.update_task(decision.task_id, updates)
        if not success:
            logger.error("Failed to update task %s.", decision.task_id)
            return None

        updated = self.backend.get_task(decision.task_id)
        logger.info(
            "Decision recorded | task=%s | action=%s | reviewer=%s | tiger=%s",
            decision.task_id,
            decision.action,
            decision.reviewer,
            final_id,
        )
        return updated

    def get_summary(self) -> HITLSummary:
        """Compute queue statistics."""
        counts = self.backend.count_by_status()
        all_tasks = self.backend.get_all()

        # Collect reviewers
        reviewers = set()
        review_times = []

        for task in all_tasks:
            if task.reviewer:
                reviewers.add(task.reviewer)
            if task.reviewed_at and task.created_at:
                try:
                    created = datetime.fromisoformat(task.created_at)
                    reviewed = datetime.fromisoformat(task.reviewed_at)
                    delta = (reviewed - created).total_seconds()
                    if delta >= 0:
                        review_times.append(delta)
                except ValueError:
                    pass

        avg_time = (
            sum(review_times) / len(review_times)
            if review_times
            else None
        )

        return HITLSummary(
            total_tasks=len(all_tasks),
            pending=counts.get("PENDING", 0),
            confirmed=counts.get("CONFIRMED", 0),
            rejected=counts.get("REJECTED", 0),
            merged=counts.get("MERGED", 0),
            split=counts.get("SPLIT", 0),
            expired=counts.get("EXPIRED", 0),
            avg_review_time_seconds=round(avg_time, 2) if avg_time else None,
            reviewers=sorted(reviewers),
        )

    def export_decisions(self, batch_id: str) -> HITLResult:
        """Export all decisions for a batch as HITLResult."""
        tasks = self.backend.get_by_batch(batch_id)
        summary = self.get_summary()

        return HITLResult(
            batch_id=batch_id,
            summary=summary,
            decisions=tasks,
        )


# =====================================================================
#  Backend Factory
# =====================================================================

def create_queue_backend(
    use_redis: bool = False,
    redis_url: str = "redis://localhost:6379/0",
) -> QueueBackend:
    """Create the appropriate queue backend."""
    if use_redis:
        try:
            backend = RedisQueueBackend(redis_url)
            backend._get_client()  # Test connection
            logger.info("Redis queue backend connected: %s", redis_url)
            return backend
        except (ConnectionError, Exception) as exc:
            logger.warning(
                "Redis unavailable (%s). Falling back to in-memory.", exc
            )

    logger.info("Using in-memory queue backend.")
    return InMemoryQueueBackend()


# =====================================================================
#  CLI Entry Point
# =====================================================================

def main() -> None:
    """CLI entry point for the HITL review queue."""
    parser = argparse.ArgumentParser(
        prog="m6_hitl_queue",
        description="TERA-STRIPE M6 -- Human-in-the-Loop Review Queue",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--enqueue", action="store_true",
        help="Enqueue review tasks from a reid_result.json.",
    )
    mode.add_argument(
        "--list-pending", action="store_true",
        help="List pending review tasks.",
    )
    mode.add_argument(
        "--decide", action="store_true",
        help="Submit a reviewer decision.",
    )
    mode.add_argument(
        "--stats", action="store_true",
        help="Show queue statistics.",
    )
    mode.add_argument(
        "--export", action="store_true",
        help="Export decisions for a batch.",
    )

    parser.add_argument("--reid-result", type=Path, default=None)
    parser.add_argument("--task-id", type=str, default=None)
    parser.add_argument(
        "--action", type=str, default=None,
        choices=["CONFIRM", "REJECT", "MERGE", "SPLIT"],
    )
    parser.add_argument("--tiger-id", type=str, default=None)
    parser.add_argument("--merge-target", type=str, default=None)
    parser.add_argument("--reviewer", type=str, default=None)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--batch-id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--use-redis", action="store_true")
    parser.add_argument(
        "--redis-url", type=str, default="redis://localhost:6379/0",
    )
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    backend = create_queue_backend(
        use_redis=args.use_redis, redis_url=args.redis_url
    )
    manager = HITLQueueManager(backend)

    # ── Enqueue ──
    if args.enqueue:
        if not args.reid_result or not args.reid_result.exists():
            logger.error("--reid-result required and must exist.")
            sys.exit(1)
        with open(args.reid_result, "r", encoding="utf-8") as f:
            reid_data = json.load(f)
        tasks = manager.enqueue_from_reid(reid_data)
        print(f"Enqueued {len(tasks)} review tasks.")

    # ── List pending ──
    elif args.list_pending:
        pending = manager.get_pending(args.limit)
        if not pending:
            print("No pending review tasks.")
        else:
            for t in pending:
                print(
                    f"  [{t.task_id}] {t.image_id} | "
                    f"candidate={t.candidate_tiger_id} "
                    f"sim={t.candidate_similarity:.3f} | "
                    f"{t.flank_side}"
                )

    # ── Decide ──
    elif args.decide:
        if not all([args.task_id, args.action, args.reviewer]):
            logger.error("--task-id, --action, and --reviewer required.")
            sys.exit(1)

        decision = ReviewDecision(
            task_id=args.task_id,
            action=args.action,
            reviewer=args.reviewer,
            final_tiger_id=args.tiger_id,
            merge_target_id=args.merge_target,
            notes=args.notes,
        )
        result = manager.submit_decision(decision)
        if result:
            print(
                f"Decision recorded: {result.status} | "
                f"tiger={result.final_tiger_id}"
            )

    # ── Stats ──
    elif args.stats:
        summary = manager.get_summary()
        print(json.dumps(summary.model_dump(), indent=2))

    # ── Export ──
    elif args.export:
        if not args.batch_id:
            logger.error("--batch-id required for export.")
            sys.exit(1)
        result = manager.export_decisions(args.batch_id)
        out = args.output or Path(f"hitl_{args.batch_id}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"Exported to {out}")


if __name__ == "__main__":
    main()
