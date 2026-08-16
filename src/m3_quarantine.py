"""
TERA-STRIPE Module 3 -- Staged Quarantine & ROI Telemetry
===========================================================
Moves blank/empty frames identified by M2 into a 30-day soft-retention
quarantine bucket, maintains a ledger for rollback, and computes
operational ROI telemetry (storage saved, manual hours saved).

Tier Model
----------
  Tier-0 : Active Working Set  (fauna-bearing images retained for CV pipeline)
  Tier-1 : Quarantine Bucket   (blanks held for 30 days, then eligible for purge)

Data contract:
  Input  : triage_result.json  (from M2)
  Output : quarantine_ledger.json

CLI Usage
---------
  # Quarantine blank frames
  python -m src.m3_quarantine \\
      --triage-result ./data/manifests/triage_STN_104B.json \\
      --quarantine-dir ./data/quarantine

  # Rollback a batch (restore files to original paths)
  python -m src.m3_quarantine \\
      --rollback --batch-id BATCH_20260816_PENCH_STN_104B \\
      --quarantine-dir ./data/quarantine

  # Purge expired quarantine (retention > 30 days)
  python -m src.m3_quarantine \\
      --purge-expired \\
      --quarantine-dir ./data/quarantine

Reference: Master Context Packet -- Tier-0/Tier-1, ROI Telemetry
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
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
logger = logging.getLogger("tera_stripe.m3_quarantine")

# ── IST timezone ─────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))

# ── Constants ────────────────────────────────────────────────────
DEFAULT_RETENTION_DAYS = 30
MANUAL_SECONDS_PER_IMAGE = 4.5  # NTCA statutory average


# =====================================================================
#  Pydantic Models -- quarantine_ledger.json
# =====================================================================

class QuarantineEntry(BaseModel):
    """A single quarantined file record."""
    image_id: str
    original_path: str
    quarantine_path: str
    batch_id: str
    file_size_bytes: int
    quarantined_at: str  # ISO-8601
    retention_expires_at: str  # ISO-8601
    status: Literal["QUARANTINED", "RESTORED", "PURGED"] = "QUARANTINED"


class ROISummary(BaseModel):
    """Operational return-on-investment telemetry."""
    total_quarantined: int
    total_restored: int = 0
    total_purged: int = 0
    storage_saved_bytes: int
    storage_saved_gb: float
    manual_hours_saved: float
    retention_days: int = DEFAULT_RETENTION_DAYS


class QuarantineLedger(BaseModel):
    """Per-batch quarantine tracking ledger."""
    batch_id: str
    quarantine_dir: str
    created_at: str
    entries: list[QuarantineEntry]
    roi_summary: ROISummary


# =====================================================================
#  Quarantine Manager
# =====================================================================

class QuarantineManager:
    """
    Manages the Tier-1 quarantine lifecycle:

    1. **quarantine_batch()** -- Move blank frames from Tier-0 to Tier-1
    2. **rollback_batch()**   -- Restore quarantined files to original paths
    3. **purge_expired()**    -- Permanently delete files past retention
    4. **get_batch_status()**  -- Query current state of a quarantined batch
    """

    def __init__(
        self,
        quarantine_dir: Path,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> None:
        self.quarantine_dir = Path(quarantine_dir)
        self.retention_days = retention_days
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

    def _ledger_path(self, batch_id: str) -> Path:
        """Path to the ledger JSON for a given batch."""
        return self.quarantine_dir / f"ledger_{batch_id}.json"

    def _load_ledger(self, batch_id: str) -> QuarantineLedger | None:
        """Load an existing ledger from disk."""
        path = self._ledger_path(batch_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return QuarantineLedger(**json.load(f))

    def _save_ledger(self, ledger: QuarantineLedger) -> None:
        """Persist ledger to disk."""
        path = self._ledger_path(ledger.batch_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ledger.model_dump(), f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------
    #  Quarantine Operation
    # -----------------------------------------------------------------

    def quarantine_batch(
        self,
        triage_data: dict,
        copy_mode: bool = False,
    ) -> QuarantineLedger:
        """
        Move (or copy) blank frames from their original paths into the
        quarantine directory.

        Parameters
        ----------
        triage_data : dict
            Parsed triage_result.json (M2 output).
        copy_mode : bool
            If True, copy files instead of moving (useful for testing
            when you want to preserve originals).

        Returns
        -------
        QuarantineLedger
        """
        batch_id = triage_data["batch_id"]
        dispatches = triage_data.get("dispatches", [])

        # Filter to QUARANTINED_BLANK frames
        blank_dispatches = [
            d for d in dispatches if d["status"] == "QUARANTINED_BLANK"
        ]

        if not blank_dispatches:
            logger.info("No blank frames to quarantine in batch %s.", batch_id)

        # Create batch subdirectory
        batch_dir = self.quarantine_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(IST)
        expires_at = now + timedelta(days=self.retention_days)
        now_iso = now.isoformat()
        expires_iso = expires_at.isoformat()

        entries: list[QuarantineEntry] = []
        total_size = 0
        moved_count = 0

        for dispatch in blank_dispatches:
            image_id = dispatch["image_id"]

            # Resolve original path -- check triage dispatch or infer
            original_path = dispatch.get("target_quarantine_path")
            if not original_path:
                # If no quarantine path was set by M2, we need the
                # original path from the triage records context.
                # This is a fallback -- M2 should always set this.
                logger.warning(
                    "No quarantine path for %s, skipping.", image_id
                )
                continue

            # The triage result stores the *target* quarantine path.
            # We need to find the *source* (original) path.
            # Look in the triage data for additional context, or
            # check if the target path contains the image filename.
            source_path = self._resolve_source_path(
                dispatch, triage_data
            )
            if source_path is None:
                logger.warning(
                    "Cannot resolve source for %s, skipping.", image_id
                )
                continue

            source = Path(source_path)
            if not source.exists():
                logger.warning(
                    "Source file not found: %s, skipping.", source
                )
                continue

            # Compute destination inside quarantine
            dest = batch_dir / source.name
            file_size = source.stat().st_size

            try:
                if copy_mode:
                    shutil.copy2(str(source), str(dest))
                else:
                    shutil.move(str(source), str(dest))
                moved_count += 1
                total_size += file_size

                entries.append(
                    QuarantineEntry(
                        image_id=image_id,
                        original_path=str(source),
                        quarantine_path=str(dest),
                        batch_id=batch_id,
                        file_size_bytes=file_size,
                        quarantined_at=now_iso,
                        retention_expires_at=expires_iso,
                        status="QUARANTINED",
                    )
                )
            except (OSError, shutil.Error) as exc:
                logger.error(
                    "Failed to quarantine %s: %s", source.name, exc
                )

        # Build ROI summary
        storage_gb = round(total_size / (1024 ** 3), 6)
        hours_saved = round(
            len(entries) * MANUAL_SECONDS_PER_IMAGE / 3600.0, 4
        )

        roi = ROISummary(
            total_quarantined=len(entries),
            storage_saved_bytes=total_size,
            storage_saved_gb=storage_gb,
            manual_hours_saved=hours_saved,
            retention_days=self.retention_days,
        )

        ledger = QuarantineLedger(
            batch_id=batch_id,
            quarantine_dir=str(self.quarantine_dir),
            created_at=now_iso,
            entries=entries,
            roi_summary=roi,
        )

        # Persist ledger
        self._save_ledger(ledger)

        logger.info(
            "Quarantine complete | batch=%s | moved=%d | "
            "saved=%.4f GB / %.2f hrs",
            batch_id,
            moved_count,
            storage_gb,
            hours_saved,
        )

        return ledger

    def _resolve_source_path(
        self,
        dispatch: dict,
        triage_data: dict,
    ) -> str | None:
        """
        Resolve the original source file path for a blank dispatch.

        Strategy:
          1. Check if the original manifest records are embedded
          2. Use the quarantine target path basename to find the source
          3. Fallback to None
        """
        image_id = dispatch["image_id"]

        # If the triage data has embedded manifest records with
        # absolute_path, use those.
        records = triage_data.get("_source_records", [])
        for rec in records:
            if rec.get("image_id") == image_id:
                return rec.get("absolute_path")

        # Try to reconstruct from quarantine target path
        # The target quarantine path was set by M2 as:
        #   quarantine_dir / batch_id / image_id.ext
        # The source is the original image in the raw_camera_traps dir.
        # We can look at what the batch source directory was.
        source_dir = triage_data.get("_source_directory")
        if source_dir:
            target_path = dispatch.get("target_quarantine_path", "")
            basename = Path(target_path).name if target_path else ""
            if basename:
                candidate = Path(source_dir) / basename
                if candidate.exists():
                    return str(candidate)

        return None

    # -----------------------------------------------------------------
    #  Rollback Operation
    # -----------------------------------------------------------------

    def rollback_batch(self, batch_id: str) -> int:
        """
        Restore all quarantined files in a batch to their original paths.

        Parameters
        ----------
        batch_id : str
            Batch identifier to rollback.

        Returns
        -------
        int
            Number of files successfully restored.
        """
        ledger = self._load_ledger(batch_id)
        if ledger is None:
            logger.error("No ledger found for batch %s.", batch_id)
            return 0

        restored_count = 0
        for entry in ledger.entries:
            if entry.status != "QUARANTINED":
                continue

            q_path = Path(entry.quarantine_path)
            orig_path = Path(entry.original_path)

            if not q_path.exists():
                logger.warning(
                    "Quarantined file missing: %s, skipping.", q_path
                )
                continue

            try:
                orig_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(q_path), str(orig_path))
                entry.status = "RESTORED"
                restored_count += 1
            except (OSError, shutil.Error) as exc:
                logger.error("Rollback failed for %s: %s", q_path.name, exc)

        # Update ledger
        ledger.roi_summary.total_restored = sum(
            1 for e in ledger.entries if e.status == "RESTORED"
        )
        self._save_ledger(ledger)

        # Clean up empty batch directory
        batch_dir = self.quarantine_dir / batch_id
        if batch_dir.is_dir() and not any(batch_dir.iterdir()):
            batch_dir.rmdir()

        logger.info(
            "Rollback complete | batch=%s | restored=%d",
            batch_id,
            restored_count,
        )
        return restored_count

    # -----------------------------------------------------------------
    #  Purge Expired
    # -----------------------------------------------------------------

    def purge_expired(self) -> int:
        """
        Permanently delete quarantined files past the retention window.

        Returns
        -------
        int
            Number of files purged.
        """
        now = datetime.now(IST)
        purged_total = 0

        for ledger_file in self.quarantine_dir.glob("ledger_*.json"):
            with open(ledger_file, "r", encoding="utf-8") as f:
                ledger = QuarantineLedger(**json.load(f))

            purged_in_batch = 0
            for entry in ledger.entries:
                if entry.status != "QUARANTINED":
                    continue

                try:
                    expires = datetime.fromisoformat(
                        entry.retention_expires_at
                    )
                except ValueError:
                    continue

                if now >= expires:
                    q_path = Path(entry.quarantine_path)
                    if q_path.exists():
                        q_path.unlink()
                        purged_in_batch += 1
                    entry.status = "PURGED"

            if purged_in_batch > 0:
                ledger.roi_summary.total_purged = sum(
                    1 for e in ledger.entries if e.status == "PURGED"
                )
                self._save_ledger(ledger)
                purged_total += purged_in_batch
                logger.info(
                    "Purged %d expired files from batch %s.",
                    purged_in_batch,
                    ledger.batch_id,
                )

        if purged_total == 0:
            logger.info("No expired quarantine files to purge.")

        return purged_total

    # -----------------------------------------------------------------
    #  Status Query
    # -----------------------------------------------------------------

    def get_batch_status(self, batch_id: str) -> dict | None:
        """Return current status of a quarantined batch."""
        ledger = self._load_ledger(batch_id)
        if ledger is None:
            return None

        active = sum(1 for e in ledger.entries if e.status == "QUARANTINED")
        restored = sum(1 for e in ledger.entries if e.status == "RESTORED")
        purged = sum(1 for e in ledger.entries if e.status == "PURGED")

        return {
            "batch_id": batch_id,
            "total_entries": len(ledger.entries),
            "active_quarantined": active,
            "restored": restored,
            "purged": purged,
            "roi_summary": ledger.roi_summary.model_dump(),
        }

    def list_batches(self) -> list[str]:
        """List all batch IDs with existing ledgers."""
        return [
            p.stem.replace("ledger_", "")
            for p in self.quarantine_dir.glob("ledger_*.json")
        ]


# =====================================================================
#  Convenience: Quarantine from triage + manifest pair
# =====================================================================

def quarantine_from_triage_and_manifest(
    triage_path: Path,
    manifest_path: Path | None,
    quarantine_dir: Path,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    copy_mode: bool = False,
) -> QuarantineLedger:
    """
    Load triage result, enrich with manifest source paths, and execute
    quarantine.

    Parameters
    ----------
    triage_path : Path
        triage_result.json from M2.
    manifest_path : Path, optional
        ingest_manifest.json from M1 (for source path resolution).
    quarantine_dir : Path
        Target quarantine directory.
    retention_days : int
        Soft retention period in days.
    copy_mode : bool
        Copy instead of move (preserves originals).

    Returns
    -------
    QuarantineLedger
    """
    with open(triage_path, "r", encoding="utf-8") as f:
        triage_data = json.load(f)

    # Enrich triage data with source paths from manifest
    if manifest_path and Path(manifest_path).exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        triage_data["_source_records"] = manifest_data.get("records", [])
        triage_data["_source_directory"] = manifest_data.get(
            "source_directory", ""
        )

    manager = QuarantineManager(
        quarantine_dir=quarantine_dir,
        retention_days=retention_days,
    )
    return manager.quarantine_batch(triage_data, copy_mode=copy_mode)


# =====================================================================
#  CLI Entry Point
# =====================================================================

def main() -> None:
    """CLI entry point for the quarantine manager."""
    parser = argparse.ArgumentParser(
        prog="m3_quarantine",
        description="TERA-STRIPE M3 -- Staged Quarantine & ROI Telemetry",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--triage-result",
        type=Path,
        default=None,
        help="Path to triage_result.json to quarantine blank frames.",
    )
    mode_group.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback (restore) quarantined files for --batch-id.",
    )
    mode_group.add_argument(
        "--purge-expired",
        action="store_true",
        help="Permanently delete files past retention window.",
    )
    mode_group.add_argument(
        "--status",
        action="store_true",
        help="Show status of a quarantined batch (requires --batch-id).",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to ingest_manifest.json (for source path resolution).",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        required=True,
        help="Quarantine storage directory.",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Batch ID (required for --rollback and --status).",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help=f"Soft retention period in days (default: {DEFAULT_RETENTION_DAYS}).",
    )
    parser.add_argument(
        "--copy-mode",
        action="store_true",
        help="Copy files instead of moving (preserves originals).",
    )

    args = parser.parse_args()

    manager = QuarantineManager(
        quarantine_dir=args.quarantine_dir,
        retention_days=args.retention_days,
    )

    # ── Quarantine mode ──
    if args.triage_result:
        if not args.triage_result.exists():
            logger.error("Triage result not found: %s", args.triage_result)
            sys.exit(1)

        ledger = quarantine_from_triage_and_manifest(
            triage_path=args.triage_result,
            manifest_path=args.manifest,
            quarantine_dir=args.quarantine_dir,
            retention_days=args.retention_days,
            copy_mode=args.copy_mode,
        )
        s = ledger.roi_summary
        print(
            f"\n{'='*60}\n"
            f"  TERA-STRIPE M3 Quarantine Complete\n"
            f"  Batch       : {ledger.batch_id}\n"
            f"  Quarantined : {s.total_quarantined}\n"
            f"  Storage Save: {s.storage_saved_gb:.4f} GB\n"
            f"  Hours Saved : {s.manual_hours_saved:.2f} hrs\n"
            f"  Retention   : {s.retention_days} days\n"
            f"  Ledger      : {manager._ledger_path(ledger.batch_id)}\n"
            f"{'='*60}"
        )

    # ── Rollback mode ──
    elif args.rollback:
        if not args.batch_id:
            logger.error("--batch-id required for rollback.")
            sys.exit(1)
        restored = manager.rollback_batch(args.batch_id)
        print(
            f"\n{'='*60}\n"
            f"  TERA-STRIPE M3 Rollback Complete\n"
            f"  Batch    : {args.batch_id}\n"
            f"  Restored : {restored}\n"
            f"{'='*60}"
        )

    # ── Purge mode ──
    elif args.purge_expired:
        purged = manager.purge_expired()
        print(
            f"\n{'='*60}\n"
            f"  TERA-STRIPE M3 Purge Complete\n"
            f"  Purged   : {purged}\n"
            f"{'='*60}"
        )

    # ── Status mode ──
    elif args.status:
        if not args.batch_id:
            logger.error("--batch-id required for status.")
            sys.exit(1)
        status = manager.get_batch_status(args.batch_id)
        if status is None:
            print(f"No ledger found for batch: {args.batch_id}")
        else:
            print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
