"""
TERA-STRIPE Module 7 -- Database Manager & Observation Logger
==============================================================
High-level API that writes pipeline results (M1-M6) into the
SQLAlchemy ORM tables and provides query / export capabilities.

Responsibilities:
  1. Log pipeline observations (triage, flank, re-ID, HITL decisions)
  2. Create / update tiger profiles from Re-ID and HITL results
  3. Register camera stations from ingestion manifests
  4. Query tiger sighting history, station stats, and profiles
  5. Export data to CSV / JSON for NTCA reporting

Data contract:
  Input  : Pipeline JSON artifacts (manifest, triage, flank, reid, hitl)
  Output : Populated database tables + CSV/JSON exports

CLI Usage
---------
  # Ingest all pipeline artifacts for a batch
  python -m src.m7_db_manager \\
      --manifest ./data/manifests/manifest_STN_104B.json \\
      --triage   ./data/manifests/triage_BATCH.json \\
      --flank    ./data/manifests/flank_BATCH.json \\
      --reid     ./data/manifests/reid_BATCH.json \\
      --db-url   sqlite:///tera_stripe.db

  # Query tiger profile
  python -m src.m7_db_manager --query-tiger PTR_M_001

  # Export sightings to CSV
  python -m src.m7_db_manager --export-csv ./data/exports/sightings.csv

Reference: Master Context Packet -- §2 Database, Observation Logging
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.m7_database import (
    Base,
    CameraStation,
    HomeRange,
    SecurityAlert,
    Tiger,
    TigerSighting,
    get_engine,
    get_session_factory,
    init_db,
)

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("tera_stripe.m7_db_manager")

IST = timezone(timedelta(hours=5, minutes=30))


# =====================================================================
#  Pydantic Summary Models
# =====================================================================

class IngestionStats(BaseModel):
    """Statistics from a database ingestion run."""
    batch_id: str = ""
    stations_created: int = 0
    stations_updated: int = 0
    tigers_created: int = 0
    tigers_updated: int = 0
    sightings_created: int = 0
    errors: list[str] = Field(default_factory=list)


class TigerProfile(BaseModel):
    """Public-facing tiger profile summary."""
    tiger_id: str
    common_name: str | None = None
    sex: str | None = None
    status: str = "UNKNOWN"
    first_detected: str | None = None
    last_detected: str | None = None
    total_sightings: int = 0
    stations_visited: list[str] = Field(default_factory=list)
    verification_breakdown: dict[str, int] = Field(default_factory=dict)


class StationStats(BaseModel):
    """Station-level capture statistics."""
    station_id: str
    zone_type: str | None = None
    range_name: str = ""
    total_sightings: int = 0
    unique_tigers: int = 0
    tiger_ids: list[str] = Field(default_factory=list)
    is_active: bool = True


# =====================================================================
#  Database Manager
# =====================================================================

class DatabaseManager:
    """
    High-level database API for the TERA-STRIPE pipeline.

    Manages the lifecycle of writing pipeline results to the database,
    querying tiger profiles and station statistics, and exporting data.
    """

    def __init__(self, db_url: str | None = None) -> None:
        self.engine = get_engine(url=db_url)
        self.SessionFactory = get_session_factory(self.engine)
        init_db(self.engine)

    # -----------------------------------------------------------------
    #  Station Registration (from M1 manifest)
    # -----------------------------------------------------------------

    def register_stations_from_manifest(
        self,
        manifest_data: dict,
    ) -> int:
        """
        Create or update camera stations from an ingest manifest.

        Returns the number of stations registered.
        """
        station = manifest_data.get("station", {})
        station_id = station.get("station_id")
        if not station_id:
            return 0

        with self.SessionFactory() as session:
            existing = session.get(CameraStation, station_id)

            if existing is None:
                cs = CameraStation(
                    station_id=station_id,
                    zone_type=station.get("zone_type"),
                    range_name=station.get("range_name", "UNKNOWN"),
                    geom=station.get("geom_wkt", "POINT(0 0)"),
                    elevation_m=station.get("elevation_m"),
                    h3_res8_index=station.get("h3_res8"),
                    h3_res9_index=station.get("h3_res9"),
                    is_active=True,
                )
                session.add(cs)
                session.commit()
                logger.info("Station registered: %s", station_id)
                return 1
            else:
                # Update last telemetry
                existing.last_telemetry_at = datetime.now(timezone.utc)
                session.commit()
                return 0

    # -----------------------------------------------------------------
    #  Tiger Profile Management (from M5 / M6)
    # -----------------------------------------------------------------

    def create_or_update_tiger(
        self,
        tiger_id: str,
        timestamp: datetime | None = None,
        common_name: str | None = None,
        sex: str | None = None,
        status: str = "RESIDENT",
        notes: str | None = None,
    ) -> Tiger:
        """Create a new tiger profile or update timestamps."""
        now = timestamp or datetime.now(timezone.utc)

        with self.SessionFactory() as session:
            existing = session.get(Tiger, tiger_id)

            if existing is None:
                tiger = Tiger(
                    tiger_id=tiger_id,
                    common_name=common_name,
                    sex=sex,
                    status=status,
                    first_detected_at=now,
                    last_detected_at=now,
                    profile_notes=notes,
                )
                session.add(tiger)
                session.commit()
                session.refresh(tiger)
                logger.info("Tiger profile created: %s", tiger_id)
                return tiger
            else:
                # SQLite strips tzinfo; normalise both to naive UTC for compare
                last = existing.last_detected_at
                now_cmp = now.replace(tzinfo=None) if now.tzinfo else now
                last_cmp = last.replace(tzinfo=None) if last and last.tzinfo else last
                if last_cmp is None or now_cmp > last_cmp:
                    existing.last_detected_at = now
                if common_name:
                    existing.common_name = common_name
                if sex and existing.sex == "UNKNOWN":
                    existing.sex = sex
                if notes:
                    existing.profile_notes = notes
                session.commit()
                session.refresh(existing)
                return existing

    # -----------------------------------------------------------------
    #  Sighting Logging (from M4/M5 pipeline)
    # -----------------------------------------------------------------

    def log_sighting(
        self,
        tiger_id: str,
        station_id: str,
        captured_at: datetime,
        flank_orientation: str,
        reid_confidence: float,
        verification_status: str,
        raw_image_path: str,
        flank_crop_path: str,
        geom_wkt: str = "POINT(0 0)",
        ambient_temp_c: float | None = None,
        flash_fired: bool | None = None,
        verified_by: str | None = None,
    ) -> str:
        """
        Log a single tiger sighting to the database.

        Returns the sighting_id.
        """
        sighting_id = str(uuid.uuid4())

        with self.SessionFactory() as session:
            sighting = TigerSighting(
                sighting_id=sighting_id,
                tiger_id=tiger_id,
                station_id=station_id,
                captured_at=captured_at,
                flank_orientation=flank_orientation,
                reid_confidence_score=reid_confidence,
                verification_status=verification_status,
                verified_by_user=verified_by,
                raw_image_path=raw_image_path,
                flank_crop_path=flank_crop_path,
                geom=geom_wkt,
                ambient_temp_c=ambient_temp_c,
                flash_fired=flash_fired,
            )
            session.add(sighting)
            session.commit()

        return sighting_id

    # -----------------------------------------------------------------
    #  Batch Pipeline Ingestion
    # -----------------------------------------------------------------

    def ingest_pipeline_results(
        self,
        manifest_data: dict | None = None,
        reid_data: dict | None = None,
        hitl_data: dict | None = None,
        flank_data: dict | None = None,
    ) -> IngestionStats:
        """
        Ingest all pipeline artifacts for a batch into the database.

        Processes in order:
          1. Stations from manifest
          2. Tiger profiles from Re-ID NEW_INDIVIDUAL + AUTO_MATCH
          3. Sightings from Re-ID dispatches
          4. HITL decision updates
        """
        stats = IngestionStats()

        # 1. Register station from manifest
        if manifest_data:
            stats.batch_id = manifest_data.get("batch_id", "")
            try:
                n = self.register_stations_from_manifest(manifest_data)
                stats.stations_created = n
                stats.stations_updated = 0 if n else 1
            except Exception as exc:
                stats.errors.append(f"Station registration: {exc}")

        # 2. Process Re-ID results
        if reid_data:
            stats.batch_id = stats.batch_id or reid_data.get("batch_id", "")
            dispatches = reid_data.get("dispatches", [])

            # Build image metadata lookup from manifest
            img_meta = {}
            if manifest_data:
                for rec in manifest_data.get("records", []):
                    img_meta[rec.get("image_id")] = rec

            for dispatch in dispatches:
                tiger_id = dispatch.get("assigned_tiger_id")
                if not tiger_id:
                    continue

                status = dispatch.get("status", "")
                image_id = dispatch.get("image_id", "")
                crop_path = dispatch.get("crop_path", "")
                flank_side = dispatch.get("flank_side", "AMBIGUOUS")
                confidence = dispatch.get("confidence", 0.0)

                # Map flank_side to database enum
                orientation_map = {
                    "LEFT": "LEFT_FLANK",
                    "RIGHT": "RIGHT_FLANK",
                    "AMBIGUOUS": "AMBIGUOUS",
                }
                flank_orientation = orientation_map.get(
                    flank_side, "AMBIGUOUS"
                )

                # Map Re-ID status to verification status
                verification_map = {
                    "AUTO_MATCH": "AUTO_COMMITTED",
                    "REVIEW": "PROVISIONAL",
                    "NEW_INDIVIDUAL": "AUTO_COMMITTED",
                }
                verification_status = verification_map.get(
                    status, "PROVISIONAL"
                )

                # Get image metadata
                meta = img_meta.get(image_id, {})
                station_id = meta.get("station_id", "UNKNOWN")
                captured_str = meta.get("exif", {}).get(
                    "datetime_original", ""
                )
                raw_path = meta.get("absolute_path", "")
                geom_wkt = meta.get("geom_wkt", "POINT(0 0)")
                temp = meta.get("exif", {}).get("ambient_temp_c")
                flash = meta.get("exif", {}).get("flash_fired")

                try:
                    captured_at = (
                        datetime.fromisoformat(captured_str)
                        if captured_str
                        else datetime.now(timezone.utc)
                    )
                except ValueError:
                    captured_at = datetime.now(timezone.utc)

                try:
                    # Create/update tiger profile
                    self.create_or_update_tiger(
                        tiger_id=tiger_id,
                        timestamp=captured_at,
                    )
                    if status == "NEW_INDIVIDUAL":
                        stats.tigers_created += 1
                    else:
                        stats.tigers_updated += 1

                    # Log sighting
                    self.log_sighting(
                        tiger_id=tiger_id,
                        station_id=station_id,
                        captured_at=captured_at,
                        flank_orientation=flank_orientation,
                        reid_confidence=confidence,
                        verification_status=verification_status,
                        raw_image_path=raw_path or crop_path,
                        flank_crop_path=crop_path,
                        geom_wkt=geom_wkt,
                        ambient_temp_c=temp,
                        flash_fired=flash,
                    )
                    stats.sightings_created += 1

                except Exception as exc:
                    stats.errors.append(
                        f"Sighting {image_id}: {exc}"
                    )

        # 3. Apply HITL decisions
        if hitl_data:
            decisions = hitl_data.get("decisions", [])
            for d in decisions:
                if d.get("status") in ("CONFIRMED", "MERGED"):
                    final_id = d.get("final_tiger_id")
                    reviewer = d.get("reviewer")
                    if final_id:
                        self._apply_hitl_decision(
                            d.get("image_id", ""),
                            final_id,
                            reviewer,
                        )

        logger.info(
            "Pipeline ingestion | batch=%s | stations=%d | "
            "tigers=+%d/%d updated | sightings=%d | errors=%d",
            stats.batch_id,
            stats.stations_created,
            stats.tigers_created,
            stats.tigers_updated,
            stats.sightings_created,
            len(stats.errors),
        )

        return stats

    def _apply_hitl_decision(
        self,
        image_id: str,
        final_tiger_id: str,
        reviewer: str | None,
    ) -> None:
        """Update sighting verification after HITL review."""
        with self.SessionFactory() as session:
            # Find the sighting by image path pattern
            sightings = session.query(TigerSighting).filter(
                TigerSighting.raw_image_path.contains(image_id)
            ).all()

            for sighting in sightings:
                sighting.tiger_id = final_tiger_id
                sighting.verification_status = "HITL_VERIFIED"
                sighting.verified_by_user = reviewer

            session.commit()

    # -----------------------------------------------------------------
    #  Query APIs
    # -----------------------------------------------------------------

    def get_tiger_profile(self, tiger_id: str) -> TigerProfile | None:
        """Retrieve a full tiger profile with sighting statistics."""
        with self.SessionFactory() as session:
            tiger = session.get(Tiger, tiger_id)
            if tiger is None:
                return None

            sightings = session.query(TigerSighting).filter(
                TigerSighting.tiger_id == tiger_id
            ).all()

            stations = list(set(
                s.station_id for s in sightings if s.station_id
            ))

            verification = {}
            for s in sightings:
                vs = s.verification_status
                verification[vs] = verification.get(vs, 0) + 1

            return TigerProfile(
                tiger_id=tiger.tiger_id,
                common_name=tiger.common_name,
                sex=tiger.sex,
                status=tiger.status,
                first_detected=tiger.first_detected_at.isoformat() if tiger.first_detected_at else None,
                last_detected=tiger.last_detected_at.isoformat() if tiger.last_detected_at else None,
                total_sightings=len(sightings),
                stations_visited=stations,
                verification_breakdown=verification,
            )

    def get_station_stats(self, station_id: str) -> StationStats | None:
        """Get capture statistics for a camera station."""
        with self.SessionFactory() as session:
            station = session.get(CameraStation, station_id)
            if station is None:
                return None

            sightings = session.query(TigerSighting).filter(
                TigerSighting.station_id == station_id
            ).all()

            tiger_ids = list(set(
                s.tiger_id for s in sightings if s.tiger_id
            ))

            return StationStats(
                station_id=station.station_id,
                zone_type=station.zone_type,
                range_name=station.range_name,
                total_sightings=len(sightings),
                unique_tigers=len(tiger_ids),
                tiger_ids=tiger_ids,
                is_active=station.is_active,
            )

    def get_all_tigers(self) -> list[TigerProfile]:
        """List all tiger profiles."""
        with self.SessionFactory() as session:
            tigers = session.query(Tiger).all()
            return [
                TigerProfile(
                    tiger_id=t.tiger_id,
                    common_name=t.common_name,
                    sex=t.sex,
                    status=t.status,
                    first_detected=t.first_detected_at.isoformat() if t.first_detected_at else None,
                    last_detected=t.last_detected_at.isoformat() if t.last_detected_at else None,
                )
                for t in tigers
            ]

    def get_recent_sightings(self, limit: int = 50) -> list[dict]:
        """Get the most recent sightings."""
        with self.SessionFactory() as session:
            sightings = (
                session.query(TigerSighting)
                .order_by(TigerSighting.captured_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "sighting_id": s.sighting_id,
                    "tiger_id": s.tiger_id,
                    "station_id": s.station_id,
                    "captured_at": s.captured_at.isoformat() if s.captured_at else "",
                    "flank_orientation": s.flank_orientation,
                    "confidence": s.reid_confidence_score,
                    "verification": s.verification_status,
                }
                for s in sightings
            ]

    # -----------------------------------------------------------------
    #  Export
    # -----------------------------------------------------------------

    def export_sightings_csv(self, output_path: Path) -> int:
        """Export all sightings to CSV. Returns row count."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with self.SessionFactory() as session:
            sightings = session.query(TigerSighting).order_by(
                TigerSighting.captured_at.desc()
            ).all()

            fieldnames = [
                "sighting_id", "tiger_id", "station_id",
                "captured_at", "flank_orientation",
                "reid_confidence_score", "verification_status",
                "verified_by_user", "raw_image_path",
                "flank_crop_path", "ambient_temp_c", "flash_fired",
            ]

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for s in sightings:
                    writer.writerow({
                        "sighting_id": s.sighting_id,
                        "tiger_id": s.tiger_id,
                        "station_id": s.station_id,
                        "captured_at": s.captured_at.isoformat() if s.captured_at else "",
                        "flank_orientation": s.flank_orientation,
                        "reid_confidence_score": s.reid_confidence_score,
                        "verification_status": s.verification_status,
                        "verified_by_user": s.verified_by_user or "",
                        "raw_image_path": s.raw_image_path,
                        "flank_crop_path": s.flank_crop_path,
                        "ambient_temp_c": s.ambient_temp_c or "",
                        "flash_fired": s.flash_fired or "",
                    })

            logger.info("Exported %d sightings to %s", len(sightings), output_path)
            return len(sightings)

    def export_tigers_json(self, output_path: Path) -> int:
        """Export all tiger profiles to JSON. Returns count."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profiles = self.get_all_tigers()
        data = [p.model_dump() for p in profiles]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Exported %d tiger profiles to %s", len(data), output_path)
        return len(data)

    # -----------------------------------------------------------------
    #  Aggregate Statistics
    # -----------------------------------------------------------------

    def get_database_summary(self) -> dict:
        """Return high-level database statistics."""
        with self.SessionFactory() as session:
            n_stations = session.query(CameraStation).count()
            n_tigers = session.query(Tiger).count()
            n_sightings = session.query(TigerSighting).count()
            n_alerts = session.query(SecurityAlert).count()

            return {
                "camera_stations": n_stations,
                "tigers": n_tigers,
                "sightings": n_sightings,
                "security_alerts": n_alerts,
            }


# =====================================================================
#  CLI Entry Point
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="m7_db_manager",
        description="TERA-STRIPE M7 -- Database Manager & Observation Logger",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ingest", action="store_true", help="Ingest pipeline artifacts.")
    mode.add_argument("--query-tiger", type=str, default=None, help="Query tiger profile.")
    mode.add_argument("--query-station", type=str, default=None, help="Query station stats.")
    mode.add_argument("--export-csv", type=Path, default=None, help="Export sightings CSV.")
    mode.add_argument("--export-tigers", type=Path, default=None, help="Export tiger profiles JSON.")
    mode.add_argument("--summary", action="store_true", help="Show database summary.")

    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--triage", type=Path, default=None)
    parser.add_argument("--flank", type=Path, default=None)
    parser.add_argument("--reid", type=Path, default=None)
    parser.add_argument("--hitl", type=Path, default=None)
    parser.add_argument("--db-url", type=str, default=None)

    args = parser.parse_args()

    db = DatabaseManager(db_url=args.db_url)

    if args.ingest:
        manifest_data = _load_json(args.manifest) if args.manifest else None
        reid_data = _load_json(args.reid) if args.reid else None
        hitl_data = _load_json(args.hitl) if args.hitl else None
        flank_data = _load_json(args.flank) if args.flank else None

        stats = db.ingest_pipeline_results(
            manifest_data=manifest_data,
            reid_data=reid_data,
            hitl_data=hitl_data,
            flank_data=flank_data,
        )
        print(json.dumps(stats.model_dump(), indent=2))

    elif args.query_tiger:
        profile = db.get_tiger_profile(args.query_tiger)
        if profile:
            print(json.dumps(profile.model_dump(), indent=2))
        else:
            print(f"Tiger not found: {args.query_tiger}")

    elif args.query_station:
        stats = db.get_station_stats(args.query_station)
        if stats:
            print(json.dumps(stats.model_dump(), indent=2))
        else:
            print(f"Station not found: {args.query_station}")

    elif args.export_csv:
        n = db.export_sightings_csv(args.export_csv)
        print(f"Exported {n} sightings to {args.export_csv}")

    elif args.export_tigers:
        n = db.export_tigers_json(args.export_tigers)
        print(f"Exported {n} tiger profiles to {args.export_tigers}")

    elif args.summary:
        summary = db.get_database_summary()
        print(json.dumps(summary, indent=2))


def _load_json(path: Path) -> dict | None:
    if path and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    main()
