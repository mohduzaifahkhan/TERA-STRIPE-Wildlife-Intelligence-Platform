"""
TERA-STRIPE Module 9 -- Security Alerts & Anomaly Detection
==============================================================
Detects spatiotemporal anomalies from tiger sighting patterns and
generates structured alerts for ranger dispatch and conservation
management.

Alert Types:
  CORE_RANGE_SHIFT    -- Tiger centroid moved significantly between seasons
  VILLAGE_PROXIMITY   -- Tiger sighted within danger radius of settlement
  NOVEL_STATION       -- Tiger detected at a previously unvisited station
  PROLONGED_ABSENCE   -- No sightings for a tiger beyond expected interval

Severity Levels:
  INFO     -- Informational, no action required
  WARNING  -- Monitor closely, potential concern
  CRITICAL -- Immediate action required

Data contract:
  Input  : Spatial analysis (M8), sighting history (M7)
  Output : alerts.json + security_alerts DB table

CLI Usage
---------
  python -m src.m9_alerts \\
      --db-url sqlite:///tera_stripe.db \\
      --spatial-json ./data/spatial/spatial_analysis.json \\
      --village-radius-km 5.0 \\
      --output ./data/manifests/alerts.json

Reference: Master Context Packet -- Security & Conflict Alerts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("tera_stripe.m9_alerts")

IST = timezone(timedelta(hours=5, minutes=30))


# =====================================================================
#  Constants & Thresholds
# =====================================================================

# Village proximity alert radius (km)
DEFAULT_VILLAGE_RADIUS_KM = 5.0

# Core range shift threshold (km)
DEFAULT_RANGE_SHIFT_KM = 10.0

# Prolonged absence threshold (days)
DEFAULT_ABSENCE_DAYS = 30

# Alert type enum values (must match DB check constraint)
ALERT_TYPES = (
    "CORE_RANGE_SHIFT",
    "VILLAGE_PROXIMITY",
    "NOVEL_STATION",
    "PROLONGED_ABSENCE",
)

SEVERITIES = ("INFO", "WARNING", "CRITICAL")


# =====================================================================
#  Pydantic Contract Models
# =====================================================================

class Alert(BaseModel):
    """A single security/anomaly alert."""
    alert_id: str = Field(default_factory=lambda: f"ALR_{uuid.uuid4().hex[:12].upper()}")
    alert_type: str
    severity: str
    tiger_id: str | None = None
    station_id: str | None = None
    description: str = ""
    distance_to_village_km: float | None = None
    centroid_shift_km: float | None = None
    payload: dict = Field(default_factory=dict)
    is_acknowledged: bool = False
    created_at: str = ""


class AlertSummary(BaseModel):
    """Summary statistics for a batch of alerts."""
    total_alerts: int = 0
    critical: int = 0
    warning: int = 0
    info: int = 0
    by_type: dict[str, int] = Field(default_factory=dict)
    tigers_affected: list[str] = Field(default_factory=list)


class AlertResult(BaseModel):
    """Output contract for alerts.json."""
    batch_id: str = ""
    generated_at: str = ""
    summary: AlertSummary = Field(default_factory=AlertSummary)
    alerts: list[Alert] = []


# =====================================================================
#  Alert Detectors
# =====================================================================

def detect_village_proximity(
    spatial_data: dict,
    radius_km: float = DEFAULT_VILLAGE_RADIUS_KM,
) -> list[Alert]:
    """
    Generate alerts when tiger sightings are within radius of a village.

    Parameters
    ----------
    spatial_data : dict
        A single TigerSpatialAnalysis (from M8).
    radius_km : float
        Alert trigger distance threshold.
    """
    alerts = []
    tiger_id = spatial_data.get("tiger_id", "UNKNOWN")
    proximities = spatial_data.get("village_proximity", [])

    for prox in proximities:
        distance = prox.get("distance_km", 999.0)
        village = prox.get("nearest_village", "Unknown")

        if distance <= radius_km:
            # Determine severity
            if distance <= 1.0:
                severity = "CRITICAL"
            elif distance <= 3.0:
                severity = "WARNING"
            else:
                severity = "INFO"

            alerts.append(Alert(
                alert_type="VILLAGE_PROXIMITY",
                severity=severity,
                tiger_id=tiger_id,
                description=(
                    f"Tiger {tiger_id} detected {distance:.1f} km from "
                    f"village {village}."
                ),
                distance_to_village_km=round(distance, 3),
                payload={
                    "village_name": village,
                    "bearing_deg": prox.get("bearing_deg", 0),
                    "sighting_point": prox.get("sighting_point", []),
                },
                created_at=datetime.now(IST).isoformat(),
            ))

    return alerts


def detect_core_range_shift(
    current_spatial: dict,
    previous_spatial: dict | None = None,
    shift_threshold_km: float = DEFAULT_RANGE_SHIFT_KM,
) -> list[Alert]:
    """
    Detect significant centroid shifts between analysis periods.

    Parameters
    ----------
    current_spatial : dict
        Current period TigerSpatialAnalysis.
    previous_spatial : dict, optional
        Previous period TigerSpatialAnalysis for comparison.
    shift_threshold_km : float
        Minimum shift distance to trigger alert.
    """
    if previous_spatial is None:
        return []

    tiger_id = current_spatial.get("tiger_id", "UNKNOWN")

    curr_centroid = current_spatial.get("mcp", {}).get("centroid", [0, 0])
    prev_centroid = previous_spatial.get("mcp", {}).get("centroid", [0, 0])

    if curr_centroid == [0, 0] or prev_centroid == [0, 0]:
        return []

    from src.m8_spatial import haversine_km
    shift = haversine_km(
        curr_centroid[1], curr_centroid[0],  # lat, lon
        prev_centroid[1], prev_centroid[0],
    )

    if shift < shift_threshold_km:
        return []

    if shift >= 20.0:
        severity = "CRITICAL"
    elif shift >= 15.0:
        severity = "WARNING"
    else:
        severity = "INFO"

    return [Alert(
        alert_type="CORE_RANGE_SHIFT",
        severity=severity,
        tiger_id=tiger_id,
        description=(
            f"Tiger {tiger_id} core range shifted {shift:.1f} km "
            f"from previous period."
        ),
        centroid_shift_km=round(shift, 3),
        payload={
            "current_centroid": curr_centroid,
            "previous_centroid": prev_centroid,
            "current_season": current_spatial.get("season_year", ""),
            "previous_season": previous_spatial.get("season_year", ""),
        },
        created_at=datetime.now(IST).isoformat(),
    )]


def detect_novel_station(
    tiger_id: str,
    current_stations: list[str],
    historical_stations: list[str],
) -> list[Alert]:
    """
    Detect when a tiger appears at a station never visited before.

    Parameters
    ----------
    current_stations : list[str]
        Stations visited in the current batch.
    historical_stations : list[str]
        All stations historically visited by this tiger.
    """
    novel = set(current_stations) - set(historical_stations)
    alerts = []

    for station in sorted(novel):
        alerts.append(Alert(
            alert_type="NOVEL_STATION",
            severity="WARNING",
            tiger_id=tiger_id,
            station_id=station,
            description=(
                f"Tiger {tiger_id} detected at novel station {station} "
                f"(not in historical record of {len(historical_stations)} stations)."
            ),
            payload={
                "novel_station": station,
                "historical_count": len(historical_stations),
                "historical_stations": historical_stations,
            },
            created_at=datetime.now(IST).isoformat(),
        ))

    return alerts


def detect_prolonged_absence(
    tiger_id: str,
    last_seen: datetime | str | None,
    threshold_days: int = DEFAULT_ABSENCE_DAYS,
    reference_time: datetime | None = None,
) -> list[Alert]:
    """
    Detect prolonged absence of a tiger from camera trap network.

    Parameters
    ----------
    last_seen : datetime or ISO string
        Last sighting timestamp.
    threshold_days : int
        Number of days after which absence triggers an alert.
    """
    if last_seen is None:
        return []

    if isinstance(last_seen, str):
        try:
            last_seen = datetime.fromisoformat(last_seen)
        except ValueError:
            return []

    now = reference_time or datetime.now(IST)

    # Ensure both are comparable (strip tzinfo if needed)
    now_naive = now.replace(tzinfo=None) if now.tzinfo else now
    last_naive = last_seen.replace(tzinfo=None) if last_seen.tzinfo else last_seen

    days_absent = (now_naive - last_naive).days

    if days_absent < threshold_days:
        return []

    if days_absent >= 90:
        severity = "CRITICAL"
    elif days_absent >= 60:
        severity = "WARNING"
    else:
        severity = "INFO"

    return [Alert(
        alert_type="PROLONGED_ABSENCE",
        severity=severity,
        tiger_id=tiger_id,
        description=(
            f"Tiger {tiger_id} has not been sighted for {days_absent} days "
            f"(threshold: {threshold_days} days)."
        ),
        payload={
            "days_absent": days_absent,
            "last_seen": last_naive.isoformat(),
            "threshold_days": threshold_days,
        },
        created_at=datetime.now(IST).isoformat(),
    )]


# =====================================================================
#  Alert Engine (Orchestrator)
# =====================================================================

class AlertEngine:
    """
    Orchestrates anomaly detection across all alert types.

    Combines spatial analysis, sighting history, and configuration
    thresholds to produce a unified alert feed.
    """

    def __init__(
        self,
        village_radius_km: float = DEFAULT_VILLAGE_RADIUS_KM,
        range_shift_km: float = DEFAULT_RANGE_SHIFT_KM,
        absence_days: int = DEFAULT_ABSENCE_DAYS,
    ) -> None:
        self.village_radius_km = village_radius_km
        self.range_shift_km = range_shift_km
        self.absence_days = absence_days

    def process_spatial_results(
        self,
        spatial_data: dict,
        previous_spatial: dict | None = None,
        historical_stations: dict[str, list[str]] | None = None,
        last_seen_dates: dict[str, str | datetime] | None = None,
        reference_time: datetime | None = None,
    ) -> AlertResult:
        """
        Run all alert detectors on spatial analysis results.

        Parameters
        ----------
        spatial_data : dict
            SpatialAnalysisResult from M8 (contains multiple tiger analyses).
        previous_spatial : dict, optional
            Previous period SpatialAnalysisResult for range shift detection.
        historical_stations : dict, optional
            {tiger_id: [station_ids]} for novel station detection.
        last_seen_dates : dict, optional
            {tiger_id: datetime/ISO} for absence detection.

        Returns
        -------
        AlertResult
        """
        all_alerts: list[Alert] = []
        analyses = spatial_data.get("analyses", [])

        # Build previous spatial lookup
        prev_lookup: dict[str, dict] = {}
        if previous_spatial:
            for a in previous_spatial.get("analyses", []):
                prev_lookup[a.get("tiger_id", "")] = a

        hist_stations = historical_stations or {}
        last_seen = last_seen_dates or {}

        for analysis in analyses:
            tiger_id = analysis.get("tiger_id", "")

            # 1. Village proximity
            all_alerts.extend(
                detect_village_proximity(analysis, self.village_radius_km)
            )

            # 2. Core range shift
            prev = prev_lookup.get(tiger_id)
            all_alerts.extend(
                detect_core_range_shift(
                    analysis, prev, self.range_shift_km
                )
            )

            # 3. Novel station
            current_stations = [
                c.get("h3_index", "") or ""
                for c in analysis.get("h3_occupancy", {}).get("cells", [])
            ]
            # Also extract station_ids from movement data if available
            movement = analysis.get("movement_stats", {})
            hist = hist_stations.get(tiger_id, [])
            all_alerts.extend(
                detect_novel_station(tiger_id, current_stations, hist)
            )

            # 4. Prolonged absence
            ls = last_seen.get(tiger_id)
            if ls:
                all_alerts.extend(
                    detect_prolonged_absence(
                        tiger_id, ls, self.absence_days, reference_time
                    )
                )

        # Build summary
        summary = self._build_summary(all_alerts)

        now = datetime.now(IST).isoformat()
        return AlertResult(
            batch_id=spatial_data.get("batch_id", ""),
            generated_at=now,
            summary=summary,
            alerts=all_alerts,
        )

    def _build_summary(self, alerts: list[Alert]) -> AlertSummary:
        """Compute alert summary statistics."""
        by_type: dict[str, int] = {}
        tigers = set()
        critical = warning = info = 0

        for a in alerts:
            by_type[a.alert_type] = by_type.get(a.alert_type, 0) + 1
            if a.tiger_id:
                tigers.add(a.tiger_id)
            if a.severity == "CRITICAL":
                critical += 1
            elif a.severity == "WARNING":
                warning += 1
            else:
                info += 1

        return AlertSummary(
            total_alerts=len(alerts),
            critical=critical,
            warning=warning,
            info=info,
            by_type=by_type,
            tigers_affected=sorted(tigers),
        )


# =====================================================================
#  Database Integration
# =====================================================================

def write_alerts_to_db(
    alerts: list[Alert],
    db_url: str,
) -> int:
    """
    Write alerts to the security_alerts database table.

    Returns the number of alerts written.
    """
    from src.m7_database import SecurityAlert, get_engine, get_session_factory, init_db

    engine = get_engine(url=db_url)
    init_db(engine)
    Session = get_session_factory(engine)

    count = 0
    with Session() as session:
        for alert in alerts:
            sa = SecurityAlert(
                alert_id=alert.alert_id,
                alert_type=alert.alert_type,
                severity=alert.severity,
                tiger_id=alert.tiger_id,
                station_id=alert.station_id,
                distance_to_village_km=alert.distance_to_village_km,
                centroid_shift_sq_km=alert.centroid_shift_km,
                alert_payload=json.dumps(alert.payload),
                is_acknowledged=False,
            )
            session.add(sa)
            count += 1
        session.commit()

    logger.info("Wrote %d alerts to database.", count)
    return count


# =====================================================================
#  CLI Entry Point
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="m9_alerts",
        description="TERA-STRIPE M9 -- Security Alerts & Anomaly Detection",
    )

    parser.add_argument(
        "--spatial-json", type=Path, required=True,
        help="Path to spatial_analysis.json (M8 output).",
    )
    parser.add_argument(
        "--previous-json", type=Path, default=None,
        help="Previous period spatial analysis for range shift detection.",
    )
    parser.add_argument(
        "--village-radius-km", type=float,
        default=DEFAULT_VILLAGE_RADIUS_KM,
    )
    parser.add_argument(
        "--range-shift-km", type=float,
        default=DEFAULT_RANGE_SHIFT_KM,
    )
    parser.add_argument(
        "--absence-days", type=int,
        default=DEFAULT_ABSENCE_DAYS,
    )
    parser.add_argument("--db-url", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    # Load spatial data
    with open(args.spatial_json, "r", encoding="utf-8") as f:
        spatial_data = json.load(f)

    previous = None
    if args.previous_json and args.previous_json.exists():
        with open(args.previous_json, "r", encoding="utf-8") as f:
            previous = json.load(f)

    engine = AlertEngine(
        village_radius_km=args.village_radius_km,
        range_shift_km=args.range_shift_km,
        absence_days=args.absence_days,
    )

    result = engine.process_spatial_results(
        spatial_data=spatial_data,
        previous_spatial=previous,
    )

    # Write to DB
    if args.db_url and result.alerts:
        write_alerts_to_db(result.alerts, args.db_url)

    # Write JSON
    out = args.output or Path("alerts.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    # Print summary
    s = result.summary
    print(f"Alerts: {s.total_alerts} total | "
          f"CRITICAL: {s.critical} | WARNING: {s.warning} | INFO: {s.info}")
    for a in result.alerts:
        print(f"  [{a.severity}] {a.alert_type} | {a.description}")


if __name__ == "__main__":
    main()
