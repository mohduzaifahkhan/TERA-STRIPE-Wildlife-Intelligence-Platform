"""
TERA-STRIPE Database Models — SQLAlchemy 2.0 + PostGIS / SQLite Fallback
=========================================================================
Defines all 5 core tables per the Master Context Packet §2.

Tables
------
  camera_stations   — Camera trap deployment locations (Point EPSG:4326)
  tigers            — Individual tiger master catalog
  tiger_sightings   — Camera-trap detection ledger (Point EPSG:4326)
  home_ranges       — Cached territory polygons (MCP / KDE)
  security_alerts   — Spatiotemporal anomaly alert log

PostGIS vs SQLite
-----------------
When ``USE_POSTGIS=true`` and geoalchemy2 is installed, geometry columns use
native PostGIS ``Geometry`` types with GiST indexes.  Otherwise, geometry is
stored as WKT ``Text`` columns (sufficient for unit testing and local dev).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ── Conditional PostGIS import ───────────────────────────────────
_USE_POSTGIS = os.environ.get("USE_POSTGIS", "").lower() in ("1", "true", "yes")
_HAS_GEOALCHEMY2 = False
try:
    from geoalchemy2 import Geometry as _PGGeometry

    _HAS_GEOALCHEMY2 = True
except ImportError:
    pass


def _geom_col(
    geom_type: str = "Point",
    srid: int = 4326,
    nullable: bool = False,
) -> Column:
    """Return a PostGIS Geometry column or a Text fallback for SQLite."""
    if _HAS_GEOALCHEMY2 and _USE_POSTGIS:
        return Column(
            _PGGeometry(geometry_type=geom_type, srid=srid),
            nullable=nullable,
        )
    return Column(Text, nullable=nullable)


def _json_col(nullable: bool = False) -> Column:
    """Return JSONB on PostgreSQL, Text on SQLite."""
    if _USE_POSTGIS:
        return Column(JSONB, nullable=nullable)
    return Column(Text, nullable=nullable)


# ── Base ─────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Table 1: Camera Trap Deployment Stations ─────────────────────
class CameraStation(Base):
    __tablename__ = "camera_stations"

    station_id: str = Column(String(50), primary_key=True)
    zone_type: str | None = Column(
        String(20),
        CheckConstraint("zone_type IN ('CORE','BUFFER','CORRIDOR','FRINGE')"),
        nullable=True,
    )
    range_name: str = Column(String(100), nullable=False)
    geom = _geom_col("Point", 4326, nullable=False)
    elevation_m: float | None = Column(Float, nullable=True)
    h3_res8_index: str | None = Column(String(15), nullable=True)
    h3_res9_index: str | None = Column(String(15), nullable=True)
    is_active: bool = Column(Boolean, default=True)
    installed_at: datetime = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    last_telemetry_at: datetime | None = Column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f"<CameraStation {self.station_id} zone={self.zone_type}>"


# ── Table 2: Individual Tiger Master Catalog ─────────────────────
class Tiger(Base):
    __tablename__ = "tigers"

    tiger_id: str = Column(String(50), primary_key=True)
    common_name: str | None = Column(String(100), nullable=True)
    sex: str | None = Column(
        String(10),
        CheckConstraint("sex IN ('MALE','FEMALE','UNKNOWN')"),
        nullable=True,
    )
    estimated_dob = Column(DateTime, nullable=True)
    status: str = Column(
        String(20),
        CheckConstraint(
            "status IN ('RESIDENT','DISPERSING','TRANSIENT','DEAD','UNKNOWN')"
        ),
        default="RESIDENT",
    )
    first_detected_at: datetime = Column(DateTime(timezone=True), nullable=False)
    last_detected_at: datetime = Column(DateTime(timezone=True), nullable=False)
    profile_notes: str | None = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<Tiger {self.tiger_id} ({self.common_name})>"


# ── Table 3: Camera Trap Sightings Ledger ────────────────────────
class TigerSighting(Base):
    __tablename__ = "tiger_sightings"

    sighting_id: str = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    tiger_id: str | None = Column(
        String(50),
        # ForeignKey deferred — works on both backends
        nullable=True,
    )
    station_id: str | None = Column(String(50), nullable=True)
    captured_at: datetime = Column(DateTime(timezone=True), nullable=False)
    flank_orientation: str | None = Column(
        String(20),
        CheckConstraint(
            "flank_orientation IN "
            "('LEFT_FLANK','RIGHT_FLANK','FRONTAL','REAR','AMBIGUOUS')"
        ),
        nullable=True,
    )
    reid_confidence_score: float = Column(Float, nullable=False)
    verification_status: str = Column(
        String(20),
        CheckConstraint(
            "verification_status IN "
            "('AUTO_COMMITTED','HITL_VERIFIED','PROVISIONAL','FLAGGED')"
        ),
        default="AUTO_COMMITTED",
    )
    verified_by_user: str | None = Column(String(100), nullable=True)
    raw_image_path: str = Column(Text, nullable=False)
    flank_crop_path: str = Column(Text, nullable=False)
    geom = _geom_col("Point", 4326, nullable=False)
    ambient_temp_c: float | None = Column(Float, nullable=True)
    flash_fired: bool | None = Column(Boolean, nullable=True)
    created_at: datetime = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self) -> str:
        return f"<TigerSighting {self.sighting_id[:8]}… tiger={self.tiger_id}>"


# ── Table 4: Dynamic Home Range & Territory Cache ────────────────
class HomeRange(Base):
    __tablename__ = "home_ranges"

    calculation_id: str = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    tiger_id: str = Column(String(50), nullable=False)
    season_year: str = Column(String(20), nullable=False)
    mcp_95_geom = _geom_col("Polygon", 4326, nullable=True)
    mcp_100_geom = _geom_col("Polygon", 4326, nullable=True)
    kde_50_geom = _geom_col("Polygon", 4326, nullable=True)
    kde_95_geom = _geom_col("Polygon", 4326, nullable=True)
    centroid_geom = _geom_col("Point", 4326, nullable=False)
    mcp_area_sq_km: float = Column(Float, nullable=False)
    kde_core_area_sq_km: float | None = Column(Float, nullable=True)
    sighting_count: int = Column(Integer, nullable=False)
    updated_at: datetime = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self) -> str:
        return (
            f"<HomeRange tiger={self.tiger_id} season={self.season_year}>"
        )


# ── Table 5: Security & Conflict Alert Logs ─────────────────────
class SecurityAlert(Base):
    __tablename__ = "security_alerts"

    alert_id: str = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    alert_type: str = Column(
        String(50),
        CheckConstraint(
            "alert_type IN "
            "('CORE_RANGE_SHIFT','VILLAGE_PROXIMITY','NOVEL_STATION','PROLONGED_ABSENCE')"
        ),
        nullable=False,
    )
    severity: str = Column(
        String(20),
        CheckConstraint("severity IN ('INFO','WARNING','CRITICAL')"),
        nullable=False,
    )
    tiger_id: str | None = Column(String(50), nullable=True)
    station_id: str | None = Column(String(50), nullable=True)
    distance_to_village_km: float | None = Column(Float, nullable=True)
    centroid_shift_sq_km: float | None = Column(Float, nullable=True)
    alert_payload = _json_col(nullable=False)
    is_acknowledged: bool = Column(Boolean, default=False)
    acknowledged_by: str | None = Column(String(100), nullable=True)
    created_at: datetime = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self) -> str:
        return f"<SecurityAlert {self.alert_id[:8]}… type={self.alert_type}>"


# ── Engine & Session Factory ─────────────────────────────────────

def get_engine(url: str | None = None, echo: bool = False) -> Engine:
    """
    Create an SQLAlchemy engine.

    Parameters
    ----------
    url : str, optional
        Database URL.  Falls back to ``SQLITE_FALLBACK_URL`` from config
        when PostgreSQL is unreachable.
    echo : bool
        Emit SQL to stdout for debugging.

    Returns
    -------
    Engine
    """
    if url is None:
        # Import here to avoid circular dependency at module level
        from src.config import settings

        if settings.USE_POSTGIS:
            url = settings.DATABASE_URL
        else:
            url = settings.SQLITE_FALLBACK_URL

    engine = create_engine(url, echo=echo)

    # Enable WAL mode for SQLite (better concurrent reads)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn: Any, _rec: Any) -> None:
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.close()

    return engine


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Return a sessionmaker bound to the given (or default) engine."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine | None = None) -> None:
    """Create all tables on the given engine (idempotent)."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)


# ── PostGIS DDL for production deployment ────────────────────────
POSTGIS_DDL = """
-- Run on PostgreSQL 16 + PostGIS 3.4 for full spatial support.
-- These statements add GiST spatial indexes that SQLAlchemy cannot
-- create portably via its metadata API when using the Text fallback.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE INDEX IF NOT EXISTS idx_stations_geom
    ON camera_stations USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_stations_h3
    ON camera_stations(h3_res8_index);

CREATE INDEX IF NOT EXISTS idx_sightings_geom
    ON tiger_sightings USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_sightings_tiger_time
    ON tiger_sightings(tiger_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_sightings_station
    ON tiger_sightings(station_id);

CREATE INDEX IF NOT EXISTS idx_home_range_mcp
    ON home_ranges USING GIST(mcp_95_geom);
CREATE INDEX IF NOT EXISTS idx_home_range_centroid
    ON home_ranges USING GIST(centroid_geom);

CREATE INDEX IF NOT EXISTS idx_alerts_created
    ON security_alerts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity
    ON security_alerts(severity, is_acknowledged);
"""
