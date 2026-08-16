"""
TERA-STRIPE Configuration — Pydantic v2 BaseSettings
=====================================================
Central configuration for the Wildlife Intelligence Platform.
All fields are overridable via environment variables or .env file.

Reference: Master Context Packet §8.3
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """
    Pydantic v2 settings for the entire TERA-STRIPE platform.

    Load order:
      1. Constructor kwargs
      2. Environment variables
      3. .env file (field defaults last)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Environment ──────────────────────────────────────────────
    ENVIRONMENT: Literal["development", "testing", "production"] = "development"
    PROJECT_NAME: str = "TERA-STRIPE Wildlife Intelligence Platform"

    # ── Storage Paths (derived in model_post_init) ───────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path | None = None
    RAW_CAMERA_TRAPS_DIR: Path | None = None
    QUARANTINE_DIR: Path | None = None
    ACTIVE_WORKING_SET_DIR: Path | None = None
    VECTOR_STORE_DIR: Path | None = None
    MANIFESTS_DIR: Path | None = None
    EXPORTS_DIR: Path | None = None
    SPATIAL_DIR: Path | None = None
    WEIGHTS_DIR: Path | None = None

    # ── Database & Queues ────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="postgresql://ranger_admin:pench_secure_pass_2026@localhost:5432/pench_wildlife_db",
    )
    SQLITE_FALLBACK_URL: str = Field(
        default="sqlite:///./pench_wildlife_fallback.db",
    )
    USE_POSTGIS: bool = Field(default=False)
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # ── Hardware & Model Limits (≤ 6 GB VRAM Hard Ceiling) ───────
    VRAM_BUDGET_GB: float = 6.0
    DEVICE: str = "cuda:0"
    MD_BATCH_SIZE: int = 32
    DINO_BATCH_SIZE: int = 16

    # ── Detection & Re-ID Thresholds ─────────────────────────────
    CONF_BLANK_RECALL_GATE: float = 0.15
    SIM_REID_AUTO_MATCH: float = 0.85
    SIM_REID_REVIEW_MIN: float = 0.60

    # ── Spatial Alert Parameters ─────────────────────────────────
    DIST_VILLAGE_ALERT_KM: float = 5.0
    SHIFT_CORE_CENTROID_KM2: float = 15.0
    SHIFT_BUFFER_CENTROID_KM2: float = 5.0
    DAYS_ABSENCE_ALERT: int = 45

    # ── Spatial CRS Projections ──────────────────────────────────
    STORAGE_CRS: str = "EPSG:4326"
    METRIC_CRS: str = "EPSG:32644"

    # ── Deduplication ────────────────────────────────────────────
    PHASH_HAMMING_THRESHOLD: int = 2
    BURST_WINDOW_SECONDS: float = 2.0

    def model_post_init(self, __context: object) -> None:
        """Resolve all derived directory paths after field initialization."""
        if self.DATA_DIR is None:
            self.DATA_DIR = self.BASE_DIR / "data"
        if self.RAW_CAMERA_TRAPS_DIR is None:
            self.RAW_CAMERA_TRAPS_DIR = self.DATA_DIR / "raw_camera_traps"
        if self.QUARANTINE_DIR is None:
            self.QUARANTINE_DIR = self.DATA_DIR / "quarantine"
        if self.ACTIVE_WORKING_SET_DIR is None:
            self.ACTIVE_WORKING_SET_DIR = self.DATA_DIR / "active_working_set"
        if self.VECTOR_STORE_DIR is None:
            self.VECTOR_STORE_DIR = self.DATA_DIR / "vector_store"
        if self.MANIFESTS_DIR is None:
            self.MANIFESTS_DIR = self.DATA_DIR / "manifests"
        if self.EXPORTS_DIR is None:
            self.EXPORTS_DIR = self.DATA_DIR / "exports"
        if self.SPATIAL_DIR is None:
            self.SPATIAL_DIR = self.DATA_DIR / "spatial"
        if self.WEIGHTS_DIR is None:
            self.WEIGHTS_DIR = self.BASE_DIR / "weights"


# ── Module-level singleton ───────────────────────────────────────
settings = AppConfig()
