#!/usr/bin/env python3
"""
TERA-STRIPE Test Fixture Generator
====================================
Creates synthetic camera-trap JPEG images with embedded EXIF metadata
inside ``data/raw_camera_traps/STN_104B/`` for deterministic testing.

Generated images:
  - IMG_00001 – IMG_00008 : 8 unique images (varied patterns, times)
  - IMG_00009, IMG_00010  : Near-duplicates of IMG_00008 (burst test)

Usage:
  python scripts/create_test_fixtures.py
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

# Resolve project root (one level above scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw_camera_traps" / "STN_104B"


def _make_exif_dict(
    dt_str: str,
    flash: bool = False,
    camera_model: str = "TrailCam TC-500",
    serial: str = "SN_PTR_104B_2026",
    temp_c: float = 14.0,
    lat: float = 21.68502,
    lon: float = 79.28504,
) -> dict:
    """Build a piexif-compatible EXIF dictionary."""
    import piexif

    # 0th IFD (basic image info + serial/temp in ImageDescription)
    zeroth_ifd = {
        piexif.ImageIFD.Make: b"WildlifeTech",
        piexif.ImageIFD.Model: camera_model.encode("utf-8"),
        piexif.ImageIFD.Software: b"TERA-STRIPE TestFixture v1.0",
        piexif.ImageIFD.ImageDescription: (
            f"Serial:{serial};Temp:{temp_c}C"
        ).encode("utf-8"),
    }

    # Exif IFD (datetime, flash)
    exif_ifd = {
        piexif.ExifIFD.DateTimeOriginal: dt_str.encode("utf-8"),
        piexif.ExifIFD.Flash: 1 if flash else 0,
    }

    # GPS IFD — encode lat/lon as rational tuples
    def _to_rational(deg_float: float):
        d = int(deg_float)
        m = int((deg_float - d) * 60)
        s = int(round(((deg_float - d) * 60 - m) * 60 * 10000))
        return ((d, 1), (m, 1), (s, 10000))

    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: b"N",
        piexif.GPSIFD.GPSLatitude: _to_rational(abs(lat)),
        piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W",
        piexif.GPSIFD.GPSLongitude: _to_rational(abs(lon)),
    }

    return {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd, "1st": {}}


def _create_image(
    filepath: Path,
    width: int = 640,
    height: int = 480,
    seed: int | None = None,
    exif_dict: dict | None = None,
) -> None:
    """Generate a synthetic JPEG with random visual patterns and EXIF."""
    import piexif
    from PIL import Image, ImageDraw

    rng = random.Random(seed)

    # Base color
    bg_color = (rng.randint(20, 80), rng.randint(30, 60), rng.randint(10, 50))
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw random rectangles to create unique visual fingerprint
    for _ in range(rng.randint(4, 12)):
        x1 = rng.randint(0, width - 80)
        y1 = rng.randint(0, height - 80)
        x2 = x1 + rng.randint(30, 120)
        y2 = y1 + rng.randint(30, 120)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        draw.rectangle([x1, y1, x2, y2], fill=color)

    # Draw some ellipses (simulate animal shapes)
    for _ in range(rng.randint(1, 3)):
        cx = rng.randint(100, width - 100)
        cy = rng.randint(100, height - 100)
        rx = rng.randint(40, 100)
        ry = rng.randint(20, 60)
        color = (rng.randint(150, 255), rng.randint(100, 200), rng.randint(50, 150))
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    if exif_dict:
        exif_bytes = piexif.dump(exif_dict)
        img.save(str(filepath), "JPEG", exif=exif_bytes, quality=85)
    else:
        img.save(str(filepath), "JPEG", quality=85)


def create_fixtures() -> None:
    """Generate the full set of test fixtures."""
    print(f"Creating test fixtures in: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 8 unique images with varied timestamps and EXIF ──
    image_specs = [
        ("IMG_00001", "2026:02:14 02:15:30", False, 12.5, 1001),
        ("IMG_00002", "2026:02:14 03:22:15", True, 14.0, 1002),
        ("IMG_00003", "2026:02:14 04:05:42", False, 11.8, 1003),
        ("IMG_00004", "2026:02:14 06:30:00", True, 10.2, 1004),
        ("IMG_00005", "2026:02:14 08:45:20", False, 16.5, 1005),
        ("IMG_00006", "2026:02:14 11:10:55", False, 22.3, 1006),
        ("IMG_00007", "2026:02:14 15:30:10", False, 28.1, 1007),
        ("IMG_00008", "2026:02:14 19:45:00", True, 18.7, 1008),
    ]

    for name, dt_str, flash, temp, seed in image_specs:
        exif = _make_exif_dict(
            dt_str=dt_str,
            flash=flash,
            temp_c=temp,
            serial="SN_PTR_104B_2026",
        )
        filepath = OUTPUT_DIR / f"{name}.JPG"
        _create_image(filepath, seed=seed, exif_dict=exif)
        print(f"  [OK] {filepath.name}  (dt={dt_str}, flash={flash}, temp={temp}C)")

    # ── 2 burst duplicates of IMG_00008 ──
    #   Same seed → identical visual content → pHash Hamming distance = 0
    #   Timestamps within 2 seconds of IMG_00008
    burst_specs = [
        ("IMG_00009", "2026:02:14 19:45:01", True, 18.7, 1008),  # +1 second
        ("IMG_00010", "2026:02:14 19:45:01", True, 18.7, 1008),  # +1.5 seconds (same EXIF second)
    ]

    for name, dt_str, flash, temp, seed in burst_specs:
        exif = _make_exif_dict(
            dt_str=dt_str,
            flash=flash,
            temp_c=temp,
            serial="SN_PTR_104B_2026",
        )
        filepath = OUTPUT_DIR / f"{name}.JPG"
        _create_image(filepath, seed=seed, exif_dict=exif)
        print(f"  [OK] {filepath.name}  (BURST DUPLICATE, dt={dt_str})")

    print(f"\nDone -- {len(image_specs) + len(burst_specs)} fixture images created.")


if __name__ == "__main__":
    create_fixtures()
