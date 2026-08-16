"""
TERA-STRIPE Module 1 — Camera Trap Ingestion Engine
=====================================================
High-throughput EXIF harvester, station token resolver, perceptual-hash
deduplicator, and ingest manifest generator.

CLI Usage
---------
  python -m src.m1_ingestion \\
      --input-dir ./data/raw_camera_traps/STN_104B \\
      --output-manifest ./data/manifests/manifest_STN_104B.json

Reference: Master Context Packet §8.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("tera_stripe.m1_ingestion")

# ── IST timezone ─────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))

# ── Supported image extensions ───────────────────────────────────
IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".dng", ".bmp",
})


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Version-Safe H3 Cell Indexing                               ║
# ╚═══════════════════════════════════════════════════════════════╝

def _safe_h3_index(lat: float, lng: float, resolution: int) -> str:
    """
    Resilient H3 cell lookup supporting H3 v4, v3, and missing library.

    - H3 v4 API: ``h3.latlng_to_cell(lat, lng, res)``
    - H3 v3 API: ``h3.geo_to_h3(lat, lng, res)``
    - Fallback:  ``"h3_unavailable_res{resolution}"``
    """
    try:
        import h3

        try:
            # H3 v4 API (preferred)
            return h3.latlng_to_cell(lat, lng, resolution)
        except AttributeError:
            pass

        try:
            # H3 v3 API (legacy)
            return h3.geo_to_h3(lat, lng, resolution)
        except AttributeError:
            pass

        return f"h3_api_unknown_res{resolution}"
    except ImportError:
        return f"h3_unavailable_res{resolution}"


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Pench Tiger Reserve — Known Station Registry                ║
# ╚═══════════════════════════════════════════════════════════════╝

PENCH_STATION_REGISTRY: dict[str, dict[str, Any]] = {
    "STN_101": {
        "station_id": "PTR_STN_101",
        "zone": "CORE",
        "range_name": "Karmajhiri",
        "latitude": 21.6780,
        "longitude": 79.2920,
        "elevation_m": 415.0,
    },
    "STN_102": {
        "station_id": "PTR_STN_102",
        "zone": "CORE",
        "range_name": "Karmajhiri",
        "latitude": 21.6810,
        "longitude": 79.2870,
        "elevation_m": 418.0,
    },
    "STN_103": {
        "station_id": "PTR_STN_103",
        "zone": "BUFFER",
        "range_name": "Rukhad",
        "latitude": 21.6950,
        "longitude": 79.3100,
        "elevation_m": 435.0,
    },
    "STN_104B": {
        "station_id": "PTR_STN_104B",
        "zone": "CORE",
        "range_name": "Karmajhiri",
        "latitude": 21.68502,
        "longitude": 79.28504,
        "elevation_m": 420.5,
    },
    "STN_105": {
        "station_id": "PTR_STN_105",
        "zone": "CORRIDOR",
        "range_name": "Turiya",
        "latitude": 21.7120,
        "longitude": 79.3250,
        "elevation_m": 445.0,
    },
    "STN_106": {
        "station_id": "PTR_STN_106",
        "zone": "BUFFER",
        "range_name": "Rukhad",
        "latitude": 21.7020,
        "longitude": 79.2990,
        "elevation_m": 428.0,
    },
    "STN_107": {
        "station_id": "PTR_STN_107",
        "zone": "FRINGE",
        "range_name": "Khawasa",
        "latitude": 21.7250,
        "longitude": 79.3400,
        "elevation_m": 460.0,
    },
}


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Pydantic Contract Models — ingest_manifest.json             ║
# ╚═══════════════════════════════════════════════════════════════╝

class StationMetadata(BaseModel):
    """Resolved camera-trap station metadata."""
    station_id: str
    zone: Literal["CORE", "BUFFER", "CORRIDOR", "FRINGE"]
    range_name: str
    latitude: float
    longitude: float
    elevation_m: float | None = None
    h3_res8: str
    h3_res9: str


class ImageRecord(BaseModel):
    """Per-image EXIF extraction record."""
    image_id: str
    absolute_path: str
    phash: str
    md5_checksum: str
    captured_at: str  # ISO-8601 string
    flash_fired: bool
    ambient_temp_c: float | None = None
    exif_status: Literal["VALID", "CORRUPTED", "FALLBACK_INFERRED"]
    is_burst_duplicate: bool = False


class IngestManifest(BaseModel):
    """Complete batch ingestion manifest per the data contract."""
    batch_id: str
    source_directory: str
    total_frames: int
    station_metadata: StationMetadata
    records: list[ImageRecord]


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Station Token Resolver                                      ║
# ╚═══════════════════════════════════════════════════════════════╝

def parse_station_token(folder_path: Path) -> StationMetadata:
    """
    Resolve a camera-trap station from its folder name.

    Parses the final component of ``folder_path`` against the Pench
    station registry.  Falls back to a generic entry when the token
    is unregistered.

    Parameters
    ----------
    folder_path : Path
        Absolute or relative path ending in the station folder name
        (e.g., ``./data/raw_camera_traps/STN_104B``).

    Returns
    -------
    StationMetadata
    """
    folder_name = folder_path.name.upper()

    # Try direct registry lookup
    if folder_name in PENCH_STATION_REGISTRY:
        entry = PENCH_STATION_REGISTRY[folder_name]
    else:
        # Attempt fuzzy match: strip prefixes like "PENCH_CORE_"
        matched = None
        for key in PENCH_STATION_REGISTRY:
            if key in folder_name:
                matched = key
                break
        if matched:
            entry = PENCH_STATION_REGISTRY[matched]
        else:
            logger.warning(
                "Station token '%s' not in registry — using fallback coordinates.",
                folder_name,
            )
            entry = {
                "station_id": f"PTR_{folder_name}",
                "zone": "BUFFER",
                "range_name": "Unknown",
                "latitude": 21.6900,
                "longitude": 79.3000,
                "elevation_m": 420.0,
            }

    return StationMetadata(
        station_id=entry["station_id"],
        zone=entry["zone"],
        range_name=entry["range_name"],
        latitude=entry["latitude"],
        longitude=entry["longitude"],
        elevation_m=entry.get("elevation_m"),
        h3_res8=_safe_h3_index(entry["latitude"], entry["longitude"], 8),
        h3_res9=_safe_h3_index(entry["latitude"], entry["longitude"], 9),
    )


# ╔═══════════════════════════════════════════════════════════════╗
# ║  EXIF Metadata Extraction                                    ║
# ╚═══════════════════════════════════════════════════════════════╝

def _parse_exif_datetime(dt_string: str) -> str:
    """
    Convert EXIF datetime ``'YYYY:MM:DD HH:MM:SS'`` to ISO-8601 with IST
    timezone offset ``'+05:30'``.
    """
    # Handle both ':' and '-' date separators
    normalized = dt_string.strip().replace("-", ":")
    # Try the standard EXIF format first
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(normalized, fmt)
            return dt.replace(tzinfo=IST).isoformat()
        except ValueError:
            continue

    # If already ISO-8601, return as-is
    try:
        datetime.fromisoformat(dt_string)
        return dt_string
    except ValueError:
        pass

    raise ValueError(f"Cannot parse EXIF datetime: {dt_string!r}")


def _parse_temperature_from_description(desc: str) -> float | None:
    """Extract ambient temperature from ImageDescription field."""
    if not desc:
        return None
    match = re.search(r"Temp[:\s]*([+-]?\d+(?:\.\d+)?)", desc, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def extract_exif_metadata(image_path: Path) -> dict[str, Any]:
    """
    Harvest EXIF fields from a single image file using piexif.

    Extracted fields
    ----------------
    - ``DateTimeOriginal`` → ISO-8601 IST string
    - ``Flash`` → ``bool``
    - ``Make`` / ``Model`` → camera identification
    - ``ImageDescription`` → parsed for serial & temperature
    - ``GPS`` → lat / lon (informational; station registry is authoritative)

    Returns
    -------
    dict
        Keys: ``captured_at``, ``flash_fired``, ``ambient_temp_c``,
        ``camera_model``, ``serial_number``, ``exif_status``.
    """
    result: dict[str, Any] = {
        "captured_at": None,
        "flash_fired": False,
        "ambient_temp_c": None,
        "camera_model": "Unknown",
        "serial_number": "Unknown",
        "exif_status": "CORRUPTED",
    }

    try:
        import piexif

        exif_dict = piexif.load(str(image_path))

        # ── DateTimeOriginal ──
        dt_bytes = exif_dict.get("Exif", {}).get(
            piexif.ExifIFD.DateTimeOriginal
        )
        if dt_bytes:
            dt_str = (
                dt_bytes.decode("utf-8", errors="replace")
                if isinstance(dt_bytes, bytes)
                else str(dt_bytes)
            )
            result["captured_at"] = _parse_exif_datetime(dt_str)
        else:
            # Fallback: use file modification time
            mtime = image_path.stat().st_mtime
            dt_fallback = datetime.fromtimestamp(mtime, tz=IST)
            result["captured_at"] = dt_fallback.isoformat()
            result["exif_status"] = "FALLBACK_INFERRED"

        # ── Flash ──
        flash_val = exif_dict.get("Exif", {}).get(piexif.ExifIFD.Flash, 0)
        result["flash_fired"] = bool(flash_val & 0x01) if isinstance(flash_val, int) else False

        # ── Camera Make / Model ──
        make = exif_dict.get("0th", {}).get(piexif.ImageIFD.Make, b"")
        model = exif_dict.get("0th", {}).get(piexif.ImageIFD.Model, b"")
        if isinstance(make, bytes):
            make = make.decode("utf-8", errors="replace").strip()
        if isinstance(model, bytes):
            model = model.decode("utf-8", errors="replace").strip()
        result["camera_model"] = f"{make} {model}".strip() or "Unknown"

        # ── ImageDescription (serial + temperature) ──
        desc_bytes = exif_dict.get("0th", {}).get(
            piexif.ImageIFD.ImageDescription, b""
        )
        if isinstance(desc_bytes, bytes):
            desc = desc_bytes.decode("utf-8", errors="replace")
        else:
            desc = str(desc_bytes)

        # Parse serial
        serial_match = re.search(r"Serial[:\s]*(\S+)", desc, re.IGNORECASE)
        if serial_match:
            result["serial_number"] = serial_match.group(1)

        # Parse temperature
        result["ambient_temp_c"] = _parse_temperature_from_description(desc)

        # Mark as VALID if we got a DateTimeOriginal
        if result["exif_status"] != "FALLBACK_INFERRED":
            result["exif_status"] = "VALID"

    except Exception as exc:
        logger.warning("EXIF extraction failed for %s: %s", image_path, exc)
        # Use file modification time as ultimate fallback
        try:
            mtime = image_path.stat().st_mtime
            dt_fallback = datetime.fromtimestamp(mtime, tz=IST)
            result["captured_at"] = dt_fallback.isoformat()
        except OSError:
            result["captured_at"] = datetime.now(IST).isoformat()
        result["exif_status"] = "CORRUPTED"

    return result


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Perceptual Hash & MD5 Computation                           ║
# ╚═══════════════════════════════════════════════════════════════╝

def compute_phash_and_md5(image_path: Path) -> tuple[str, str]:
    """
    Compute a 64-bit perceptual hash and MD5 checksum for an image.

    Returns
    -------
    tuple[str, str]
        ``(phash_hex, md5_hex)``
    """
    import imagehash
    from PIL import Image

    # Perceptual hash (64-bit DCT-based)
    with Image.open(image_path) as img:
        phash = imagehash.phash(img)

    # MD5 checksum
    md5 = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)

    return str(phash), md5.hexdigest()


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Intra-Burst Duplicate Detection                             ║
# ╚═══════════════════════════════════════════════════════════════╝

def detect_burst_duplicates(
    records: list[ImageRecord],
    hamming_threshold: int = 2,
    time_window_seconds: float = 2.0,
) -> list[ImageRecord]:
    """
    Flag intra-burst duplicate frames.

    Two images are duplicates when:
      - Hamming distance between their perceptual hashes ≤ ``hamming_threshold``
      - AND their capture timestamps differ by ≤ ``time_window_seconds``

    The first image in each burst is kept as the primary; subsequent
    duplicates are flagged with ``is_burst_duplicate = True``.

    Parameters
    ----------
    records : list[ImageRecord]
        Chronologically sorted image records.
    hamming_threshold : int
        Maximum Hamming distance to consider a match (default 2).
    time_window_seconds : float
        Maximum seconds between captures (default 2.0).

    Returns
    -------
    list[ImageRecord]
        Updated records with burst duplicate flags set.
    """
    import imagehash

    if len(records) < 2:
        return records

    # Sort by capture timestamp
    sorted_records = sorted(records, key=lambda r: r.captured_at)

    for i in range(1, len(sorted_records)):
        current = sorted_records[i]
        if current.is_burst_duplicate:
            continue

        # Parse timestamps for comparison
        try:
            t_current = datetime.fromisoformat(current.captured_at)
        except ValueError:
            continue

        # Look backward within the time window
        for j in range(i - 1, -1, -1):
            prev = sorted_records[j]
            try:
                t_prev = datetime.fromisoformat(prev.captured_at)
            except ValueError:
                continue

            delta = abs((t_current - t_prev).total_seconds())
            if delta > time_window_seconds:
                break  # Outside window, no need to look further

            # Compute Hamming distance
            h_current = imagehash.hex_to_hash(current.phash)
            h_prev = imagehash.hex_to_hash(prev.phash)
            hamming_dist = h_current - h_prev

            if hamming_dist <= hamming_threshold:
                # Flag current as duplicate (keep the earlier one)
                current.is_burst_duplicate = True
                logger.debug(
                    "Burst duplicate: %s ↔ %s (H=%d, Δt=%.1fs)",
                    prev.image_id,
                    current.image_id,
                    hamming_dist,
                    delta,
                )
                break

    return sorted_records


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Manifest Generation                                        ║
# ╚═══════════════════════════════════════════════════════════════╝

def generate_manifest(
    input_dir: Path,
    output_path: Path | None = None,
) -> IngestManifest:
    """
    Scan a camera-trap station folder, extract EXIF & hashes, detect
    burst duplicates, and produce a validated ``IngestManifest``.

    Parameters
    ----------
    input_dir : Path
        Station folder (e.g., ``./data/raw_camera_traps/STN_104B``).
    output_path : Path, optional
        Where to write the JSON manifest.  ``None`` skips file output.

    Returns
    -------
    IngestManifest
    """
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    logger.info("▶ Ingesting station folder: %s", input_dir)

    # ── 1. Resolve station metadata ──
    station = parse_station_token(input_dir)
    logger.info("  Station resolved: %s (%s / %s)", station.station_id, station.zone, station.range_name)

    # ── 2. Enumerate image files ──
    image_files = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    logger.info("  Found %d image files.", len(image_files))

    if not image_files:
        raise ValueError(f"No images found in {input_dir}")

    # ── 3. Extract metadata for each image ──
    records: list[ImageRecord] = []
    for img_path in image_files:
        exif = extract_exif_metadata(img_path)
        phash_hex, md5_hex = compute_phash_and_md5(img_path)

        record = ImageRecord(
            image_id=img_path.stem,
            absolute_path=str(img_path),
            phash=phash_hex,
            md5_checksum=md5_hex,
            captured_at=exif["captured_at"],
            flash_fired=exif["flash_fired"],
            ambient_temp_c=exif["ambient_temp_c"],
            exif_status=exif["exif_status"],
        )
        records.append(record)

    # ── 4. Detect burst duplicates ──
    records = detect_burst_duplicates(records)
    dup_count = sum(1 for r in records if r.is_burst_duplicate)
    if dup_count > 0:
        logger.info("  Detected %d burst duplicate(s).", dup_count)

    # ── 5. Assemble and validate manifest ──
    # Build a deterministic batch ID
    date_token = datetime.now(IST).strftime("%Y_Q%q" if False else "%Y%m%d")
    batch_id = f"BATCH_{date_token}_PENCH_{station.station_id.replace('PTR_', '')}"

    manifest = IngestManifest(
        batch_id=batch_id,
        source_directory=str(input_dir),
        total_frames=len(records),
        station_metadata=station,
        records=records,
    )

    logger.info(
        "  Manifest assembled: %d frames (%d unique, %d duplicates).",
        manifest.total_frames,
        manifest.total_frames - dup_count,
        dup_count,
    )

    # ── 6. Write to disk ──
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info("  ✓ Manifest written to: %s", output_path)

    return manifest


# ╔═══════════════════════════════════════════════════════════════╗
# ║  CLI Entry Point                                             ║
# ╚═══════════════════════════════════════════════════════════════╝

def main() -> None:
    """CLI entry point for the ingestion engine."""
    parser = argparse.ArgumentParser(
        prog="m1_ingestion",
        description="TERA-STRIPE M1 — Camera Trap Ingestion Engine",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to camera-trap station folder (e.g., ./data/raw_camera_traps/STN_104B)",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Output path for the ingest manifest JSON.",
    )
    args = parser.parse_args()

    try:
        manifest = generate_manifest(args.input_dir, args.output_manifest)
        print(
            f"\n{'='*60}\n"
            f"  TERA-STRIPE M1 Ingestion Complete\n"
            f"  Station   : {manifest.station_metadata.station_id}\n"
            f"  Zone      : {manifest.station_metadata.zone}\n"
            f"  Range     : {manifest.station_metadata.range_name}\n"
            f"  Frames    : {manifest.total_frames}\n"
            f"  Duplicates: {sum(1 for r in manifest.records if r.is_burst_duplicate)}\n"
            f"  Manifest  : {args.output_manifest}\n"
            f"{'='*60}"
        )
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
