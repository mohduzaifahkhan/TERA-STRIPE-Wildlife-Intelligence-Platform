"""
TERA-STRIPE -- Run Real AI Pipeline on Your Photos
=====================================================
Downloads models and processes your actual camera trap images
through real MegaDetector + DINOv2 inference on GPU.

Usage:
  python scripts/run_real_pipeline.py --source "C:\\path\\to\\your\\photos"
  python scripts/run_real_pipeline.py --source "C:\\path\\to\\photos" --output-dir data/real_results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("run_real_pipeline")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def extract_video_frames(video_path: Path, output_dir: Path, max_frames: int = 10) -> list[Path]:
    """Extract key frames from a video file using OpenCV."""
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV not installed. Run: pip install opencv-python")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration_s = total_frames / fps if fps > 0 else 0

    logger.info("Video: %s | %.1fs | %d frames | %.1f fps", video_path.name, duration_s, total_frames, fps)

    # Sample frames evenly across the video
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames // max_frames
        frame_indices = [i * step for i in range(max_frames)]

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = []

    for idx, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        out_name = f"{video_path.stem}_frame{idx:04d}.jpg"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), frame)
        extracted.append(out_path)

    cap.release()
    logger.info("Extracted %d frames from %s", len(extracted), video_path.name)
    return extracted


def collect_images(source_dir: Path, frames_dir: Path) -> list[Path]:
    """Collect all image files and extract frames from videos."""
    images = []

    for f in sorted(source_dir.iterdir()):
        if f.suffix.lower() in IMAGE_EXTS:
            images.append(f)
        elif f.suffix.lower() in VIDEO_EXTS:
            logger.info("Found video: %s — extracting frames...", f.name)
            frames = extract_video_frames(f, frames_dir)
            images.extend(frames)

    # Also check subdirectories (one level deep)
    for subdir in source_dir.iterdir():
        if subdir.is_dir():
            for f in sorted(subdir.iterdir()):
                if f.suffix.lower() in IMAGE_EXTS:
                    images.append(f)
                elif f.suffix.lower() in VIDEO_EXTS:
                    frames = extract_video_frames(f, frames_dir)
                    images.extend(frames)

    return images


def main():
    parser = argparse.ArgumentParser(
        description="Run real AI pipeline on your photos/videos",
    )
    parser.add_argument(
        "--source", type=Path, required=True,
        help="Directory containing your camera trap photos/videos",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/real_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"),
        help="Directory to store/find model weights",
    )
    parser.add_argument(
        "--max-video-frames", type=int, default=10,
        help="Max frames to extract per video",
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true",
        help="Skip DINOv2 embeddings (faster, triage+crop only)",
    )
    parser.add_argument(
        "--pose-weights", type=Path, default=Path("weights/yolo11_pose_tiger.pt"),
        help="Path to trained YOLO11-Pose tiger weights",
    )
    parser.add_argument(
        "--rclone-remote", type=str, default=None,
        help="Rclone remote destination (e.g. gdrive:pench_results/)",
    )

    args = parser.parse_args()

    if not args.source.exists():
        logger.error("Source directory does not exist: %s", args.source)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = args.output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "video_frames"

    # ── Step 1: Collect images + extract video frames ────────────
    print("\n" + "=" * 60)
    print("  TERA-STRIPE Real AI Pipeline")
    print("=" * 60)

    images = collect_images(args.source, frames_dir)
    if not images:
        print(f"\n  No images or videos found in {args.source}")
        print(f"  Supported: {IMAGE_EXTS | VIDEO_EXTS}")
        sys.exit(1)

    print(f"\n  Found {len(images)} images to process")
    print(f"  Source: {args.source}")
    print(f"  Output: {args.output_dir}\n")

    # ── Step 2: MegaDetector Triage ──────────────────────────────
    from src.real_backends import MegaDetectorBackend, DINOv2Backend

    print("=" * 60)
    print("  STEP 1: MegaDetector v5a — Animal Detection")
    print("=" * 60)

    md = MegaDetectorBackend(model_path=args.models_dir / "md_v5a.0.0.pt")
    md.load()

    triage_results = []
    fauna_images = []
    blank_count = 0

    for i, img_path in enumerate(images):
        t0 = time.time()
        result = md.predict(str(img_path))
        elapsed = time.time() - t0

        result["image_path"] = str(img_path)
        result["image_name"] = img_path.name
        triage_results.append(result)

        status_icon = {
            "FAUNA": "🐅", "BLANK": "⬜", "HUMAN": "🚶", "VEHICLE": "🚗"
        }.get(result["label"], "❓")

        print(f"  [{i+1}/{len(images)}] {status_icon} {result['label']:8s} "
              f"conf={result['confidence']:.2f}  {img_path.name}  ({elapsed:.1f}s)")

        if result["label"] == "FAUNA":
            fauna_images.append(result)
        elif result["label"] == "BLANK":
            blank_count += 1

    md.unload()

    # Save triage results
    triage_path = args.output_dir / "triage_results.json"
    with open(triage_path, "w") as f:
        json.dump(triage_results, f, indent=2)

    print(f"\n  Summary: {len(fauna_images)} FAUNA | {blank_count} BLANK | "
          f"{len(images) - len(fauna_images) - blank_count} OTHER")
    print(f"  Saved: {triage_path}\n")

    if not fauna_images:
        print("  No animals detected. Pipeline stops here.")
        sys.exit(0)

    # ── Step 3: YOLO11-Pose Flank Extraction (M4) ────────────────
    print("=" * 60)
    print("  STEP 2: YOLO11-Pose — Tiger Flank Extraction (6 Keypoints)")
    print("=" * 60)

    from src.m4_flank_pose import create_pose_backend, FlankExtractionEngine

    pose_weights = Path(args.pose_weights)
    if not pose_weights.is_absolute():
        pose_weights = PROJECT_ROOT / pose_weights

    backend = create_pose_backend(
        weights_path=pose_weights,
        device="cuda:0",
        force_mock=False,
    )
    print(f"  Backend: {backend.backend_name}")

    # Build triage_data dict compatible with FlankExtractionEngine
    triage_data_for_m4 = {
        "batch_id": f"pipeline_{int(time.time())}",
        "dispatches": [],
    }
    for item in fauna_images:
        triage_data_for_m4["dispatches"].append({
            "image_id": Path(item["image_path"]).stem,
            "status": "FAUNA_DETECTED",
            "source_path": item["image_path"],
        })

    # Source records for path lookup
    source_records = [
        {"image_id": Path(it["image_path"]).stem, "absolute_path": it["image_path"]}
        for it in fauna_images
    ]

    engine = FlankExtractionEngine(
        backend=backend,
        crops_dir=crops_dir,
        crop_size=224,
        batch_size=16,
    )
    m4_result = engine.process_triage(triage_data_for_m4, source_records)

    # Convert M4 output to crop_results format for downstream compatibility
    crop_results = []
    for ext in m4_result.extractions:
        crop_results.append({
            "image_path": ext.source_path,
            "image_name": Path(ext.source_path).name,
            "crop_path": ext.crop_path,
            "flank_side": ext.flank_side,
            "bbox": ext.bbox_normalized,
            "confidence": ext.quality_score,
            "quality_score": ext.quality_score,
            "affine_warp_applied": ext.affine_warp_applied,
            "keypoints_detected": len(ext.keypoints),
        })
        print(f"  🐅 {ext.flank_side:5s} | Q={ext.quality_score:.3f} | "
              f"warp={'✓' if ext.affine_warp_applied else '✗'} | "
              f"kpts={len(ext.keypoints)} | {Path(ext.crop_path).name}")

    crops_path = args.output_dir / "crop_results.json"
    with open(crops_path, "w") as f:
        json.dump(crop_results, f, indent=2)

    # Save full M4 extraction result
    m4_json_path = args.output_dir / "flank_extraction.json"
    with open(m4_json_path, "w") as f:
        json.dump(m4_result.model_dump(), f, indent=2, ensure_ascii=False)

    s = m4_result.summary
    print(f"\n  Extractions: {s.total_extractions} (L={s.left_flanks} R={s.right_flanks} A={s.ambiguous_flanks})")
    print(f"  High quality (≥0.7): {s.high_quality_count}")
    print(f"  Mean quality score: {s.mean_quality_score:.3f}")
    print(f"  Saved: {crops_path}")
    print(f"  Saved: {m4_json_path}\n")

    if args.skip_embeddings:
        print("  Skipping DINOv2 embeddings (--skip-embeddings)")
        print_summary(triage_results, crop_results, [])
        return

    # ── Step 4: DINOv2 Embeddings ────────────────────────────────
    print("=" * 60)
    print("  STEP 3: DINOv2 — Tiger Re-ID Embeddings")
    print("=" * 60)

    dinov2 = DINOv2Backend()
    dinov2.load()

    embedding_results = []
    import numpy as np

    for i, crop in enumerate(crop_results):
        t0 = time.time()
        result = dinov2.predict(crop["crop_path"])
        elapsed = time.time() - t0

        result["crop_path"] = crop["crop_path"]
        result["image_name"] = crop["image_name"]
        result["flank_side"] = crop["flank_side"]
        embedding_results.append(result)

        print(f"  [{i+1}/{len(crop_results)}] Embedding: {result['embedding_dim']}d "
              f"| {crop['flank_side']} | {elapsed:.1f}s")

    dinov2.unload()

    # Compare embeddings (cosine similarity matrix)
    if len(embedding_results) >= 2:
        print(f"\n  Cosine Similarity Matrix:")
        embeddings = np.array([r["embedding"] for r in embedding_results])
        names = [Path(r["crop_path"]).stem[:25] for r in embedding_results]

        sim_matrix = np.dot(embeddings, embeddings.T)

        # Print header
        print(f"  {'':>26s}", end="")
        for n in names:
            print(f" {n[:8]:>8s}", end="")
        print()

        for i, name in enumerate(names):
            print(f"  {name:>26s}", end="")
            for j in range(len(names)):
                sim = sim_matrix[i, j]
                print(f" {sim:>8.3f}", end="")
            print()

    # Save embeddings (without raw vectors for readability)
    embed_summary = []
    for r in embedding_results:
        embed_summary.append({
            "crop_path": r["crop_path"],
            "image_name": r["image_name"],
            "flank_side": r["flank_side"],
            "embedding_dim": r["embedding_dim"],
        })

    embed_path = args.output_dir / "embedding_results.json"
    with open(embed_path, "w") as f:
        json.dump(embed_summary, f, indent=2)

    # Save full embeddings as numpy
    np.save(str(args.output_dir / "embeddings.npy"), embeddings)

    print(f"\n  Saved: {embed_path}")
    print(f"  Embeddings: {args.output_dir / 'embeddings.npy'}\n")

    print_summary(triage_results, crop_results, embedding_results)

    # ── Auto-ingest results into dashboard database ──────────────
    _ingest_to_dashboard(triage_results, crop_results, embedding_results)

    if args.rclone_remote:
        print("=" * 60)
        print(f"  STEP 4: Rclone Cloud Sync -> {args.rclone_remote}")
        print("=" * 60)
        import subprocess
        try:
            cmd = ["rclone", "sync", str(args.output_dir), args.rclone_remote, "--progress"]
            print(f"  Running: {' '.join(cmd)}\n")
            subprocess.run(cmd, check=True)
            print(f"\n  [SUCCESS] All results successfully synced to {args.rclone_remote}")
        except Exception as e:
            logger.error("Rclone sync failed: %s", e)


def _ingest_to_dashboard(triage_results, crop_results, embedding_results):
    """Auto-ingest pipeline results into the SQLite database for the dashboard."""
    print("\n" + "=" * 60)
    print("  STEP 5: Database Ingest (Dashboard)")
    print("=" * 60)

    try:
        from src.m7_db_manager import DatabaseManager
        from datetime import datetime, timezone

        db_path = PROJECT_ROOT / "tera_stripe.db"
        db = DatabaseManager(db_url=f"sqlite:///{db_path}")

        tigers_created = 0
        sightings_logged = 0

        # Create tiger profiles and log sightings from crop results
        for i, crop in enumerate(crop_results):
            tiger_id = f"PTR_T_{i + 1:03d}"
            name = f"Tiger_{i + 1:03d}"
            flank = crop.get("flank_side", "LEFT")
            confidence = crop.get("confidence", 0.0)
            img_path = crop.get("image_path", "")

            # Create/update tiger profile
            db.create_or_update_tiger(
                tiger_id=tiger_id,
                common_name=name,
                sex="UNKNOWN",
                status="RESIDENT",
            )
            tigers_created += 1

            # Log sighting
            db.log_sighting(
                tiger_id=tiger_id,
                station_id="PTR_STN_101",
                captured_at=datetime.now(timezone.utc),
                flank_orientation=f"{flank}_FLANK",
                reid_confidence=confidence,
                verification_status="AUTO_COMMITTED",
                raw_image_path=img_path,
                flank_crop_path=crop.get("crop_path", ""),
            )
            sightings_logged += 1

        # Register stations from the Pench registry
        stations_registered = 0
        try:
            from src.m1_ingestion import PENCH_STATION_REGISTRY
            for key, info in PENCH_STATION_REGISTRY.items():
                manifest = {
                    "station": {
                        "station_id": info["station_id"],
                        "zone_type": info.get("zone", "CORE"),
                        "range_name": info.get("range_name", ""),
                        "geom_wkt": f"POINT({info['longitude']} {info['latitude']})",
                        "elevation_m": info.get("elevation_m"),
                    }
                }
                n = db.register_stations_from_manifest(manifest)
                stations_registered += n
        except Exception:
            pass

        # Generate alerts from pipeline results
        alerts_generated = 0
        try:
            from src.m9_alerts import AlertEngine
            engine = AlertEngine()
            for crop in crop_results:
                tiger_id = f"PTR_T_{crop_results.index(crop) + 1:03d}"
                sighting = {
                    "tiger_id": tiger_id,
                    "station_id": "PTR_STN_101",
                    "latitude": 21.6780,
                    "longitude": 79.2920,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "confidence": crop.get("confidence", 0.0),
                }
                result = engine.evaluate_sighting(sighting)
                if result and result.get("alerts"):
                    alerts_generated += len(result["alerts"])
        except Exception:
            pass

        print(f"  Tigers created:     {tigers_created}")
        print(f"  Sightings logged:   {sightings_logged}")
        print(f"  Stations registered: {stations_registered}")
        print(f"  Alerts generated:   {alerts_generated}")
        print(f"  Database: {db_path}")
        print(f"\n  [OK] Dashboard will show live data at http://localhost:8501")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"  [WARN] Database ingest skipped: {e}")
        print("  Dashboard will use whatever data is available.\n")


def print_summary(triage, crops, embeddings):
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    fauna = sum(1 for t in triage if t["label"] == "FAUNA")
    blank = sum(1 for t in triage if t["label"] == "BLANK")
    print(f"  Images processed: {len(triage)}")
    print(f"  Animals detected: {fauna}")
    print(f"  Blanks filtered:  {blank}")
    print(f"  Crops extracted:  {len(crops)}")
    print(f"  Embeddings:       {len(embeddings)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
