"""
TERA-STRIPE Module 4 Real Model Validation Script
===================================================
Runs real trained YOLO11-Pose model on ATRW validation images, extracts
6 anatomical landmarks, computes flank laterality (LEFT/RIGHT), applies
affine warp orientation, computes quality scores, and saves crops.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.m4_flank_pose import (
    FlankExtraction,
    FlankExtractionResult,
    FlankExtractionSummary,
    Keypoint,
    YOLOPoseBackend,
    affine_warp_flank,
    compute_quality_score,
    determine_flank_side,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_m4_real")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate M4 with real trained YOLO11-Pose model")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/yolo11_pose_tiger.pt"),
        help="Path to trained YOLO11-Pose weights",
    )
    parser.add_argument(
        "--val-images-dir",
        type=Path,
        default=Path("data/tiger_pose_dataset/images/val"),
        help="Directory containing validation images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/m4_validation_results"),
        help="Output directory for validation artifacts and crops",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=25,
        help="Number of validation images to test",
    )
    args = parser.parse_args()

    if not args.weights.exists():
        logger.error("Weights file not found: %s", args.weights)
        sys.exit(1)

    crops_dir = args.output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    val_images = sorted(list(args.val_images_dir.glob("*.jpg")) + list(args.val_images_dir.glob("*.png")))
    if not val_images:
        logger.error("No validation images found in: %s", args.val_images_dir)
        sys.exit(1)

    selected_images = val_images[: args.num_samples]

    print("=" * 65)
    print("  TERA-STRIPE M4 REAL YOLO11-POSE VALIDATION")
    print("=" * 65)
    print(f"Weights      : {args.weights.resolve()}")
    print(f"Val Images   : {len(selected_images)} samples from {args.val_images_dir}")
    print(f"Crops Output : {crops_dir.resolve()}")
    print("=" * 65)

    backend = YOLOPoseBackend(
        weights_path=args.weights,
        device="cuda:0",
        confidence_threshold=0.25,
    )
    backend.load()

    batch_detections = backend.predict_batch(selected_images)
    backend.unload()

    extractions: list[FlankExtraction] = []
    left_count = 0
    right_count = 0
    ambiguous_count = 0
    high_quality_count = 0

    for img_path, detections in zip(selected_images, batch_detections):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        if not detections:
            logger.warning("No tiger detected in: %s", img_path.name)
            continue

        best_det = detections[0]  # First detected tiger
        kps_list = [
            Keypoint(
                name=k["name"],
                x=k["x"],
                y=k["y"],
                confidence=k["confidence"],
            )
            for k in best_det["keypoints"]
        ]

        flank_side = determine_flank_side(kps_list)
        crop_img, warp_applied = affine_warp_flank(img, kps_list, crop_size=224)
        quality = compute_quality_score(kps_list, crop_img, flank_side)

        crop_filename = f"{img_path.stem}_{flank_side}.jpg"
        crop_save_path = crops_dir / crop_filename
        crop_img.save(crop_save_path, quality=95)

        if flank_side == "LEFT":
            left_count += 1
        elif flank_side == "RIGHT":
            right_count += 1
        else:
            ambiguous_count += 1

        if quality >= 0.70:
            high_quality_count += 1

        ext = FlankExtraction(
            image_id=img_path.stem,
            source_path=str(img_path),
            crop_path=str(crop_save_path),
            flank_side=flank_side,
            keypoints=kps_list,
            quality_score=quality,
            bbox_normalized=best_det["bbox"],
            crop_size_px=224,
            affine_warp_applied=warp_applied,
        )
        extractions.append(ext)

        kpt_conf_mean = sum(k.confidence for k in kps_list) / len(kps_list) if kps_list else 0.0
        print(
            f"  [{img_path.name}] Flank: {flank_side:9s} | Quality: {quality:.3f} | "
            f"Mean Kpt Conf: {kpt_conf_mean:.3f} | Warp: {'Yes' if warp_applied else 'No'}"
        )

    mean_quality = sum(e.quality_score for e in extractions) / len(extractions) if extractions else 0.0

    summary = FlankExtractionSummary(
        total_fauna_frames=len(selected_images),
        total_extractions=len(extractions),
        high_quality_count=high_quality_count,
        left_flanks=left_count,
        right_flanks=right_count,
        ambiguous_flanks=ambiguous_count,
        mean_quality_score=round(mean_quality, 4),
    )

    result = FlankExtractionResult(
        batch_id="M4_REAL_VAL_ATRW",
        crops_directory=str(crops_dir),
        summary=summary,
        extractions=extractions,
    )

    result_json = args.output_dir / "flank_extraction.json"
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2)

    print("\n" + "=" * 65)
    print("  VALIDATION RESULTS SUMMARY")
    print("=" * 65)
    print(f"Total Validated Images : {len(selected_images)}")
    print(f"Successful Extractions : {len(extractions)} ({len(extractions)/len(selected_images)*100:.1f}%)")
    print(f"Flank Laterality       : {left_count} LEFT | {right_count} RIGHT | {ambiguous_count} AMBIGUOUS")
    print(f"High Quality (>= 0.70) : {high_quality_count}/{len(extractions)} ({high_quality_count/max(1, len(extractions))*100:.1f}%)")
    print(f"Mean Quality Score     : {mean_quality:.4f}")
    print(f"Saved Extraction JSON  : {result_json.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
