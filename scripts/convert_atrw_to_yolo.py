"""
TERA-STRIPE ATRW to YOLO11-Pose Dataset Conversion Script
===========================================================
Converts Amur Tiger Re-identification in the Wild (ATRW) pose dataset
annotations to YOLO11-Pose format with 6 TERA-STRIPE anatomical landmarks:

Landmark Mapping:
  K1 (0): Shoulder / Scapula
  K2 (1): Hip / Pelvis root
  K3 (2): Spine midpoint
  K4 (3): Ventral / Belly contour
  K5 (4): Foreleg root
  K6 (5): Hindleg root

Output format per image:
  <class_id> <x_center> <y_center> <w> <h> <k1_x> <k1_y> <k1_v> ... <k6_x> <k6_y> <k6_v>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("convert_atrw_to_yolo")

# 6 Landmark Names for TERA-STRIPE
TERA_STRIPE_KPT_NAMES = [
    "shoulder_scapula",
    "hip_pelvis_root",
    "spine_midpoint",
    "ventral_belly_contour",
    "foreleg_root",
    "hindleg_root",
]


def map_atrw_15_to_6_landmarks(
    kpts: list[float],
) -> list[tuple[float, float, int]]:
    """
    Map 15 ATRW COCO keypoints (x, y, v) to 6 TERA-STRIPE anatomical landmarks.

    ATRW 15 keypoint indices:
      0: left_ear, 1: right_ear, 2: nose,
      3: right_shoulder, 4: right_front_paw,
      5: left_shoulder, 6: left_front_paw,
      7: right_hip, 8: right_knee, 9: right_back_paw,
      10: left_hip, 11: left_knee, 12: left_back_paw,
      13: tail, 14: center
    """
    if len(kpts) < 45:
        return [(0.0, 0.0, 0)] * 6

    parsed = [(float(kpts[i * 3]), float(kpts[i * 3 + 1]), int(kpts[i * 3 + 2])) for i in range(15)]
    (
        l_ear, r_ear, nose,
        r_sh, r_fp,
        l_sh, l_fp,
        r_hip, r_kn, r_bp,
        l_hip, l_kn, l_bp,
        tail, center
    ) = parsed

    # 1. K1 (Shoulder / Scapula)
    sh_candidates = [p for p in (l_sh, r_sh) if p[2] > 0]
    if len(sh_candidates) == 2:
        k1 = ((sh_candidates[0][0] + sh_candidates[1][0]) / 2.0, (sh_candidates[0][1] + sh_candidates[1][1]) / 2.0, 2)
    elif len(sh_candidates) == 1:
        k1 = (sh_candidates[0][0], sh_candidates[0][1], sh_candidates[0][2])
    else:
        k1 = (0.0, 0.0, 0)

    # 2. K2 (Hip / Pelvis root)
    hip_candidates = [p for p in (l_hip, r_hip) if p[2] > 0]
    if len(hip_candidates) == 2:
        k2 = ((hip_candidates[0][0] + hip_candidates[1][0]) / 2.0, (hip_candidates[0][1] + hip_candidates[1][1]) / 2.0, 2)
    elif len(hip_candidates) == 1:
        k2 = (hip_candidates[0][0], hip_candidates[0][1], hip_candidates[0][2])
    else:
        k2 = (0.0, 0.0, 0)

    # 3. K3 (Spine midpoint)
    if center[2] > 0:
        k3 = (center[0], center[1], center[2])
    elif k1[2] > 0 and k2[2] > 0:
        k3 = ((k1[0] + k2[0]) / 2.0, (k1[1] + k2[1]) / 2.0, 1)
    else:
        k3 = (0.0, 0.0, 0)

    # 5. K5 (Foreleg root)
    fl_candidates = []
    if l_sh[2] > 0 and l_fp[2] > 0:
        fl_candidates.append((l_sh[0] + 0.35 * (l_fp[0] - l_sh[0]), l_sh[1] + 0.35 * (l_fp[1] - l_sh[1]), min(l_sh[2], l_fp[2])))
    if r_sh[2] > 0 and r_fp[2] > 0:
        fl_candidates.append((r_sh[0] + 0.35 * (r_fp[0] - r_sh[0]), r_sh[1] + 0.35 * (r_fp[1] - r_sh[1]), min(r_sh[2], r_fp[2])))
    if fl_candidates:
        if len(fl_candidates) == 1:
            k5 = fl_candidates[0]
        else:
            k5 = ((fl_candidates[0][0] + fl_candidates[1][0]) / 2.0, (fl_candidates[0][1] + fl_candidates[1][1]) / 2.0, 2)
    elif k1[2] > 0:
        k5 = (k1[0], k1[1], 1)
    else:
        k5 = (0.0, 0.0, 0)

    # 6. K6 (Hindleg root)
    hl_candidates = [p for p in (l_kn, r_kn) if p[2] > 0]
    if not hl_candidates:
        if l_hip[2] > 0 and l_bp[2] > 0:
            hl_candidates.append((l_hip[0] + 0.35 * (l_bp[0] - l_hip[0]), l_hip[1] + 0.35 * (l_bp[1] - l_hip[1]), min(l_hip[2], l_bp[2])))
        if r_hip[2] > 0 and r_bp[2] > 0:
            hl_candidates.append((r_hip[0] + 0.35 * (r_bp[0] - r_hip[0]), r_hip[1] + 0.35 * (r_bp[1] - r_hip[1]), min(r_hip[2], r_bp[2])))
    if hl_candidates:
        if len(hl_candidates) == 1:
            k6 = hl_candidates[0]
        else:
            k6 = ((hl_candidates[0][0] + hl_candidates[1][0]) / 2.0, (hl_candidates[0][1] + hl_candidates[1][1]) / 2.0, 2)
    elif k2[2] > 0:
        k6 = (k2[0], k2[1], 1)
    else:
        k6 = (0.0, 0.0, 0)

    # 4. K4 (Ventral / Belly contour)
    if k5[2] > 0 and k6[2] > 0:
        k4 = ((k5[0] + k6[0]) / 2.0, (k5[1] + k6[1]) / 2.0, min(k5[2], k6[2]))
    elif k3[2] > 0 and (k1[2] > 0 or k2[2] > 0):
        k4 = (k3[0], k3[1], 1)
    else:
        k4 = (0.0, 0.0, 0)

    return [k1, k2, k3, k4, k5, k6]


def convert_split(
    split_name: str,
    anno_json_path: Path,
    source_img_dir: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    copy_images: bool = True,
) -> dict[str, int]:
    """Convert a single split (train or val) to YOLO Pose format."""
    logger.info("Processing %s split: %s", split_name, anno_json_path)

    with open(anno_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_info = {img["id"]: img for img in data.get("images", [])}
    annotations_by_img: dict[int, list[dict[str, Any]]] = {}

    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        annotations_by_img.setdefault(img_id, []).append(ann)

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_images": len(images_info),
        "images_with_keypoints": 0,
        "images_converted": 0,
        "total_annotations": 0,
    }

    for img_id, img_info in images_info.items():
        filename = img_info["filename"]
        img_w = float(img_info["width"])
        img_h = float(img_info["height"])

        src_img_file = source_img_dir / filename
        if not src_img_file.exists():
            candidates = list(source_img_dir.glob(f"**/{filename}"))
            if candidates:
                src_img_file = candidates[0]
            else:
                logger.warning("Image file not found: %s", filename)
                continue

        dst_img_file = out_img_dir / filename
        dst_lbl_file = out_lbl_dir / f"{Path(filename).stem}.txt"

        anns = annotations_by_img.get(img_id, [])
        valid_label_lines = []

        for ann in anns:
            kpts = ann.get("keypoints", [])
            has_kpts = any(kpts)

            if not has_kpts:
                continue

            mapped_6kpts = map_atrw_15_to_6_landmarks(kpts)
            num_valid_kpts = sum(1 for kp in mapped_6kpts if kp[2] > 0)
            if num_valid_kpts == 0:
                continue

            # Bounding box in COCO format: [x_min, y_min, width, height]
            bbox = ann.get("bbox", [0, 0, img_w, img_h])
            bx_min, by_min, bw, bh = bbox
            if bw <= 0 or bh <= 0:
                bw = img_w
                bh = img_h
                bx_min = 0
                by_min = 0

            # Normalize bounding box
            x_center = (bx_min + bw / 2.0) / img_w
            y_center = (by_min + bh / 2.0) / img_h
            norm_w = bw / img_w
            norm_h = bh / img_h

            # Clamp bbox values
            x_center = min(1.0, max(0.0, x_center))
            y_center = min(1.0, max(0.0, y_center))
            norm_w = min(1.0, max(0.001, norm_w))
            norm_h = min(1.0, max(0.001, norm_h))

            # Normalize keypoints
            kpt_tokens = []
            for kx, ky, kv in mapped_6kpts:
                if kv > 0:
                    norm_kx = min(1.0, max(0.0, kx / img_w))
                    norm_ky = min(1.0, max(0.0, ky / img_h))
                    v_flag = kv
                else:
                    norm_kx = 0.0
                    norm_ky = 0.0
                    v_flag = 0
                kpt_tokens.append(f"{norm_kx:.6f} {norm_ky:.6f} {v_flag}")

            line = f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f} " + " ".join(kpt_tokens)
            valid_label_lines.append(line)
            stats["total_annotations"] += 1

        if valid_label_lines:
            stats["images_with_keypoints"] += 1
            with open(dst_lbl_file, "w", encoding="utf-8") as f:
                f.write("\n".join(valid_label_lines) + "\n")

            if copy_images and not dst_img_file.exists():
                shutil.copy2(src_img_file, dst_img_file)
            stats["images_converted"] += 1

    logger.info(
        "Finished %s: Converted %d/%d images (%d total tiger annotations)",
        split_name,
        stats["images_converted"],
        stats["total_images"],
        stats["total_annotations"],
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ATRW Pose to YOLO11-Pose format")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\all projects\data"),
        help="Path to extracted ATRW dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tiger_pose_dataset"),
        help="Target directory for converted dataset",
    )
    args = parser.parse_args()

    data_root = args.data_root
    train_anno = data_root / "atrw_anno_pose_train" / "keypoint_train.json"
    val_anno = data_root / "atrw_anno_pose_train" / "keypoint_val.json"

    train_imgs = data_root / "atrw_pose_train" / "train"
    if not train_imgs.exists():
        train_imgs = data_root / "atrw_pose_train"

    val_imgs = data_root / "atrw_pose_val" / "val"
    if not val_imgs.exists():
        val_imgs = data_root / "atrw_pose_val"

    out_dataset = args.output_dir
    train_out_imgs = out_dataset / "images" / "train"
    train_out_lbls = out_dataset / "labels" / "train"
    val_out_imgs = out_dataset / "images" / "val"
    val_out_lbls = out_dataset / "labels" / "val"

    print("=" * 60)
    print("  TERA-STRIPE ATRW -> YOLO11-Pose Dataset Converter")
    print("=" * 60)
    print(f"Data Root    : {data_root}")
    print(f"Output Dataset: {out_dataset.resolve()}")
    print("=" * 60)

    train_stats = convert_split(
        "train",
        train_anno,
        train_imgs,
        train_out_imgs,
        train_out_lbls,
        copy_images=True,
    )

    val_stats = convert_split(
        "val",
        val_anno,
        val_imgs,
        val_out_imgs,
        val_out_lbls,
        copy_images=True,
    )

    print("\n" + "=" * 60)
    print("  CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Train: {train_stats['images_converted']} images converted ({train_stats['total_annotations']} tiger labels)")
    print(f"Val  : {val_stats['images_converted']} images converted ({val_stats['total_annotations']} tiger labels)")
    print(f"Destination: {out_dataset.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
