"""
TERA-STRIPE YOLO11-Pose Tiger Training Script
==============================================
Fine-tunes YOLO11n-Pose on the Amur Tiger Re-identification in the Wild (ATRW)
dataset with 6 anatomical landmarks on the local NVIDIA GPU (RTX 4050, 6GB VRAM).

Parameters:
  - Device: cuda:0 (RTX 4050)
  - Precision: AMP / FP16 (half=True)
  - Epochs: 25
  - Batch size: 16
  - Output weights: weights/yolo11_pose_tiger.pt
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_tiger_pose")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO11-Pose on Tiger Dataset")
    parser.add_argument(
        "--data",
        type=str,
        default="data/tiger_pose.yaml",
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-pose.pt",
        help="Base pretrained YOLO pose model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="CUDA device index or cpu",
    )
    parser.add_argument(
        "--output-weights",
        type=Path,
        default=Path("weights/yolo11_pose_tiger.pt"),
        help="Destination path for best weights",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  TERA-STRIPE YOLO11-Pose Tiger Training Pipeline")
    print("=" * 60)
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name       : {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM        : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Target Device     : {args.device}")
    print(f"Dataset Config    : {args.data}")
    print(f"Base Model        : {args.model}")
    print(f"Epochs            : {args.epochs}")
    print(f"Batch Size        : {args.batch}")
    print(f"Image Size        : {args.imgsz}")
    print(f"AMP (FP16)        : Enabled")
    print("=" * 60)

    # Initialize model
    logger.info("Initializing base model: %s", args.model)
    model = YOLO(args.model)

    start_time = time.time()

    # Train model
    logger.info("Starting training on %s...", args.device)
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        amp=True,
        plots=True,
        save=True,
        project="runs/pose",
        name="tiger_yolo11_pose",
        exist_ok=True,
        verbose=True,
        workers=2,
    )

    elapsed = time.time() - start_time
    logger.info("Training completed in %.1f seconds (%.2f minutes)", elapsed, elapsed / 60.0)

    # Copy best weights
    best_candidates = list(Path("runs/pose").glob("**/weights/best.pt"))
    if best_candidates:
        best_pt = best_candidates[-1]
    else:
        last_candidates = list(Path("runs/pose").glob("**/weights/last.pt"))
        best_pt = last_candidates[-1] if last_candidates else Path("weights/yolo11_pose_tiger.pt")

    if best_pt.exists():
        args.output_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, args.output_weights)
        logger.info("Saved best model weights to: %s (%.2f MB)", args.output_weights, args.output_weights.stat().st_size / (1024**2))
    else:
        logger.warning("Could not find best.pt in runs/pose/")

    # Validate model
    print("\n" + "=" * 60)
    print("  VALIDATING TRAINED MODEL")
    print("=" * 60)
    val_model = YOLO(str(args.output_weights))
    val_results = val_model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        half=True,
    )

    print("\n" + "=" * 60)
    print("  TRAINING & VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Weights Exported : {args.output_weights.resolve()}")
    if hasattr(val_results, "pose") and val_results.pose is not None:
        print(f"Pose mAP50       : {val_results.pose.map50:.4f}")
        print(f"Pose mAP50-95    : {val_results.pose.map:.4f}")
    if hasattr(val_results, "box") and val_results.box is not None:
        print(f"Box mAP50        : {val_results.box.map50:.4f}")
        print(f"Box mAP50-95     : {val_results.box.map:.4f}")
    print("=" * 60)

    # Clean up VRAM
    del model
    del val_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
