"""
TERA-STRIPE Module 4 -- YOLO11-Pose Flank Extraction & Keypoint Engine
========================================================================
Detects tiger body keypoints using YOLO11-Pose, determines flank
laterality, applies affine warp to produce normalised 224x224 crops
suitable for the DINOv2 + ArcFace re-identification engine (M5).

Pipeline position: Stage 2 of the sequential VRAM execution model.
VRAM budget: < 1.5 GB (FP16), loaded AFTER M2 MegaDetector unloads.

Data contract:
  Input  : triage_result.json  (from M2, FAUNA_DETECTED frames only)
  Output : flank_extraction.json + normalised crops in crops/ directory

CLI Usage
---------
  python -m src.m4_flank_pose \\
      --triage-result ./data/manifests/triage_STN_104B.json \\
      --weights ./weights/yolo11_pose_tiger.pt \\
      --crops-dir ./data/active_working_set/crops \\
      --crop-size 224 \\
      --device cuda:0

Reference: Master Context Packet -- Stage 2, Affine Warp, Quality Score
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageFilter
from pydantic import BaseModel, Field

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("tera_stripe.m4_flank_pose")

# ── Tiger Keypoint Definitions ──────────────────────────────────
TERA_STRIPE_6_KEYPOINT_NAMES: list[str] = [
    "shoulder_scapula",
    "hip_pelvis_root",
    "spine_midpoint",
    "ventral_belly_contour",
    "foreleg_root",
    "hindleg_root",
]

# 18-point animal pose (legacy/mock)
TIGER_KEYPOINT_NAMES: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "tail_base",
]

# Default crop size for DINOv2 input
DEFAULT_CROP_SIZE = 224

# Quality score weights
QW_KEYPOINT_CONF = 0.35
QW_SHARPNESS = 0.30
QW_VISIBILITY = 0.20
QW_LATERALITY = 0.15


# =====================================================================
#  Pydantic Contract Models -- flank_extraction.json
# =====================================================================

class Keypoint(BaseModel):
    """A single detected keypoint on the tiger body."""
    name: str
    x: float = Field(..., ge=0.0, le=1.0, description="Normalised x coord")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalised y coord")
    confidence: float = Field(..., ge=0.0, le=1.0)


class FlankExtraction(BaseModel):
    """Per-image flank extraction result."""
    image_id: str
    source_path: str
    crop_path: str
    flank_side: Literal["LEFT", "RIGHT", "AMBIGUOUS"]
    keypoints: list[Keypoint]
    quality_score: float = Field(..., ge=0.0, le=1.0)
    bbox_normalized: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="[x1, y1, x2, y2] normalised animal bbox",
    )
    crop_size_px: int = DEFAULT_CROP_SIZE
    affine_warp_applied: bool = False


class FlankExtractionSummary(BaseModel):
    """Aggregate extraction statistics."""
    total_fauna_frames: int
    total_extractions: int
    high_quality_count: int = 0  # quality_score >= 0.7
    left_flanks: int = 0
    right_flanks: int = 0
    ambiguous_flanks: int = 0
    mean_quality_score: float = 0.0


class FlankExtractionResult(BaseModel):
    """Complete output conforming to flank_extraction.json contract."""
    batch_id: str
    crops_directory: str
    summary: FlankExtractionSummary
    extractions: list[FlankExtraction]


# =====================================================================
#  Quality Scoring
# =====================================================================

def compute_sharpness(image: Image.Image) -> float:
    """
    Estimate image sharpness using Laplacian variance.

    Returns a 0..1 score where 1 = very sharp, 0 = very blurry.
    """
    grey = image.convert("L")
    laplacian = grey.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
        scale=1,
        offset=128,
    ))
    arr = np.array(laplacian, dtype=np.float32) - 128.0
    variance = float(np.var(arr))

    # Map variance to 0..1 (empirical range: 0-2000 for camera traps)
    return min(1.0, variance / 800.0)


def compute_quality_score(
    keypoints: list[Keypoint],
    crop: Image.Image,
    flank_side: str,
) -> float:
    """
    Compute composite quality score for a flank extraction.

    Components (weighted):
      - Keypoint confidence average   (35%)
      - Image sharpness               (30%)
      - Keypoint visibility ratio     (20%)
      - Laterality clarity            (15%)
    """
    # Keypoint confidence
    if keypoints:
        conf_avg = sum(k.confidence for k in keypoints) / len(keypoints)
    else:
        conf_avg = 0.0

    # Sharpness
    sharpness = compute_sharpness(crop)

    # Visibility ratio (keypoints with confidence > 0.3)
    if keypoints:
        visible = sum(1 for k in keypoints if k.confidence > 0.3)
        visibility = visible / len(keypoints)
    else:
        visibility = 0.0

    # Laterality score
    laterality = 1.0 if flank_side in ("LEFT", "RIGHT") else 0.5

    score = (
        QW_KEYPOINT_CONF * conf_avg
        + QW_SHARPNESS * sharpness
        + QW_VISIBILITY * visibility
        + QW_LATERALITY * laterality
    )
    return round(min(1.0, max(0.0, score)), 4)


# =====================================================================
#  Flank Side Determination
# =====================================================================

def determine_flank_side(keypoints: list[Keypoint]) -> str:
    """
    Determine which flank is visible based on keypoint visibility / orientation.

    Logic:
      1. If 6 TERA-STRIPE landmarks are present (shoulder_scapula, hip_pelvis_root):
         Shoulder to the right of hip -> Tiger facing right -> RIGHT flank visible.
         Shoulder to the left of hip -> Tiger facing left -> LEFT flank visible.
      2. If bilateral keypoints are present (left_shoulder, right_shoulder, etc.):
         Compare visibility of left-side vs right-side keypoints.
    """
    kp_map = {k.name: k for k in keypoints}

    # 1. Check 6-landmark orientation
    for s_name, h_name in [
        ("shoulder_scapula", "hip_pelvis_root"),
        ("shoulder", "hip"),
    ]:
        if s_name in kp_map and h_name in kp_map:
            sh = kp_map[s_name]
            hp = kp_map[h_name]
            if sh.confidence > 0.15 and hp.confidence > 0.15:
                dx = sh.x - hp.x
                if abs(dx) < 0.05:
                    return "AMBIGUOUS"
                elif dx > 0:
                    return "RIGHT"
                else:
                    return "LEFT"

    # 2. Check bilateral keypoint visibility
    left_names = [
        "left_shoulder", "left_hip", "left_knee",
        "left_elbow", "left_wrist", "left_ankle",
    ]
    right_names = [
        "right_shoulder", "right_hip", "right_knee",
        "right_elbow", "right_wrist", "right_ankle",
    ]

    left_score = sum(
        kp_map[n].confidence for n in left_names if n in kp_map
    )
    right_score = sum(
        kp_map[n].confidence for n in right_names if n in kp_map
    )

    diff = abs(left_score - right_score)
    if diff < 0.3:
        return "AMBIGUOUS"
    elif left_score > right_score:
        # Left keypoints more visible -> camera sees RIGHT flank
        return "RIGHT"
    else:
        return "LEFT"


# =====================================================================
#  Affine Warp
# =====================================================================

def affine_warp_flank(
    image: Image.Image,
    keypoints: list[Keypoint],
    crop_size: int = DEFAULT_CROP_SIZE,
) -> tuple[Image.Image, bool]:
    """
    Apply affine warp to normalise the flank orientation.

    Uses shoulder and hip keypoints to define the body axis, then
    warps so the spine is vertical and the flank fills the crop.

    Parameters
    ----------
    image : PIL.Image
        Full-resolution source image.
    keypoints : list[Keypoint]
        Detected keypoints with normalised coordinates.
    crop_size : int
        Output crop dimensions (square).

    Returns
    -------
    tuple[Image.Image, bool]
        (warped_crop, warp_applied)
    """
    w, h = image.size
    kp_map = {k.name: k for k in keypoints if k.confidence > 0.2}

    # Try to find shoulder-hip pairs for affine transform
    shoulder = None
    hip = None

    for s_name, h_name in [
        ("shoulder_scapula", "hip_pelvis_root"),
        ("shoulder", "hip"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
    ]:
        if s_name in kp_map and h_name in kp_map:
            shoulder = kp_map[s_name]
            hip = kp_map[h_name]
            break

    if shoulder is None or hip is None:
        # Fallback: simple centre crop
        return _simple_crop(image, keypoints, crop_size), False

    # Compute body axis angle
    sx, sy = shoulder.x * w, shoulder.y * h
    hx, hy = hip.x * w, hip.y * h
    angle = math.degrees(math.atan2(hy - sy, hx - sx)) - 90.0

    # Compute centre of the body
    cx = (sx + hx) / 2.0
    cy = (sy + hy) / 2.0

    # Body length (shoulder to hip distance)
    body_len = math.sqrt((hx - sx) ** 2 + (hy - sy) ** 2)
    if body_len < 10:
        return _simple_crop(image, keypoints, crop_size), False

    # Pad factor for including surrounding context
    pad = body_len * 0.4

    # Rotate image to align body axis vertically
    rotated = image.rotate(
        -angle,
        center=(cx, cy),
        expand=False,
        resample=Image.BILINEAR,
    )

    # Crop around the body centre
    left = max(0, int(cx - body_len / 2 - pad))
    upper = max(0, int(cy - body_len / 2 - pad))
    right = min(w, int(cx + body_len / 2 + pad))
    lower = min(h, int(cy + body_len / 2 + pad))

    cropped = rotated.crop((left, upper, right, lower))
    resized = cropped.resize((crop_size, crop_size), Image.LANCZOS)

    return resized, True


def _simple_crop(
    image: Image.Image,
    keypoints: list[Keypoint],
    crop_size: int,
) -> Image.Image:
    """Fallback: crop around the centroid of visible keypoints."""
    w, h = image.size
    visible = [k for k in keypoints if k.confidence > 0.2]

    if visible:
        cx = sum(k.x for k in visible) / len(visible) * w
        cy = sum(k.y for k in visible) / len(visible) * h
    else:
        cx, cy = w / 2, h / 2

    # Square crop centred on centroid
    half = min(w, h) * 0.4
    left = max(0, int(cx - half))
    upper = max(0, int(cy - half))
    right = min(w, int(cx + half))
    lower = min(h, int(cy + half))

    cropped = image.crop((left, upper, right, lower))
    return cropped.resize((crop_size, crop_size), Image.LANCZOS)


# =====================================================================
#  Pose Backend Interface
# =====================================================================

class PoseBackend(ABC):
    """Abstract interface for pose estimation backends."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights."""

    @abstractmethod
    def predict_batch(
        self, image_paths: list[Path]
    ) -> list[list[dict]]:
        """
        Run pose estimation on a batch.

        Returns
        -------
        list[list[dict]]
            Outer: per image. Inner: per detected animal.
            Each dict has 'bbox': [x1,y1,x2,y2] normalised,
            'keypoints': list of Keypoint dicts.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend name."""


# =====================================================================
#  Production Backend -- YOLO11-Pose
# =====================================================================

class YOLOPoseBackend(PoseBackend):
    """
    Production YOLO11-Pose inference via ultralytics.

    Detects animal bounding boxes and 17/18 keypoints per instance.
    """

    def __init__(
        self,
        weights_path: Path,
        device: str = "cuda:0",
        img_size: int = 640,
        confidence_threshold: float = 0.25,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.device = device
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self._model: Any = None

    @property
    def backend_name(self) -> str:
        return f"YOLO11-Pose ({self.weights_path.name})"

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics required for YOLO11-Pose. "
                "Install: pip install ultralytics"
            ) from exc

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Pose weights not found: {self.weights_path}"
            )

        logger.info("Loading YOLO11-Pose: %s", self.weights_path.name)
        self._model = YOLO(str(self.weights_path))

        try:
            import torch
            if "cuda" in self.device and torch.cuda.is_available():
                self._model.to(self.device)
        except ImportError:
            pass

    def predict_batch(
        self, image_paths: list[Path]
    ) -> list[list[dict]]:
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        results = self._model.predict(
            source=[str(p) for p in image_paths],
            imgsz=self.img_size,
            conf=self.confidence_threshold,
            device=self.device,
            half=True,
            verbose=False,
        )

        batch_output: list[list[dict]] = []
        for result in results:
            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    bbox = box.xyxyn.cpu().numpy().flatten().tolist()

                    kps = []
                    if (
                        result.keypoints is not None
                        and i < len(result.keypoints)
                    ):
                        kp_data = result.keypoints[i]
                        xy = kp_data.xyn.cpu().numpy()
                        conf = kp_data.conf.cpu().numpy() if kp_data.conf is not None else None

                        if len(xy[0]) == 6:
                            kp_names = TERA_STRIPE_6_KEYPOINT_NAMES
                        else:
                            kp_names = TIGER_KEYPOINT_NAMES

                        for j in range(len(xy[0])):
                            name = kp_names[j] if j < len(kp_names) else f"kpt_{j}"
                            k_conf = float(conf[0][j]) if (conf is not None and len(conf.shape) > 1 and j < len(conf[0])) else 1.0
                            kps.append({
                                "name": name,
                                "x": float(xy[0][j][0]),
                                "y": float(xy[0][j][1]),
                                "confidence": k_conf,
                            })

                    detections.append({
                        "bbox": [round(v, 4) for v in bbox],
                        "keypoints": kps,
                    })
            batch_output.append(detections)

        return batch_output

    def unload(self) -> None:
        logger.info("Unloading YOLO11-Pose...")
        if self._model is not None:
            del self._model
            self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()
        logger.info("YOLO11-Pose unloaded.")


# =====================================================================
#  Heuristic Mock Backend
# =====================================================================

class MockPoseBackend(PoseBackend):
    """
    Deterministic mock pose estimator for testing without GPU/weights.

    Generates synthetic keypoints at anatomically plausible positions
    relative to a generated bounding box, seeded by filename.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._loaded = False

    @property
    def backend_name(self) -> str:
        return "MockPose (no GPU)"

    def load(self) -> None:
        self._loaded = True
        logger.info("Mock pose backend loaded (seed=%d).", self.seed)

    def predict_batch(
        self, image_paths: list[Path]
    ) -> list[list[dict]]:
        if not self._loaded:
            raise RuntimeError("Mock pose backend not loaded.")

        batch_output: list[list[dict]] = []
        for path in image_paths:
            rng = random.Random(hash(path.stem) ^ self.seed)

            # Generate one animal detection per image
            x1 = rng.uniform(0.10, 0.30)
            y1 = rng.uniform(0.15, 0.30)
            x2 = rng.uniform(0.65, 0.90)
            y2 = rng.uniform(0.65, 0.90)
            bbox = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]

            bw = x2 - x1
            bh = y2 - y1

            # Generate keypoints at plausible relative offsets
            kp_offsets = {
                "nose": (0.5, 0.05),
                "left_eye": (0.40, 0.08),
                "right_eye": (0.60, 0.08),
                "left_ear": (0.35, 0.02),
                "right_ear": (0.65, 0.02),
                "left_shoulder": (0.30, 0.25),
                "right_shoulder": (0.70, 0.25),
                "left_elbow": (0.25, 0.50),
                "right_elbow": (0.75, 0.50),
                "left_wrist": (0.20, 0.70),
                "right_wrist": (0.80, 0.70),
                "left_hip": (0.35, 0.55),
                "right_hip": (0.65, 0.55),
                "left_knee": (0.30, 0.75),
                "right_knee": (0.70, 0.75),
                "left_ankle": (0.25, 0.90),
                "right_ankle": (0.75, 0.90),
                "tail_base": (0.50, 0.60),
            }

            keypoints = []
            for name in TIGER_KEYPOINT_NAMES:
                ox, oy = kp_offsets.get(name, (0.5, 0.5))
                # Add small jitter
                jx = rng.gauss(0, 0.02)
                jy = rng.gauss(0, 0.02)
                kx = x1 + (ox + jx) * bw
                ky = y1 + (oy + jy) * bh
                kx = max(0.0, min(1.0, kx))
                ky = max(0.0, min(1.0, ky))
                # Confidence: higher for core body keypoints
                base_conf = 0.85 if "shoulder" in name or "hip" in name else 0.70
                conf = round(
                    max(0.1, min(1.0, base_conf + rng.gauss(0, 0.1))),
                    4,
                )
                keypoints.append({
                    "name": name,
                    "x": round(kx, 4),
                    "y": round(ky, 4),
                    "confidence": conf,
                })

            batch_output.append([{"bbox": bbox, "keypoints": keypoints}])

        return batch_output

    def unload(self) -> None:
        self._loaded = False
        logger.info("Mock pose backend unloaded.")


# =====================================================================
#  Backend Factory
# =====================================================================

def create_pose_backend(
    weights_path: Path | None = None,
    device: str = "cuda:0",
    force_mock: bool = False,
) -> PoseBackend:
    """Create the appropriate pose estimation backend."""
    if force_mock:
        return MockPoseBackend()

    if weights_path and Path(weights_path).exists():
        try:
            import ultralytics  # noqa: F401
            return YOLOPoseBackend(weights_path=Path(weights_path), device=device)
        except ImportError:
            logger.warning("ultralytics not installed. Using mock pose backend.")

    logger.warning(
        "Pose weights not found at '%s'. Using mock backend.", weights_path
    )
    return MockPoseBackend()


# =====================================================================
#  Flank Extraction Engine
# =====================================================================

class FlankExtractionEngine:
    """
    Orchestrates the full flank extraction pipeline.

    1. Filter triage result to FAUNA_DETECTED frames
    2. Load pose backend (sequential VRAM slot)
    3. Run pose estimation in batches
    4. For each detection: determine flank side, affine warp, crop, score
    5. Save crops to disk
    6. Unload backend, free VRAM
    7. Produce flank_extraction.json
    """

    def __init__(
        self,
        backend: PoseBackend,
        crops_dir: Path,
        crop_size: int = DEFAULT_CROP_SIZE,
        batch_size: int = 16,
    ) -> None:
        self.backend = backend
        self.crops_dir = Path(crops_dir)
        self.crop_size = crop_size
        self.batch_size = batch_size

    def process_triage(
        self,
        triage_data: dict,
        source_records: list[dict] | None = None,
    ) -> FlankExtractionResult:
        """
        Run flank extraction on all FAUNA_DETECTED frames.

        Parameters
        ----------
        triage_data : dict
            Parsed triage_result.json.
        source_records : list[dict], optional
            Manifest records with absolute_path info.
        """
        batch_id = triage_data["batch_id"]
        dispatches = triage_data.get("dispatches", [])

        # Filter to fauna frames
        fauna = [d for d in dispatches if d["status"] == "FAUNA_DETECTED"]
        logger.info(
            "Flank extraction | batch=%s | fauna_frames=%d | backend=%s",
            batch_id, len(fauna), self.backend.backend_name,
        )

        # Build image_id -> path lookup
        path_lookup = {}
        if source_records:
            path_lookup = {
                r["image_id"]: r["absolute_path"] for r in source_records
            }

        # Also check triage dispatches for embedded source info
        for d in dispatches:
            if "source_path" in d and d["image_id"] not in path_lookup:
                path_lookup[d["image_id"]] = d["source_path"]

        # Set up crops directory
        batch_crops_dir = self.crops_dir / batch_id
        batch_crops_dir.mkdir(parents=True, exist_ok=True)

        # ── Load backend ──
        t_start = time.time()
        self.backend.load()

        # ── Batch processing ──
        extractions: list[FlankExtraction] = []

        fauna_ids = [d["image_id"] for d in fauna]
        fauna_paths = []
        valid_ids = []

        for img_id in fauna_ids:
            img_path = path_lookup.get(img_id)
            if img_path and Path(img_path).exists():
                fauna_paths.append(Path(img_path))
                valid_ids.append(img_id)
            else:
                logger.warning(
                    "Image path not found for %s, skipping.", img_id
                )

        for batch_start in range(0, len(fauna_paths), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(fauna_paths))
            b_paths = fauna_paths[batch_start:batch_end]
            b_ids = valid_ids[batch_start:batch_end]

            pose_results = self.backend.predict_batch(b_paths)

            for img_id, img_path, detections in zip(
                b_ids, b_paths, pose_results
            ):
                if not detections:
                    continue

                # Process the primary detection (highest bbox area)
                primary = max(
                    detections,
                    key=lambda d: (
                        (d["bbox"][2] - d["bbox"][0])
                        * (d["bbox"][3] - d["bbox"][1])
                    ),
                )

                # Parse keypoints
                kps = [Keypoint(**kd) for kd in primary["keypoints"]]

                # Determine flank side
                flank_side = determine_flank_side(kps)

                # Load image and extract crop
                try:
                    source_img = Image.open(img_path).convert("RGB")
                except Exception as exc:
                    logger.error("Cannot open %s: %s", img_path, exc)
                    continue

                crop, warp_applied = affine_warp_flank(
                    source_img, kps, self.crop_size
                )

                # Save crop
                crop_filename = f"{img_id}_{flank_side}.jpg"
                crop_path = batch_crops_dir / crop_filename
                crop.save(str(crop_path), "JPEG", quality=95)

                # Quality score
                q_score = compute_quality_score(kps, crop, flank_side)

                extractions.append(
                    FlankExtraction(
                        image_id=img_id,
                        source_path=str(img_path),
                        crop_path=str(crop_path),
                        flank_side=flank_side,
                        keypoints=kps,
                        quality_score=q_score,
                        bbox_normalized=primary["bbox"],
                        crop_size_px=self.crop_size,
                        affine_warp_applied=warp_applied,
                    )
                )

        # ── Unload backend (VRAM discipline) ──
        self.backend.unload()
        t_elapsed = time.time() - t_start

        # ── Build summary ──
        high_q = sum(1 for e in extractions if e.quality_score >= 0.7)
        lefts = sum(1 for e in extractions if e.flank_side == "LEFT")
        rights = sum(1 for e in extractions if e.flank_side == "RIGHT")
        ambi = sum(1 for e in extractions if e.flank_side == "AMBIGUOUS")
        mean_q = (
            sum(e.quality_score for e in extractions) / len(extractions)
            if extractions
            else 0.0
        )

        summary = FlankExtractionSummary(
            total_fauna_frames=len(fauna),
            total_extractions=len(extractions),
            high_quality_count=high_q,
            left_flanks=lefts,
            right_flanks=rights,
            ambiguous_flanks=ambi,
            mean_quality_score=round(mean_q, 4),
        )

        logger.info(
            "Flank extraction complete | %.1fs | crops=%d (L=%d R=%d A=%d) "
            "| mean_q=%.3f | high_q=%d",
            t_elapsed,
            len(extractions),
            lefts,
            rights,
            ambi,
            mean_q,
            high_q,
        )

        return FlankExtractionResult(
            batch_id=batch_id,
            crops_directory=str(batch_crops_dir),
            summary=summary,
            extractions=extractions,
        )


# =====================================================================
#  CLI Entry Point
# =====================================================================

def main() -> None:
    """CLI entry point for the flank extraction engine."""
    parser = argparse.ArgumentParser(
        prog="m4_flank_pose",
        description="TERA-STRIPE M4 -- YOLO11-Pose Flank Extraction",
    )
    parser.add_argument(
        "--triage-result", type=Path, required=True,
        help="Path to triage_result.json (M2 output).",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Path to ingest_manifest.json (for source paths).",
    )
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="Path to YOLO11-Pose .pt weights.",
    )
    parser.add_argument(
        "--crops-dir", type=Path, default=Path("./data/active_working_set/crops"),
        help="Output directory for normalised flank crops.",
    )
    parser.add_argument(
        "--crop-size", type=int, default=DEFAULT_CROP_SIZE,
        help=f"Crop dimensions in pixels (default: {DEFAULT_CROP_SIZE}).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for flank_extraction.json.",
    )
    parser.add_argument(
        "--force-mock", action="store_true",
        help="Force mock backend (no GPU/weights).",
    )
    args = parser.parse_args()

    # Load triage result
    if not args.triage_result.exists():
        logger.error("Triage result not found: %s", args.triage_result)
        sys.exit(1)
    with open(args.triage_result, "r", encoding="utf-8") as f:
        triage_data = json.load(f)

    # Load manifest for source paths
    source_records = None
    if args.manifest and args.manifest.exists():
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        source_records = manifest.get("records", [])

    # Create backend
    backend = create_pose_backend(
        weights_path=args.weights,
        device=args.device,
        force_mock=args.force_mock,
    )

    # Run extraction
    engine = FlankExtractionEngine(
        backend=backend,
        crops_dir=args.crops_dir,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
    )
    result = engine.process_triage(triage_data, source_records)

    # Write output
    if args.output is None:
        args.output = args.triage_result.parent / f"flank_{triage_data['batch_id']}.json"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    s = result.summary
    print(
        f"\n{'='*60}\n"
        f"  TERA-STRIPE M4 Flank Extraction Complete\n"
        f"  Batch      : {result.batch_id}\n"
        f"  Backend    : {backend.backend_name}\n"
        f"  Fauna In   : {s.total_fauna_frames}\n"
        f"  Crops Out  : {s.total_extractions}\n"
        f"  Left/Right : {s.left_flanks} / {s.right_flanks}\n"
        f"  Ambiguous  : {s.ambiguous_flanks}\n"
        f"  High Qual  : {s.high_quality_count}\n"
        f"  Mean Q     : {s.mean_quality_score:.3f}\n"
        f"  Crops Dir  : {result.crops_directory}\n"
        f"  Output     : {args.output}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
