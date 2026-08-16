"""
TERA-STRIPE Module 2 -- MegaDetector Blank Triage Engine
==========================================================
High-recall camera-trap image classifier that separates fauna-bearing
frames from empty/blank triggers using MegaDetector MDv1000-redwood / v5a.

Pipeline position: Stage 1 of the sequential VRAM execution model.
VRAM budget: < 1.8 GB (FP16 TensorRT), batch size 32, >= 50 FPS target.

Data contract:
  Input  : ingest_manifest.json  (from M1)
  Output : triage_result.json    (to M3 / M4)

CLI Usage
---------
  python -m src.m2_triage \\
      --manifest ./data/manifests/manifest_STN_104B.json \\
      --weights ./weights/md_v1000_redwood.pt \\
      --confidence-threshold 0.15 \\
      --quarantine-dir ./data/quarantine \\
      --device cuda:0

Reference: Master Context Packet -- Stage 1, Contract M2->M3/M4
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("tera_stripe.m2_triage")

# ── MegaDetector class mapping ───────────────────────────────────
MD_CLASS_MAP: dict[int, str] = {
    1: "animal",
    2: "person",
    3: "vehicle",
}

# Average manual inspection time per image (NTCA statutory estimate)
MANUAL_SECONDS_PER_IMAGE = 4.5


# =====================================================================
#  Pydantic Contract Models -- triage_result.json
# =====================================================================

class RawDetection(BaseModel):
    """A single bounding-box detection from MegaDetector."""
    model_config = {"populate_by_name": True}

    class_name: Literal["animal", "person", "vehicle"] = Field(
        ..., alias="class", serialization_alias="class",
    )
    confidence: float
    bbox_normalized: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="[x_min, y_min, x_max, y_max] in 0..1 normalised coords",
    )


class TriageDispatch(BaseModel):
    """Per-image triage classification result."""
    image_id: str
    status: Literal["FAUNA_DETECTED", "QUARANTINED_BLANK", "HUMAN_VEHICLE_FLAG"]
    max_confidence: float
    detections: list[RawDetection] = []
    target_quarantine_path: str | None = None


class TriageSummary(BaseModel):
    """Aggregate triage statistics for the batch."""
    processed_frames: int
    blank_frames: int
    fauna_frames: int
    human_vehicle_frames: int = 0
    storage_saved_gb: float
    manual_hours_saved: float


class TriageResult(BaseModel):
    """Complete triage output conforming to the triage_result.json contract."""
    batch_id: str
    triage_summary: TriageSummary
    dispatches: list[TriageDispatch]


# =====================================================================
#  Re-import M1 manifest models
# =====================================================================

def _load_manifest(manifest_path: Path) -> dict:
    """Load and return the raw manifest dictionary."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================================
#  Detector Backend Interface
# =====================================================================

class DetectorBackend(ABC):
    """Abstract interface for object detection backends."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory (GPU or CPU)."""

    @abstractmethod
    def predict_batch(
        self, image_paths: list[Path]
    ) -> list[list[RawDetection]]:
        """
        Run inference on a batch of images.

        Returns
        -------
        list[list[RawDetection]]
            Outer list = one entry per image.
            Inner list = zero or more detections per image.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release model from GPU/CPU memory."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier."""


# =====================================================================
#  Production Backend -- MegaDetector via Ultralytics YOLO
# =====================================================================

class MegaDetectorBackend(DetectorBackend):
    """
    Production MegaDetector inference using the ultralytics YOLO API.

    Loads MegaDetector v5a or MDv1000-redwood .pt checkpoint and runs
    batched FP16 inference with strict VRAM budgeting.
    """

    def __init__(
        self,
        weights_path: Path,
        device: str = "cuda:0",
        batch_size: int = 32,
        img_size: int = 1280,
        confidence_threshold: float = 0.15,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.device = device
        self.batch_size = batch_size
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self._model: Any = None

    @property
    def backend_name(self) -> str:
        return f"MegaDetector ({self.weights_path.name})"

    def load(self) -> None:
        """Load MegaDetector model into GPU memory."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics package required for MegaDetector inference. "
                "Install with: pip install ultralytics"
            ) from exc

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights_path}"
            )

        logger.info(
            "Loading MegaDetector weights: %s -> %s",
            self.weights_path.name,
            self.device,
        )
        self._model = YOLO(str(self.weights_path))

        # Attempt to move to device (GPU if available)
        try:
            import torch

            if "cuda" in self.device and torch.cuda.is_available():
                self._model.to(self.device)
                logger.info(
                    "Model loaded on %s (VRAM: %.1f MB allocated)",
                    self.device,
                    torch.cuda.memory_allocated() / 1024**2,
                )
            else:
                logger.info("CUDA unavailable, running on CPU.")
                self.device = "cpu"
        except ImportError:
            logger.info("torch not available, running with default device.")

    def predict_batch(
        self, image_paths: list[Path]
    ) -> list[list[RawDetection]]:
        """Run MegaDetector inference on a batch of images."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = self._model.predict(
            source=[str(p) for p in image_paths],
            imgsz=self.img_size,
            conf=self.confidence_threshold,
            device=self.device,
            half=True,  # FP16 inference
            verbose=False,
        )

        batch_detections: list[list[RawDetection]] = []
        for result in results:
            frame_dets: list[RawDetection] = []
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls.item()) + 1  # MegaDetector is 1-indexed
                    cls_name = MD_CLASS_MAP.get(cls_id, "animal")
                    conf = float(box.conf.item())
                    # xyxyn = normalised [x1, y1, x2, y2]
                    bbox = box.xyxyn.cpu().numpy().flatten().tolist()

                    frame_dets.append(
                        RawDetection(
                            **{"class": cls_name},
                            confidence=round(conf, 4),
                            bbox_normalized=[round(v, 4) for v in bbox],
                        )
                    )
            batch_detections.append(frame_dets)

        return batch_detections

    def unload(self) -> None:
        """
        Release model from GPU memory per TERA-STRIPE VRAM discipline.

        Sequence: del model -> torch.cuda.empty_cache() -> gc.collect()
        """
        logger.info("Unloading MegaDetector from memory...")
        if self._model is not None:
            del self._model
            self._model = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared.")
        except ImportError:
            pass

        gc.collect()
        logger.info("MegaDetector unloaded. GC complete.")


# =====================================================================
#  Heuristic Mock Backend -- Testing Without GPU/Weights
# =====================================================================

class HeuristicMockBackend(DetectorBackend):
    """
    Deterministic mock detector for testing without model weights.

    Uses a seeded RNG keyed on each image filename to produce
    reproducible detection results across test runs.

    Default behaviour (detection_rate=0.6):
      ~60% of images -> FAUNA_DETECTED
      ~35% of images -> QUARANTINED_BLANK
      ~5%  of images -> HUMAN_VEHICLE_FLAG
    """

    def __init__(
        self,
        detection_rate: float = 0.60,
        human_vehicle_rate: float = 0.05,
        seed: int = 42,
        confidence_threshold: float = 0.15,
    ) -> None:
        self.detection_rate = detection_rate
        self.human_vehicle_rate = human_vehicle_rate
        self.seed = seed
        self.confidence_threshold = confidence_threshold
        self._loaded = False

    @property
    def backend_name(self) -> str:
        return "HeuristicMock (no GPU)"

    def load(self) -> None:
        logger.info("Mock detector loaded (seed=%d).", self.seed)
        self._loaded = True

    def predict_batch(
        self, image_paths: list[Path]
    ) -> list[list[RawDetection]]:
        if not self._loaded:
            raise RuntimeError("Mock detector not loaded.")

        batch_detections: list[list[RawDetection]] = []
        for path in image_paths:
            rng = random.Random(hash(path.stem) ^ self.seed)
            roll = rng.random()

            if roll < self.human_vehicle_rate:
                # Simulate person/vehicle detection
                cls = rng.choice(["person", "vehicle"])
                det = RawDetection(
                    **{"class": cls},
                    confidence=round(rng.uniform(0.30, 0.85), 4),
                    bbox_normalized=[
                        round(rng.uniform(0.05, 0.25), 4),
                        round(rng.uniform(0.05, 0.25), 4),
                        round(rng.uniform(0.55, 0.90), 4),
                        round(rng.uniform(0.55, 0.90), 4),
                    ],
                )
                batch_detections.append([det])

            elif roll < self.detection_rate + self.human_vehicle_rate:
                # Simulate animal detection
                conf = round(rng.uniform(0.20, 0.98), 4)
                det = RawDetection(
                    **{"class": "animal"},
                    confidence=conf,
                    bbox_normalized=[
                        round(rng.uniform(0.10, 0.30), 4),
                        round(rng.uniform(0.15, 0.35), 4),
                        round(rng.uniform(0.65, 0.90), 4),
                        round(rng.uniform(0.60, 0.90), 4),
                    ],
                )
                batch_detections.append([det])

            else:
                # No detections -> blank
                batch_detections.append([])

        return batch_detections

    def unload(self) -> None:
        self._loaded = False
        logger.info("Mock detector unloaded.")


# =====================================================================
#  Backend Factory
# =====================================================================

def create_backend(
    weights_path: Path | None = None,
    device: str = "cuda:0",
    batch_size: int = 32,
    confidence_threshold: float = 0.15,
    force_mock: bool = False,
) -> DetectorBackend:
    """
    Create the appropriate detector backend.

    Priority:
      1. If ``force_mock=True`` -> HeuristicMockBackend
      2. If weights exist and ultralytics/torch available -> MegaDetectorBackend
      3. Fallback -> HeuristicMockBackend with warning
    """
    if force_mock:
        logger.info("Mock backend forced by configuration.")
        return HeuristicMockBackend(
            confidence_threshold=confidence_threshold,
        )

    if weights_path and Path(weights_path).exists():
        try:
            import ultralytics  # noqa: F401
            return MegaDetectorBackend(
                weights_path=Path(weights_path),
                device=device,
                batch_size=batch_size,
                confidence_threshold=confidence_threshold,
            )
        except ImportError:
            logger.warning(
                "ultralytics not installed. Falling back to mock backend."
            )

    logger.warning(
        "Model weights not found at '%s'. Using heuristic mock backend.",
        weights_path,
    )
    return HeuristicMockBackend(
        confidence_threshold=confidence_threshold,
    )


# =====================================================================
#  Triage Engine Orchestrator
# =====================================================================

class TriageEngine:
    """
    Orchestrates the full blank-triage pipeline.

    1. Load manifest (M1 output)
    2. Load detector backend
    3. Batch inference with VRAM-safe sequential execution
    4. Classify each frame: FAUNA_DETECTED / QUARANTINED_BLANK / HUMAN_VEHICLE_FLAG
    5. Compute ROI telemetry (storage saved, manual hours saved)
    6. Unload detector, free VRAM
    7. Produce triage_result.json
    """

    def __init__(
        self,
        backend: DetectorBackend,
        confidence_threshold: float = 0.15,
        batch_size: int = 32,
        quarantine_dir: Path | None = None,
    ) -> None:
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.quarantine_dir = quarantine_dir

    def _classify_detections(
        self,
        detections: list[RawDetection],
    ) -> Literal["FAUNA_DETECTED", "QUARANTINED_BLANK", "HUMAN_VEHICLE_FLAG"]:
        """
        Apply MegaDetector classification logic.

        Priority:
          1. If any detection is person/vehicle -> HUMAN_VEHICLE_FLAG
          2. If any detection is animal with conf >= threshold -> FAUNA_DETECTED
          3. Otherwise -> QUARANTINED_BLANK
        """
        if not detections:
            return "QUARANTINED_BLANK"

        has_human_vehicle = any(
            d.class_name in ("person", "vehicle") for d in detections
        )
        if has_human_vehicle:
            return "HUMAN_VEHICLE_FLAG"

        has_fauna = any(
            d.class_name == "animal" and d.confidence >= self.confidence_threshold
            for d in detections
        )
        if has_fauna:
            return "FAUNA_DETECTED"

        return "QUARANTINED_BLANK"

    def _build_quarantine_path(
        self,
        batch_id: str,
        image_id: str,
        original_path: str,
    ) -> str | None:
        """Compute the quarantine destination path for a blank frame."""
        if self.quarantine_dir is None:
            return None
        ext = Path(original_path).suffix
        return str(
            self.quarantine_dir / batch_id / f"{image_id}{ext}"
        )

    def process_manifest(
        self,
        manifest_data: dict,
    ) -> TriageResult:
        """
        Run the full triage pipeline on a loaded manifest.

        Parameters
        ----------
        manifest_data : dict
            Parsed ingest manifest (M1 output).

        Returns
        -------
        TriageResult
        """
        batch_id = manifest_data["batch_id"]
        records = manifest_data["records"]

        logger.info(
            "Starting triage | batch=%s | frames=%d | backend=%s",
            batch_id,
            len(records),
            self.backend.backend_name,
        )

        # ── Stage 1: Load detector ──
        t_start = time.time()
        self.backend.load()

        # ── Stage 2: Batch inference ──
        dispatches: list[TriageDispatch] = []
        all_paths = [Path(r["absolute_path"]) for r in records]
        all_ids = [r["image_id"] for r in records]

        for batch_start in range(0, len(all_paths), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(all_paths))
            batch_paths = all_paths[batch_start:batch_end]
            batch_ids = all_ids[batch_start:batch_end]

            # Validate paths exist
            valid_paths = []
            valid_ids = []
            for p, img_id in zip(batch_paths, batch_ids):
                if p.exists():
                    valid_paths.append(p)
                    valid_ids.append(img_id)
                else:
                    logger.warning("Image not found, skipping: %s", p)
                    dispatches.append(
                        TriageDispatch(
                            image_id=img_id,
                            status="QUARANTINED_BLANK",
                            max_confidence=0.0,
                            detections=[],
                        )
                    )

            if not valid_paths:
                continue

            # Run detection
            batch_results = self.backend.predict_batch(valid_paths)

            for img_id, dets, orig_path in zip(
                valid_ids, batch_results, valid_paths
            ):
                status = self._classify_detections(dets)
                max_conf = max((d.confidence for d in dets), default=0.0)

                quarantine_path = None
                if status == "QUARANTINED_BLANK":
                    quarantine_path = self._build_quarantine_path(
                        batch_id, img_id, str(orig_path)
                    )

                dispatches.append(
                    TriageDispatch(
                        image_id=img_id,
                        status=status,
                        max_confidence=round(max_conf, 4),
                        detections=dets,
                        target_quarantine_path=quarantine_path,
                    )
                )

        # ── Stage 3: Unload detector (VRAM discipline) ──
        self.backend.unload()

        t_elapsed = time.time() - t_start

        # ── Stage 4: Compute ROI telemetry ──
        blank_count = sum(
            1 for d in dispatches if d.status == "QUARANTINED_BLANK"
        )
        fauna_count = sum(
            1 for d in dispatches if d.status == "FAUNA_DETECTED"
        )
        human_vehicle_count = sum(
            1 for d in dispatches if d.status == "HUMAN_VEHICLE_FLAG"
        )

        # Estimate storage saved: average camera-trap JPEG ~ 3.5 MB
        avg_file_size_mb = 3.5
        # Try to compute real sizes for existing files
        total_blank_bytes = 0
        for d in dispatches:
            if d.status == "QUARANTINED_BLANK":
                rec = next(
                    (r for r in records if r["image_id"] == d.image_id),
                    None,
                )
                if rec:
                    fpath = Path(rec["absolute_path"])
                    if fpath.exists():
                        total_blank_bytes += fpath.stat().st_size
                    else:
                        total_blank_bytes += int(avg_file_size_mb * 1024 * 1024)

        storage_saved_gb = round(total_blank_bytes / (1024**3), 4)

        # Manual hours saved: blank_count * 4.5s / 3600
        manual_hours_saved = round(
            blank_count * MANUAL_SECONDS_PER_IMAGE / 3600.0, 4
        )

        summary = TriageSummary(
            processed_frames=len(dispatches),
            blank_frames=blank_count,
            fauna_frames=fauna_count,
            human_vehicle_frames=human_vehicle_count,
            storage_saved_gb=storage_saved_gb,
            manual_hours_saved=manual_hours_saved,
        )

        logger.info(
            "Triage complete | %.1fs | fauna=%d blank=%d human/vehicle=%d | "
            "saved=%.3f GB / %.2f hrs",
            t_elapsed,
            fauna_count,
            blank_count,
            human_vehicle_count,
            storage_saved_gb,
            manual_hours_saved,
        )

        return TriageResult(
            batch_id=batch_id,
            triage_summary=summary,
            dispatches=dispatches,
        )


# =====================================================================
#  CLI Entry Point
# =====================================================================

def main() -> None:
    """CLI entry point for the MegaDetector triage engine."""
    parser = argparse.ArgumentParser(
        prog="m2_triage",
        description="TERA-STRIPE M2 -- MegaDetector Blank Triage Engine",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the ingest manifest JSON (M1 output).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to MegaDetector .pt weights file.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.15,
        help="Minimum detection confidence for fauna gate (default: 0.15).",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=None,
        help="Directory for quarantined blank frames.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Compute device (default: cuda:0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for triage_result.json. Defaults to manifests dir.",
    )
    parser.add_argument(
        "--force-mock",
        action="store_true",
        help="Force heuristic mock backend (no GPU/weights needed).",
    )
    args = parser.parse_args()

    # ── Load manifest ──
    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest)
        sys.exit(1)

    manifest_data = _load_manifest(args.manifest)

    # ── Create backend ──
    backend = create_backend(
        weights_path=args.weights,
        device=args.device,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        force_mock=args.force_mock,
    )

    # ── Run triage ──
    engine = TriageEngine(
        backend=backend,
        confidence_threshold=args.confidence_threshold,
        batch_size=args.batch_size,
        quarantine_dir=args.quarantine_dir,
    )

    result = engine.process_manifest(manifest_data)

    # ── Write output ──
    if args.output is None:
        args.output = args.manifest.parent / f"triage_{manifest_data['batch_id']}.json"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            result.model_dump(by_alias=True),
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ── Summary ──
    s = result.triage_summary
    print(
        f"\n{'='*60}\n"
        f"  TERA-STRIPE M2 Triage Complete\n"
        f"  Batch       : {result.batch_id}\n"
        f"  Backend     : {backend.backend_name}\n"
        f"  Processed   : {s.processed_frames}\n"
        f"  Fauna       : {s.fauna_frames}\n"
        f"  Blank       : {s.blank_frames}\n"
        f"  Human/Veh   : {s.human_vehicle_frames}\n"
        f"  Storage Save: {s.storage_saved_gb:.4f} GB\n"
        f"  Hours Saved : {s.manual_hours_saved:.2f} hrs\n"
        f"  Output      : {args.output}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
