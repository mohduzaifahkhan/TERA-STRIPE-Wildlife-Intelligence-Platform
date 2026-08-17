"""
TERA-STRIPE -- Real AI Backends
=================================
Production backends that replace mock/heuristic backends with
actual model inference on GPU.

Backends:
  MegaDetectorBackend  -- MegaDetector v5a for animal detection (M2 Triage)
  RealPoseBackend      -- MegaDetector bbox + crop extraction (M4 Flank)
  RealEmbeddingBackend -- DINOv2 ViT-B/14 for flank embeddings (M5 Re-ID)

Requirements:
  pip install torch torchvision ultralytics transformers

Hardware:
  NVIDIA GPU with >= 4GB VRAM (tested on RTX 4050 6GB)
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("tera_stripe.real_backends")


# =====================================================================
#  M2: MegaDetector Backend (Animal Detection / Triage)
# =====================================================================

class MegaDetectorBackend:
    """
    Real MegaDetector v5a backend for camera trap triage.

    Uses Ultralytics YOLOv5 under the hood. Classifies images as
    FAUNA (animal detected) or BLANK (no animal).

    MegaDetector classes: 0=animal, 1=person, 2=vehicle
    """

    MODEL_URL = "https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt"

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "auto",
        confidence_threshold: float = 0.15,
    ) -> None:
        self.model_path = model_path or self._default_model_path()
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._device = device

    def _default_model_path(self) -> Path:
        return Path("models/md_v5a.0.0.pt")

    def load(self) -> None:
        """Load MegaDetector model onto GPU using YOLOv5 torch.hub."""
        import torch

        model_path = Path(self.model_path)
        if not model_path.exists():
            logger.info("Downloading MegaDetector v5a...")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            self._download_model(model_path)

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading MegaDetector from %s on %s", model_path, device)
        # MegaDetector v5a is a YOLOv5 model — use torch.hub YOLOv5 loader
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom',
            path=str(model_path.resolve()),
            force_reload=False,
        )
        self.model.conf = self.confidence_threshold
        self.model.to(device)
        logger.info("MegaDetector loaded on %s", device)

    def unload(self) -> None:
        """Release model from GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("MegaDetector unloaded")

    def predict(self, image_path: str, **kwargs) -> dict:
        """
        Run MegaDetector on a single image.

        Returns
        -------
        dict with keys:
          - label: str ("FAUNA", "BLANK", "HUMAN", "VEHICLE")
          - confidence: float (0.0 - 1.0)
          - detections: list of detection dicts
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # YOLOv5 inference — returns a Results object with .pandas() / .xyxy
        results = self.model(image_path)

        detections = []
        best_animal_conf = 0.0
        has_animal = False
        has_person = False

        # YOLOv5 results: results.xyxy[0] is tensor [N, 6] (x1,y1,x2,y2,conf,cls)
        preds = results.xyxy[0].cpu().numpy()

        for row in preds:
            x1, y1, x2, y2, conf, cls_id = row
            conf = float(conf)
            cls_id = int(cls_id)

            # MegaDetector classes: 0=animal, 1=person, 2=vehicle
            label = {0: "animal", 1: "person", 2: "vehicle"}.get(cls_id, "unknown")

            detections.append({
                "class": label,
                "confidence": round(conf, 4),
                "bbox": [round(float(x1), 1), round(float(y1), 1),
                         round(float(x2), 1), round(float(y2), 1)],
            })

            if cls_id == 0 and conf > best_animal_conf:
                best_animal_conf = conf
                has_animal = True
            if cls_id == 1:
                has_person = True

        if has_animal:
            triage_label = "FAUNA"
            triage_conf = best_animal_conf
        elif has_person:
            triage_label = "HUMAN"
            triage_conf = max(d["confidence"] for d in detections if d["class"] == "person")
        elif detections:
            triage_label = "VEHICLE"
            triage_conf = detections[0]["confidence"]
        else:
            triage_label = "BLANK"
            triage_conf = 1.0 - (best_animal_conf if best_animal_conf > 0 else 0.0)

        return {
            "label": triage_label,
            "confidence": round(triage_conf, 4),
            "detections": detections,
        }

    def _download_model(self, dest: Path) -> None:
        """Download MegaDetector weights."""
        import urllib.request
        logger.info("Downloading MegaDetector v5a from GitHub (~280MB)...")
        urllib.request.urlretrieve(self.MODEL_URL, str(dest))
        logger.info("MegaDetector downloaded to %s", dest)


# =====================================================================
#  M4: Real Pose/Crop Backend (Bounding Box Crop)
# =====================================================================

class RealCropBackend:
    """
    Extracts tiger flank crops using MegaDetector bounding boxes.

    Since no pretrained tiger-specific pose model exists, we use
    MegaDetector's animal detection bounding box to crop the animal
    region, then split into left/right flank based on aspect ratio
    heuristics.
    """

    def __init__(self, megadetector: MegaDetectorBackend | None = None) -> None:
        self.megadetector = megadetector
        self._loaded = False

    def load(self) -> None:
        if self.megadetector and self.megadetector.model is None:
            self.megadetector.load()
        self._loaded = True

    def unload(self) -> None:
        # Don't unload megadetector here — it may be shared
        self._loaded = False
        gc.collect()

    def predict(self, image_path: str, **kwargs) -> dict:
        """
        Detect animal bbox and extract crop coordinates.

        Returns dict with:
          - keypoints: simplified bbox corners as keypoints
          - flank_side: estimated flank orientation
          - bbox: [x1, y1, x2, y2]
          - confidence: detection confidence
        """
        from PIL import Image

        img = Image.open(image_path)
        w, h = img.size

        # Get MegaDetector detections
        if self.megadetector and self.megadetector.model:
            md_result = self.megadetector.predict(image_path)
            animal_dets = [d for d in md_result["detections"] if d["class"] == "animal"]
        else:
            animal_dets = []

        if not animal_dets:
            # No animal detected — return empty
            return {
                "keypoints": [],
                "flank_side": "UNKNOWN",
                "bbox": [0, 0, w, h],
                "confidence": 0.0,
                "status": "NO_DETECTION",
            }

        # Use highest-confidence animal detection
        best = max(animal_dets, key=lambda d: d["confidence"])
        x1, y1, x2, y2 = best["bbox"]

        # Pad bbox by 5% for context
        pad_x = (x2 - x1) * 0.05
        pad_y = (y2 - y1) * 0.05
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        bbox = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

        # Estimate flank side from bbox position relative to image center
        bbox_center_x = (x1 + x2) / 2
        if bbox_center_x < w * 0.5:
            flank_side = "LEFT_FLANK"
        else:
            flank_side = "RIGHT_FLANK"

        # Create simplified keypoints (bbox corners + center)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        keypoints = [
            {"name": "top_left", "x": round(x1, 1), "y": round(y1, 1), "confidence": best["confidence"]},
            {"name": "top_right", "x": round(x2, 1), "y": round(y1, 1), "confidence": best["confidence"]},
            {"name": "bottom_left", "x": round(x1, 1), "y": round(y2, 1), "confidence": best["confidence"]},
            {"name": "bottom_right", "x": round(x2, 1), "y": round(y2, 1), "confidence": best["confidence"]},
            {"name": "center", "x": round(cx, 1), "y": round(cy, 1), "confidence": best["confidence"]},
        ]

        return {
            "keypoints": keypoints,
            "flank_side": flank_side,
            "bbox": bbox,
            "confidence": round(best["confidence"], 4),
            "status": "DETECTED",
        }

    def crop_and_save(
        self, image_path: str, bbox: list[float], output_path: str | Path
    ) -> Path:
        """Crop the bbox region from image and save."""
        from PIL import Image

        img = Image.open(image_path)
        x1, y1, x2, y2 = [int(b) for b in bbox]
        cropped = img.crop((x1, y1, x2, y2))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(str(output_path), quality=95)

        return output_path


# =====================================================================
#  M5: DINOv2 Embedding Backend (Tiger Re-ID)
# =====================================================================

class DINOv2Backend:
    """
    Real DINOv2 ViT-B/14 backend for tiger flank embeddings.

    Uses Facebook's DINOv2 (self-supervised vision transformer) to
    generate 768-dimensional embeddings from flank crop images.
    These embeddings enable cosine-similarity based re-identification.

    VRAM usage: ~1.5 GB for ViT-B/14
    """

    def __init__(self, device: str = "auto") -> None:
        self.model = None
        self.processor = None
        self._device = device
        self._actual_device = None

    def load(self) -> None:
        """Load DINOv2 model from HuggingFace."""
        import torch
        from transformers import AutoImageProcessor, AutoModel

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._actual_device = device

        logger.info("Loading DINOv2 ViT-B/14 from HuggingFace (~350MB)...")
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        self.model.eval()

        logger.info("DINOv2 loaded on %s", device)

    def unload(self) -> None:
        """Release model from GPU."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("DINOv2 unloaded")

    def predict(self, image_path: str, **kwargs) -> dict:
        """
        Generate 768-dim embedding for a flank crop image.

        Returns dict with:
          - embedding: list[float] (768 dimensions)
          - embedding_dim: 768
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self._actual_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use CLS token embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

        # Normalize to unit vector for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return {
            "embedding": embedding.tolist(),
            "embedding_dim": len(embedding),
        }


# =====================================================================
#  Sequential Model Lifecycle Manager
# =====================================================================

class ModelLifecycle:
    """
    Manages sequential model loading/unloading to stay within
    6GB VRAM budget on RTX 4050.

    Usage:
        lifecycle = ModelLifecycle()
        lifecycle.run_pipeline(image_dir, output_dir)
    """

    def __init__(self, models_dir: Path = Path("models")) -> None:
        self.models_dir = models_dir
        self.megadetector = MegaDetectorBackend(
            model_path=models_dir / "md_v5a.0.0.pt"
        )
        self.crop_backend = RealCropBackend(megadetector=self.megadetector)
        self.dinov2 = DINOv2Backend()

    def run_triage(self, image_paths: list[str]) -> list[dict]:
        """Run MegaDetector triage on image list."""
        self.megadetector.load()
        results = []
        try:
            for path in image_paths:
                try:
                    result = self.megadetector.predict(path)
                    result["image_path"] = path
                    results.append(result)
                except Exception as e:
                    logger.warning("Triage failed for %s: %s", path, e)
                    results.append({
                        "image_path": path,
                        "label": "ERROR",
                        "confidence": 0.0,
                        "detections": [],
                    })
        finally:
            self.megadetector.unload()
        return results

    def run_crop(self, fauna_images: list[dict], crops_dir: Path) -> list[dict]:
        """Run crop extraction on fauna images (MegaDetector must be loaded)."""
        self.megadetector.load()
        self.crop_backend.load()
        results = []
        try:
            for item in fauna_images:
                path = item.get("image_path", "")
                try:
                    pred = self.crop_backend.predict(path)
                    if pred["status"] == "DETECTED":
                        img_name = Path(path).stem
                        crop_path = crops_dir / f"{img_name}_{pred['flank_side']}.jpg"
                        self.crop_backend.crop_and_save(path, pred["bbox"], crop_path)
                        pred["crop_path"] = str(crop_path)
                    pred["image_path"] = path
                    results.append(pred)
                except Exception as e:
                    logger.warning("Crop failed for %s: %s", path, e)
        finally:
            self.crop_backend.unload()
            self.megadetector.unload()
        return results

    def run_embeddings(self, crop_paths: list[str]) -> list[dict]:
        """Run DINOv2 embeddings on crop images."""
        self.dinov2.load()
        results = []
        try:
            for path in crop_paths:
                try:
                    result = self.dinov2.predict(path)
                    result["crop_path"] = path
                    results.append(result)
                except Exception as e:
                    logger.warning("Embedding failed for %s: %s", path, e)
        finally:
            self.dinov2.unload()
        return results
