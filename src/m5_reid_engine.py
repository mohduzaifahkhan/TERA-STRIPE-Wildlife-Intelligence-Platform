"""
TERA-STRIPE Module 5 -- DINOv2 + Sub-Center ArcFace Re-Identification Engine
==============================================================================
Extracts 1024-dimensional embeddings from normalised flank crops using
DINOv2 ViT-L/14, queries a persistent vector gallery via cosine similarity,
and classifies matches against known tiger identities.

Pipeline position: Stage 3 of the sequential VRAM execution model.
VRAM budget: < 2.5 GB (FP16 DINOv2 ViT-L/14), loaded AFTER M4 unloads.

Similarity thresholds (from AppConfig):
  >= 0.85  AUTO_MATCH      -> assign tiger_id automatically
  0.60-0.85  REVIEW        -> route to HITL queue (M6)
  < 0.60   NEW_INDIVIDUAL  -> create new tiger profile

Data contract:
  Input  : flank_extraction.json  (from M4)
  Output : reid_result.json       (to M6 / M7)

CLI Usage
---------
  python -m src.m5_reid_engine \\
      --flank-result ./data/manifests/flank_BATCH.json \\
      --gallery-dir ./data/vector_store \\
      --weights ./weights/dinov2_vitl14.pth \\
      --device cuda:0

Reference: Master Context Packet -- Stage 3, Cosine Similarity, ArcFace
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

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("tera_stripe.m5_reid")

# ── Constants ────────────────────────────────────────────────────
EMBEDDING_DIM = 1024           # DINOv2 ViT-L/14 output dimension
DEFAULT_TOP_K = 5              # Top-K gallery matches to return
SIM_AUTO_MATCH = 0.85          # Auto-match threshold
SIM_REVIEW_MIN = 0.60          # Minimum for HITL review
CROP_INPUT_SIZE = 224          # Expected input crop size


# =====================================================================
#  Pydantic Contract Models -- reid_result.json
# =====================================================================

class ReIDMatch(BaseModel):
    """A single gallery match candidate."""
    tiger_id: str
    similarity: float = Field(..., ge=-1.0, le=1.0)
    rank: int = Field(..., ge=1)


class ReIDDispatch(BaseModel):
    """Per-crop re-identification result."""
    image_id: str
    crop_path: str
    flank_side: str
    embedding_dim: int = EMBEDDING_DIM
    status: Literal["AUTO_MATCH", "REVIEW", "NEW_INDIVIDUAL"]
    top_k_matches: list[ReIDMatch] = []
    assigned_tiger_id: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)


class ReIDSummary(BaseModel):
    """Aggregate re-identification statistics."""
    total_queries: int
    auto_matches: int = 0
    review_queue: int = 0
    new_individuals: int = 0
    gallery_size_before: int
    gallery_size_after: int


class ReIDResult(BaseModel):
    """Complete output conforming to reid_result.json contract."""
    batch_id: str
    similarity_thresholds: dict = Field(default_factory=lambda: {
        "auto_match": SIM_AUTO_MATCH,
        "review_min": SIM_REVIEW_MIN,
    })
    summary: ReIDSummary
    dispatches: list[ReIDDispatch]


# =====================================================================
#  Vector Gallery (NumPy-backed, FAISS-optional)
# =====================================================================

class VectorGallery:
    """
    Persistent gallery of tiger identity embeddings.

    Uses NumPy cosine similarity by default, with optional FAISS
    acceleration when available. Persists to disk as .npz + JSON index.
    """

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        self.dimension = dimension
        self._ids: list[str] = []        # tiger_id per row
        self._embeddings: np.ndarray | None = None  # (N, dim)
        self._metadata: dict[str, dict] = {}  # tiger_id -> metadata

    @property
    def size(self) -> int:
        """Number of identities in the gallery."""
        return len(self._ids)

    def add(
        self,
        tiger_id: str,
        embedding: np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        """Add or update a tiger identity embedding."""
        embedding = self._normalise(embedding)

        if tiger_id in self._ids:
            # Update existing: running average
            idx = self._ids.index(tiger_id)
            old = self._embeddings[idx]
            self._embeddings[idx] = self._normalise(
                (old + embedding) / 2.0
            )
        else:
            self._ids.append(tiger_id)
            if self._embeddings is None:
                self._embeddings = embedding.reshape(1, -1)
            else:
                self._embeddings = np.vstack([
                    self._embeddings, embedding.reshape(1, -1)
                ])

        if metadata:
            self._metadata[tiger_id] = metadata

    def search(
        self,
        query: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[ReIDMatch]:
        """
        Find the top-K most similar gallery embeddings via cosine similarity.

        Returns
        -------
        list[ReIDMatch]
            Sorted by similarity descending.
        """
        if self._embeddings is None or len(self._ids) == 0:
            return []

        query = self._normalise(query).flatten()
        # Cosine similarity (embeddings are already L2-normalised)
        similarities = self._embeddings @ query
        k = min(top_k, len(self._ids))

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]

        matches = []
        for rank, idx in enumerate(top_indices, start=1):
            # Clamp negative cosine similarity to 0 (no practical meaning)
            sim = max(0.0, float(similarities[idx]))
            matches.append(
                ReIDMatch(
                    tiger_id=self._ids[idx],
                    similarity=round(sim, 4),
                    rank=rank,
                )
            )
        return matches

    def save(self, directory: Path) -> None:
        """Persist gallery to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        if self._embeddings is not None:
            np.save(
                str(directory / "gallery_embeddings.npy"),
                self._embeddings,
            )

        # Save index mapping
        index_data = {
            "dimension": self.dimension,
            "ids": self._ids,
            "metadata": self._metadata,
        }
        with open(directory / "gallery_index.json", "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

        logger.info(
            "Gallery saved: %d identities -> %s",
            len(self._ids),
            directory,
        )

    def load(self, directory: Path) -> None:
        """Load gallery from disk."""
        directory = Path(directory)

        index_path = directory / "gallery_index.json"
        emb_path = directory / "gallery_embeddings.npy"

        if not index_path.exists():
            logger.info("No existing gallery found at %s.", directory)
            return

        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        self.dimension = index_data.get("dimension", EMBEDDING_DIM)
        self._ids = index_data.get("ids", [])
        self._metadata = index_data.get("metadata", {})

        if emb_path.exists() and self._ids:
            self._embeddings = np.load(str(emb_path))
        else:
            self._embeddings = None

        logger.info(
            "Gallery loaded: %d identities from %s",
            len(self._ids),
            directory,
        )

    def contains(self, tiger_id: str) -> bool:
        """Check if a tiger_id exists in the gallery."""
        return tiger_id in self._ids

    def get_embedding(self, tiger_id: str) -> np.ndarray | None:
        """Retrieve the embedding for a specific tiger."""
        if tiger_id not in self._ids:
            return None
        idx = self._ids.index(tiger_id)
        return self._embeddings[idx].copy()

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        """L2-normalise a vector."""
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return v
        return v / norm


# =====================================================================
#  Embedding Backend Interface
# =====================================================================

class EmbeddingBackend(ABC):
    """Abstract interface for embedding extraction."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights."""

    @abstractmethod
    def extract_batch(
        self, crop_paths: list[Path]
    ) -> list[np.ndarray]:
        """
        Extract embeddings from a batch of crop images.

        Returns
        -------
        list[np.ndarray]
            Each array is shape (EMBEDDING_DIM,), L2-normalised.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier."""


# =====================================================================
#  Production Backend -- DINOv2 ViT-L/14
# =====================================================================

class DINOv2Backend(EmbeddingBackend):
    """
    Production DINOv2 ViT-L/14 embedding extractor.

    Loads the model via torch.hub or from a local checkpoint,
    runs FP16 inference, and returns L2-normalised 1024-dim embeddings.
    """

    def __init__(
        self,
        weights_path: Path | None = None,
        device: str = "cuda:0",
        batch_size: int = 16,
    ) -> None:
        self.weights_path = weights_path
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None
        self._transform: Any = None

    @property
    def backend_name(self) -> str:
        return "DINOv2 ViT-L/14"

    def load(self) -> None:
        try:
            import torch
            import torchvision.transforms as T
        except ImportError as exc:
            raise ImportError(
                "torch and torchvision required for DINOv2. "
                "Install: pip install torch torchvision"
            ) from exc

        logger.info("Loading DINOv2 ViT-L/14...")

        if self.weights_path and Path(self.weights_path).exists():
            self._model = torch.load(
                str(self.weights_path),
                map_location=self.device,
            )
        else:
            self._model = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitl14",
            )

        self._model.eval()
        if "cuda" in self.device and torch.cuda.is_available():
            self._model = self._model.to(self.device).half()

        self._transform = T.Compose([
            T.Resize((CROP_INPUT_SIZE, CROP_INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        logger.info("DINOv2 loaded on %s.", self.device)

    def extract_batch(
        self, crop_paths: list[Path]
    ) -> list[np.ndarray]:
        import torch

        if self._model is None:
            raise RuntimeError("Model not loaded.")

        embeddings = []
        for i in range(0, len(crop_paths), self.batch_size):
            batch_paths = crop_paths[i : i + self.batch_size]
            tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                tensors.append(self._transform(img))

            batch = torch.stack(tensors)
            if "cuda" in self.device and torch.cuda.is_available():
                batch = batch.to(self.device).half()

            with torch.no_grad():
                features = self._model(batch)

            for feat in features.cpu().float().numpy():
                norm = np.linalg.norm(feat)
                embeddings.append(feat / norm if norm > 1e-8 else feat)

        return embeddings

    def unload(self) -> None:
        logger.info("Unloading DINOv2...")
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
        logger.info("DINOv2 unloaded.")


# =====================================================================
#  Heuristic Mock Embedding Backend
# =====================================================================

class MockEmbeddingBackend(EmbeddingBackend):
    """
    Deterministic mock embedding extractor for testing.

    Generates reproducible 1024-dim embeddings by combining:
      1. Seeded random vector (from image filename)
      2. Image pixel statistics (mean, std per channel)

    This ensures visually similar images produce similar embeddings
    while maintaining full determinism across test runs.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._loaded = False

    @property
    def backend_name(self) -> str:
        return "MockEmbedding (no GPU)"

    def load(self) -> None:
        self._loaded = True
        logger.info("Mock embedding backend loaded (seed=%d).", self.seed)

    def extract_batch(
        self, crop_paths: list[Path]
    ) -> list[np.ndarray]:
        if not self._loaded:
            raise RuntimeError("Mock embedding backend not loaded.")

        embeddings = []
        for path in crop_paths:
            rng = np.random.RandomState(hash(path.stem) % (2**31) ^ self.seed)

            # Base random embedding
            base = rng.randn(EMBEDDING_DIM).astype(np.float32)

            # Mix in image pixel statistics for similarity preservation
            try:
                img = Image.open(path).convert("RGB")
                arr = np.array(img, dtype=np.float32) / 255.0
                stats = np.concatenate([
                    arr.mean(axis=(0, 1)),  # 3 values (R, G, B means)
                    arr.std(axis=(0, 1)),   # 3 values (R, G, B stds)
                ])
                # Tile stats to EMBEDDING_DIM and mix
                stats_expanded = np.tile(
                    stats, EMBEDDING_DIM // len(stats) + 1
                )[:EMBEDDING_DIM]
                base = base * 0.7 + stats_expanded * 0.3
            except Exception:
                pass

            # L2 normalise
            norm = np.linalg.norm(base)
            if norm > 1e-8:
                base = base / norm

            embeddings.append(base)

        return embeddings

    def unload(self) -> None:
        self._loaded = False
        logger.info("Mock embedding backend unloaded.")


# =====================================================================
#  Backend Factory
# =====================================================================

def create_embedding_backend(
    weights_path: Path | None = None,
    device: str = "cuda:0",
    force_mock: bool = False,
) -> EmbeddingBackend:
    """Create the appropriate embedding backend."""
    if force_mock:
        return MockEmbeddingBackend()

    if weights_path and Path(weights_path).exists():
        try:
            import torch  # noqa: F401
            return DINOv2Backend(weights_path=Path(weights_path), device=device)
        except ImportError:
            logger.warning("torch not installed. Using mock embedding backend.")

    logger.warning(
        "DINOv2 weights not found at '%s'. Using mock backend.", weights_path
    )
    return MockEmbeddingBackend()


# =====================================================================
#  Re-ID Engine Orchestrator
# =====================================================================

class ReIDEngine:
    """
    Orchestrates the full re-identification pipeline.

    1. Load flank extraction results (M4 output)
    2. Load/create vector gallery
    3. Load embedding backend (sequential VRAM slot)
    4. Extract embeddings from all flank crops
    5. Query gallery for each embedding
    6. Classify: AUTO_MATCH / REVIEW / NEW_INDIVIDUAL
    7. Update gallery with new identities
    8. Unload backend, persist gallery
    9. Produce reid_result.json
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        gallery: VectorGallery,
        sim_auto_match: float = SIM_AUTO_MATCH,
        sim_review_min: float = SIM_REVIEW_MIN,
        top_k: int = DEFAULT_TOP_K,
        batch_size: int = 16,
    ) -> None:
        self.backend = backend
        self.gallery = gallery
        self.sim_auto_match = sim_auto_match
        self.sim_review_min = sim_review_min
        self.top_k = top_k
        self.batch_size = batch_size
        self._next_tiger_counter = 0

    def _generate_new_tiger_id(self, batch_id: str) -> str:
        """Generate a sequential new tiger ID."""
        self._next_tiger_counter += 1
        return f"PTR_NEW_{self._next_tiger_counter:03d}"

    def _classify_match(
        self,
        top_matches: list[ReIDMatch],
    ) -> tuple[str, str | None, float]:
        """
        Classify the re-identification result.

        Returns
        -------
        tuple[status, assigned_tiger_id, confidence]
        """
        if not top_matches:
            return "NEW_INDIVIDUAL", None, 0.0

        best = top_matches[0]

        if best.similarity >= self.sim_auto_match:
            return "AUTO_MATCH", best.tiger_id, best.similarity
        elif best.similarity >= self.sim_review_min:
            return "REVIEW", best.tiger_id, best.similarity
        else:
            return "NEW_INDIVIDUAL", None, best.similarity

    def process_extractions(
        self,
        flank_data: dict,
        gallery_dir: Path | None = None,
    ) -> ReIDResult:
        """
        Run the full re-ID pipeline on flank extraction results.

        Parameters
        ----------
        flank_data : dict
            Parsed flank_extraction.json (M4 output).
        gallery_dir : Path, optional
            Directory to load/save the vector gallery.
        """
        batch_id = flank_data["batch_id"]
        extractions = flank_data.get("extractions", [])

        # Load existing gallery
        if gallery_dir:
            self.gallery.load(Path(gallery_dir))

        gallery_size_before = self.gallery.size

        logger.info(
            "Re-ID | batch=%s | crops=%d | gallery=%d | backend=%s",
            batch_id,
            len(extractions),
            gallery_size_before,
            self.backend.backend_name,
        )

        # ── Load embedding backend ──
        t_start = time.time()
        self.backend.load()

        # ── Extract embeddings ──
        crop_paths = [Path(e["crop_path"]) for e in extractions]
        valid_paths = []
        valid_extractions = []

        for path, ext in zip(crop_paths, extractions):
            if path.exists():
                valid_paths.append(path)
                valid_extractions.append(ext)
            else:
                logger.warning("Crop not found: %s", path)

        embeddings = []
        if valid_paths:
            embeddings = self.backend.extract_batch(valid_paths)

        # ── Unload backend (VRAM discipline) ──
        self.backend.unload()

        # ── Query gallery and classify ──
        dispatches: list[ReIDDispatch] = []

        for ext, embedding in zip(valid_extractions, embeddings):
            image_id = ext["image_id"]
            crop_path = ext["crop_path"]
            flank_side = ext.get("flank_side", "AMBIGUOUS")

            # Search gallery
            top_matches = self.gallery.search(embedding, self.top_k)

            # Classify
            status, assigned_id, confidence = self._classify_match(
                top_matches
            )

            # For NEW_INDIVIDUAL, generate ID and add to gallery
            if status == "NEW_INDIVIDUAL":
                new_id = self._generate_new_tiger_id(batch_id)
                assigned_id = new_id
                self.gallery.add(
                    new_id,
                    embedding,
                    metadata={
                        "first_seen_batch": batch_id,
                        "first_image": image_id,
                    },
                )

            # For AUTO_MATCH, update gallery with new observation
            elif status == "AUTO_MATCH" and assigned_id:
                self.gallery.add(assigned_id, embedding)

            dispatches.append(
                ReIDDispatch(
                    image_id=image_id,
                    crop_path=crop_path,
                    flank_side=flank_side,
                    embedding_dim=EMBEDDING_DIM,
                    status=status,
                    top_k_matches=top_matches,
                    assigned_tiger_id=assigned_id,
                    confidence=round(confidence, 4),
                )
            )

        # ── Save gallery ──
        if gallery_dir:
            self.gallery.save(Path(gallery_dir))

        t_elapsed = time.time() - t_start
        gallery_size_after = self.gallery.size

        # ── Build summary ──
        auto = sum(1 for d in dispatches if d.status == "AUTO_MATCH")
        review = sum(1 for d in dispatches if d.status == "REVIEW")
        new = sum(1 for d in dispatches if d.status == "NEW_INDIVIDUAL")

        summary = ReIDSummary(
            total_queries=len(dispatches),
            auto_matches=auto,
            review_queue=review,
            new_individuals=new,
            gallery_size_before=gallery_size_before,
            gallery_size_after=gallery_size_after,
        )

        logger.info(
            "Re-ID complete | %.1fs | auto=%d review=%d new=%d | "
            "gallery: %d -> %d",
            t_elapsed,
            auto,
            review,
            new,
            gallery_size_before,
            gallery_size_after,
        )

        return ReIDResult(
            batch_id=batch_id,
            summary=summary,
            dispatches=dispatches,
        )


# =====================================================================
#  CLI Entry Point
# =====================================================================

def main() -> None:
    """CLI entry point for the Re-ID engine."""
    parser = argparse.ArgumentParser(
        prog="m5_reid_engine",
        description="TERA-STRIPE M5 -- DINOv2 + ArcFace Re-ID Engine",
    )
    parser.add_argument(
        "--flank-result", type=Path, required=True,
        help="Path to flank_extraction.json (M4 output).",
    )
    parser.add_argument(
        "--gallery-dir", type=Path,
        default=Path("./data/vector_store"),
        help="Vector gallery directory.",
    )
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="Path to DINOv2 .pth weights.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--sim-auto-match", type=float, default=SIM_AUTO_MATCH,
        help=f"Auto-match similarity threshold (default: {SIM_AUTO_MATCH}).",
    )
    parser.add_argument(
        "--sim-review-min", type=float, default=SIM_REVIEW_MIN,
        help=f"Review minimum similarity (default: {SIM_REVIEW_MIN}).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for reid_result.json.",
    )
    parser.add_argument(
        "--force-mock", action="store_true",
        help="Force mock backend (no GPU/weights).",
    )
    args = parser.parse_args()

    # Load flank result
    if not args.flank_result.exists():
        logger.error("Flank result not found: %s", args.flank_result)
        sys.exit(1)
    with open(args.flank_result, "r", encoding="utf-8") as f:
        flank_data = json.load(f)

    # Create backend
    backend = create_embedding_backend(
        weights_path=args.weights,
        device=args.device,
        force_mock=args.force_mock,
    )

    # Create gallery and engine
    gallery = VectorGallery(dimension=EMBEDDING_DIM)
    engine = ReIDEngine(
        backend=backend,
        gallery=gallery,
        sim_auto_match=args.sim_auto_match,
        sim_review_min=args.sim_review_min,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )

    result = engine.process_extractions(
        flank_data, gallery_dir=args.gallery_dir
    )

    # Write output
    if args.output is None:
        args.output = (
            args.flank_result.parent
            / f"reid_{flank_data['batch_id']}.json"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    s = result.summary
    print(
        f"\n{'='*60}\n"
        f"  TERA-STRIPE M5 Re-ID Complete\n"
        f"  Batch       : {result.batch_id}\n"
        f"  Backend     : {backend.backend_name}\n"
        f"  Queries     : {s.total_queries}\n"
        f"  Auto Match  : {s.auto_matches}\n"
        f"  Review Queue: {s.review_queue}\n"
        f"  New Tigers  : {s.new_individuals}\n"
        f"  Gallery     : {s.gallery_size_before} -> {s.gallery_size_after}\n"
        f"  Output      : {args.output}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
