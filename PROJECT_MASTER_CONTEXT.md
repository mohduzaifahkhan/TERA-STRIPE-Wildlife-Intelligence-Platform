# TERA-STRIPE Wildlife Intelligence Platform — Master Context & Project State

> **SYSTEM INSTRUCTION FOR AGENTS & RESTARTS:**  
> This document is the **Single Source of Truth** for the entire TERA-STRIPE codebase at `c:\all projects\nagpurhack`. If Antigravity restarts or starts a new session, read this document to immediately restore complete context about the architecture, models, database schemas, tests, and cloud integrations.

---

## 1. Project Overview & Mission
* **Platform Name:** TERA-STRIPE (Tactical Ecological Reconnaissance & Analysis)
* **Target Reserve:** Pench Tiger Reserve (PTR), Central India
* **Core Function:** End-to-end edge-to-cloud wildlife intelligence pipeline that processes camera trap photos/videos, filters blank images, crops tiger flank biometrics, runs DINOv2 re-identification, logs spatial telemetry in PostGIS/SQLite, triggers security/proximity alerts, and serves an interactive tactical linear dashboard.
* **Current Status:** **All 10 Milestones (M1–M10) Implemented & Verified (265/265 Tests Passing)**. Real 3-model AI pipeline (MegaDetector v5a + YOLO11-Pose Tiger + DINOv2) and Google Drive cloud streaming (`G:\`) are fully functional.
* **Dashboard Status:** **Fully wired to live data** — no mock/demo data. Dashboard pulls from SQLite database (`tera_stripe.db`) and pipeline result files. Auto-refreshes every 30 seconds.
* **Real AI Pipeline Runs:** Successfully processed photos + videos from `G:\My Drive\pench_tiger_system` → 42 tigers detected, 42 anatomical flank crops extracted via YOLO11-Pose (30 Left, 11 Right, 1 Ambiguous, Mean Quality: 0.925), 42 DINOv2 embeddings computed, 7 Pench stations registered.

---

## 2. Hardware & Runtime Environment
* **OS:** Windows 11 (AMD64)
* **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6,140 MiB / 6 GB VRAM)
* **Python Version:** Python 3.13.7
* **Deep Learning Stack:**
  * `torch==2.6.0+cu124` & `torchvision==0.21.0+cu124` (CUDA acceleration active)
  * `ultralytics` (YOLOv5 & YOLO11 engines)
  * `transformers` (Hugging Face DINOv2 ViT-B/14)
  * `opencv-python` (video key-frame extraction)
  * `rclone` + `WinFsp` (virtual cloud drive mounting)
  * `pydantic` v2, `sqlalchemy` 2.0, `shapely`, `h3`, `piexif`
* **VRAM Discipline:** Strict $\le 6\text{ GB}$ VRAM constraint. Models are loaded sequentially and explicitly unloaded with `torch.cuda.empty_cache()` and garbage collection (Peak VRAM $\le 1.4\text{ GB}$).

---

## 3. Full Architecture & Modules Summary (M1 – M10)

```
[Raw Photos / Videos] 
       ↓
[M1: Ingestion Engine] (EXIF GPS, Timestamps, H3 Hex Indexing)
       ↓
[M2: MegaDetector Triage] (FAUNA vs. BLANK vs. HUMAN vs. VEHICLE)
    ├── [BLANK] → [M3: Quarantine Manager] (Storage & Labor ROI Tracking)
    └── [FAUNA] → [M4: YOLO11-Pose Flank Cropper] (6 Anatomical Keypoints, Affine Warp, Quality Score)
                         ↓
                  [M5: Re-ID Engine] (DINOv2 768-dim Embeddings & Vector Gallery)
                         ↓
                  [M6: HITL Queue] (Human Review for Similarity 0.60 ≤ S < 0.85)
                         ↓
                  [M7: Database Manager] (SQLAlchemy 2.0 / PostGIS & SQLite)
                         ↓
                  [M8: Spatial Engine] (95% MCP, KDE 50%, H3 Density, Village Proximity)
                         ↓
                  [M9: Alert Engine] (Critical Village Proximity, Core Shift, Absence)
                         ↓
                  [M10: Reporting & Dashboard] (NTCA Census CSV, Tactical GIS Console)
```

### Module Breakdown:
| Module | File | Responsibility | Tests |
| :--- | :--- | :--- | :---: |
| **M1: Ingestion** | [`src/m1_ingestion.py`](file:///c:/all%20projects/nagpurhack/src/m1_ingestion.py) | Camera trap directory crawler, GPS/EXIF parser, H3 spatial cell assignment (v3/v4 resilient), standard JSON manifest output. | 31 |
| **M2: Triage** | [`src/m2_triage.py`](file:///c:/all%20projects/nagpurhack/src/m2_triage.py) | MegaDetector v5a animal classifier. Labels `FAUNA`, `BLANK`, `HUMAN`, `VEHICLE`. | 26 |
| **M3: Quarantine** | [`src/m3_quarantine.py`](file:///c:/all%20projects/nagpurhack/src/m3_quarantine.py) | Moves blanks to staging, calculates storage space (GB) and manual labor review hours saved. | 18 |
| **M4: Flank Pose** | [`src/m4_flank_pose.py`](file:///c:/all%20projects/nagpurhack/src/m4_flank_pose.py) | YOLO11-Pose 6-keypoint anatomical landmark extraction, geometric flank laterality (`LEFT_FLANK` / `RIGHT_FLANK`), affine warp normalization, quality score calculation. | 32 |
| **M5: Re-ID Engine** | [`src/m5_reid_engine.py`](file:///c:/all%20projects/nagpurhack/src/m5_reid_engine.py) | DINOv2 vision transformer embeddings (768-dim), in-memory/persisted `VectorGallery`, cosine similarity tiering (`AUTO_MATCH` $\ge 0.85$, `REVIEW` $0.60-0.85$, `NEW_INDIVIDUAL` $< 0.60$). | 32 |
| **M6: HITL Queue** | [`src/m6_hitl_queue.py`](file:///c:/all%20projects/nagpurhack/src/m6_hitl_queue.py) | Human-in-the-loop review queue for ambiguous sightings. Supports interactive CLI and auto-resolution. | 27 |
| **M7: DB Manager** | [`src/m7_db_manager.py`](file:///c:/all%20projects/nagpurhack/src/m7_db_manager.py)<br>[`src/m7_database.py`](file:///c:/all%20projects/nagpurhack/src/m7_database.py) | SQLAlchemy 2.0 ORM (`TigerProfile`, `Sighting`, `CameraStation`, `SecurityAlert`, `QuarantineBatch`). Ingests pipeline runs, normalizes UTC datetimes. | 20 |
| **M8: Spatial** | [`src/m8_spatial.py`](file:///c:/all%20projects/nagpurhack/src/m8_spatial.py) | Minimum Convex Polygon (MCP-95 / MCP-100 via Graham scan + Shoelace formula), grid Kernel Density Estimation (KDE-50 / KDE-95), H3 hex density, village proximity calculations, GeoJSON export. | 33 |
| **M9: Alerts** | [`src/m9_alerts.py`](file:///c:/all%20projects/nagpurhack/src/m9_alerts.py) | 4 anomaly detectors: `VILLAGE_PROXIMITY` ($\le 1$km CRITICAL, $\le 3$km WARNING, $\le 5$km INFO), `CORE_RANGE_SHIFT` ($\ge 10$km), `NOVEL_STATION`, `PROLONGED_ABSENCE` ($\ge 30-90$ days). | 27 |
| **M10: Reporting** | [`src/m10_reporting.py`](file:///c:/all%20projects/nagpurhack/src/m10_reporting.py) | Statutory NTCA census CSV generator, Storage ROI report, tiger dossier compiler, unified dashboard feed. | 19 |
| **Frontend** | [`dashboard/app.py`](file:///c:/all%20projects/nagpurhack/dashboard/app.py)<br>[`dashboard/index.html`](file:///c:/all%20projects/nagpurhack/dashboard/index.html) | Tactical Wildlife Intelligence Console. **Live data-driven** — REST API backed by SQLite DB + pipeline JSON files. CartoDB Dark Matter Leaflet GIS map with 5 layers + 4-tab right drawer (Live Alerts, HITL Queue, Tiger Dossier, Operations). Auto-refreshes every 30s. | — |
| **Total Tests** | `pytest tests/` | **265 passed** in ~9.7s. | **265** |

---

## 4. Real AI Pipeline & Model Weights Status

### Active Models on Machine:
1. **MegaDetector v5a (YOLOv5):**
   * Path: `models/md_v5a.0.0.pt` (280.8 MB)
   * Role: Animal detection & bounding-box localization.
   * Loader: PyTorch Hub YOLOv5 (`torch.hub.load('ultralytics/yolov5', 'custom', path=...)`).
2. **DINOv2 ViT-B/14 (Meta):**
   * Cache: `~/.cache/huggingface/hub/models--facebook--dinov2-base` (86.6M parameters)
   * Role: Extracts 768-dimensional invariant visual stripe embeddings.
3. **YOLO11-Pose Tiger (Custom Trained):**
   * Path: `weights/yolo11_pose_tiger.pt` (~5.6 MB)
   * Role: Detects 6 anatomical tiger landmarks (shoulder_scapula, hip_pelvis_root, spine_midpoint, ventral_belly_contour, foreleg_root, hindleg_root) for geometric flank laterality, affine warp normalization, and quality scoring.
   * Architecture: YOLO11-Pose, single-class `tiger`, keypoint shape `[6, 3]`.
   * Training: Fine-tuned on ATRW (Amur Tiger Re-identification in the Wild) dataset, 25 epochs. Box mAP50: 99.5%, Pose mAP50: 97.8%.
   * VRAM: ~0.9 GB (FP16 inference).
4. **Real Production Runner:**
   * Script: [`scripts/run_real_pipeline.py`](file:///c:/all%20projects/nagpurhack/scripts/run_real_pipeline.py)
   * Capability: Ingests real photos/videos → extracts video frames (OpenCV) → MegaDetector GPU inference → **YOLO11-Pose 6-keypoint flank extraction with affine warp** → DINOv2 embeddings → prints Cosine Similarity Matrix → **auto-ingests results into SQLite database** (`tera_stripe.db`) → registers stations → generates alerts → auto-syncs to Google Drive.
   * **Pipeline Step 5 (NEW):** Auto-ingest creates tiger profiles, logs sightings, registers Pench stations, and generates M9 alerts into the database so the dashboard displays live data immediately after a pipeline run.

---

## 5. Storage Architecture & Cloud Sync (Google Drive / Rclone)

* **Local Output Directory:** `data/real_results/` (`crops/`, `video_frames/`, `triage_results.json`, `embeddings.npy`).
* **SQLite Database:** `tera_stripe.db` (project root) — stores tigers, sightings, stations, alerts.
* **Cloud Remote:** `gdrive` (Google Drive authenticated via Rclone OAuth).
* **Virtual Drive Mount (`G:\`):**
  * Mounted via WinFsp FUSE proxy: `rclone mount gdrive: G: --vfs-cache-mode full`.
  * Allows direct streaming and saving to Google Drive with **0% permanent SSD space used**.
* **1-Click Automation Batch Files:**
  * [`mount_gdrive.bat`](file:///c:/all%20projects/nagpurhack/mount_gdrive.bat) → Double-click to mount Google Drive as `G:\`.
  * [`start_dashboard.bat`](file:///c:/all%20projects/nagpurhack/start_dashboard.bat) → Double-click to start dashboard on `http://localhost:8501` and launch browser.

---

## 5b. Live Dashboard Architecture (Data-Driven, No Mock Data)

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  run_real_pipeline.py│────▶│  SQLite Database      │────▶│  dashboard/app.py    │
│  (GPU Inference)     │     │  (tera_stripe.db)     │     │  (REST API Server)   │
│                      │     │  Tigers, Sightings,   │     │                      │
│  Saves:              │     │  Stations, Alerts     │     │  GET /api/feed       │
│  - triage_results    │     │  Crop paths           │     │  GET /api/tigers     │
│  - crop_results      │     │                       │     │  GET /api/stations   │
│  - embeddings        │     │  + Pipeline JSON      │     │  GET /api/alerts     │
│  - crops/*.jpg       │     │  in data/real_results  │     │  GET /api/hitl       │
│                      │     │                       │     │  GET /api/crops/<f>  │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
                                                                    │
                                                                    ▼
                                                          ┌──────────────────────┐
                                                          │  index.html          │
                                                          │  fetch('/api/feed')  │
                                                          │  Auto-refresh 30s    │
                                                          │  No hardcoded data   │
                                                          └──────────────────────┘
```

### Dashboard API Endpoints:
| Endpoint | Returns |
|---|---|
| `GET /api/feed` | Full dashboard payload (KPIs, tigers, stations, alerts, HITL, ROI, pipeline stats) |
| `GET /api/tigers` | All tiger profiles with sighting counts and flank crop paths |
| `GET /api/stations` | All camera stations from DB + Pench station registry |
| `GET /api/alerts` | Security alerts from DB, or pipeline-derived alerts as fallback |
| `GET /api/hitl` | Pending HITL review items with query+gallery image pairs |
| `GET /api/crops/<file>` | Serves crop images from `data/real_results/crops/` |

### Data Sources (priority order):
1. **SQLite Database** (via M7 DatabaseManager) — tigers, sightings, stations, alerts
2. **Pipeline JSON files** (`data/real_results/*.json`) — latest pipeline run stats
3. **Pench Station Registry** (`src/m1_ingestion.py PENCH_STATION_REGISTRY`) — 7 known station GPS coordinates

### Real Pipeline Execution Metrics:
| Metric | Value |
|---|---|
| Source Tested | `G:\My Drive\pench_tiger_system` (photos + videos) |
| Images Triaged | 46 (42 fauna, 3 blanks, 1 human) |
| Tiger Crops (M4 YOLO11-Pose) | **42 crops** (30 LEFT, 11 RIGHT, 1 AMBIGUOUS) |
| Crop Quality Score | **0.9251 Mean Quality** (38/42 High Quality $\ge 0.70$) |
| Affine Warp Normalization | Applied via 6 anatomical landmarks |
| DINOv2 Embeddings | 42 × 768-dim invariant vectors |
| Tigers in DB | 42 profiles created |
| Sightings Logged | 42 |
| Stations Registered | 7 (Pench Tiger Reserve) |
| Video Frames Extracted | 30 (from 3 videos, 10 frames each) |

---

## 6. CLI Command Cheatsheet

### Run Automated AI Pipeline on Real Photos/Videos:
```powershell
cd "c:\all projects\nagpurhack"

# Process photos/videos and save locally (uses all 3 models):
python scripts/run_real_pipeline.py --source "G:\My Drive\pench_tiger_system"

# Process and auto-upload to Google Drive in 1 command:
python scripts/run_real_pipeline.py --source "G:\My Drive\pench_tiger_system" --rclone-remote gdrive:pench_wildlife_data/

# Process directly into mounted G: virtual drive:
python scripts/run_real_pipeline.py --source "G:\My Drive\pench_tiger_system" --output-dir "G:\pench_wildlife_data\results"
```

### Run Full Test Suite:
```powershell
cd "c:\all projects\nagpurhack"
python -m pytest tests/ -v
```

### Launch Tactical Dashboard:
```powershell
python dashboard/app.py
# Open: http://localhost:8501
```

### Generate Statutory NTCA Census & ROI Reports:
```powershell
python -m src.m10_reporting --census --output-dir data/exports
python -m src.m10_reporting --kpi
python -m src.m10_reporting --roi
```

---

## 7. Directory Structure Reference
```
c:\all projects\nagpurhack\
├── dashboard\
│   ├── app.py                      # REST API server (live data from DB + pipeline files)
│   ├── index.html                  # Single-Page Tactical GIS Console (fetches /api/feed)
│   └── static/                     # Leaflet JS/CSS (bundled locally, no CDN)
├── data\
│   ├── exports\                    # NTCA census CSVs, reports
│   ├── manifests\                  # Pipeline JSON manifests
│   ├── real_results\               # Real AI outputs (crops, frames, vectors)
│   ├── spatial\                    # GeoJSON polygons & home ranges
│   └── test_fixtures\              # Synthetic camera trap images
├── models\
│   └── md_v5a.0.0.pt               # MegaDetector v5a weights (280.8 MB)
├── weights\
│   └── yolo11_pose_tiger.pt        # Custom YOLO11-Pose Tiger weights (~5.6 MB)
├── scripts\
│   ├── create_test_fixtures.py     # Generates synthetic test fixtures
│   ├── run_real_pipeline.py        # Real AI GPU pipeline + Cloud Sync
│   └── verify_models.py            # Hardware & model sanity checker
├── src\
│   ├── __init__.py
│   ├── config.py                   # Pydantic Settings
│   ├── m1_ingestion.py             # Ingestion & EXIF
│   ├── m2_triage.py                # Triage engine
│   ├── m3_quarantine.py            # Quarantine manager
│   ├── m4_flank_pose.py            # YOLO11-Pose flank cropper & affine warp
│   ├── m5_reid_engine.py           # DINOv2 Re-ID engine
│   ├── m6_hitl_queue.py            # Human-in-the-loop review
│   ├── m7_database.py              # SQLAlchemy 2.0 ORM schemas
│   ├── m7_db_manager.py            # Database operations
│   ├── m8_spatial.py               # MCP / KDE / H3 spatial engine
│   ├── m9_alerts.py                # Anomaly & proximity alert engine
│   ├── m10_reporting.py            # NTCA census export engine
│   └── real_backends.py            # Production GPU backends
├── tests\                          # 265 unit & integration tests
├── tera_stripe.db                  # SQLite database (tigers, sightings, stations, alerts)
├── mount_gdrive.bat                # 1-click Google Drive mounter (G:)
├── start_dashboard.bat             # 1-click Dashboard starter
├── requirements.txt
└── PROJECT_MASTER_CONTEXT.md       # THIS MASTER CONTEXT FILE
```
