# AGENTS.md — Antigravity Assistant Instructions & Context Loading

Whenever you start a session in this repository:
1. **Read [`PROJECT_MASTER_CONTEXT.md`](file:///c:/all%20projects/nagpurhack/PROJECT_MASTER_CONTEXT.md) immediately.** It contains the single source of truth for the entire TERA-STRIPE Wildlife Intelligence Platform (all 10 modules, GPU models, tests, database models, and cloud storage).
2. **Current System State:**
   - 260/260 unit & integration tests passing.
   - RTX 4050 Laptop GPU (6GB VRAM) active with PyTorch 2.6.0+cu124.
   - Real AI Pipeline: MegaDetector v5a (`models/md_v5a.0.0.pt`) + DINOv2 ViT-B/14 (`facebook/dinov2-base`).
   - Production Runner: `scripts/run_real_pipeline.py`.
   - Cloud Storage: Google Drive mounted as virtual `G:\` via Rclone & WinFsp.
   - 1-Click Launchers: `mount_gdrive.bat` and `start_dashboard.bat`.
