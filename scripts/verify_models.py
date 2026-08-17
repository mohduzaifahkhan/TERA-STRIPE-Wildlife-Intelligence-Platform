import torch, os, cv2
print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0))
from ultralytics import YOLO
print("Ultralytics: OK")
from transformers import AutoImageProcessor, AutoModel
print("Transformers: OK")
print("OpenCV:", cv2.__version__)
md_path = "models/md_v5a.0.0.pt"
md_size = os.path.getsize(md_path) / 1e6 if os.path.exists(md_path) else 0
print(f"MegaDetector weights: {os.path.exists(md_path)} ({md_size:.0f} MB)")
print()
print("ALL READY!")
