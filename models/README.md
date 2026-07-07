# models/

Place your model weight files here:

| File | Description | Source |
|---|---|---|
| `best.pt` | Custom YOLOv11s fallen-tree detector | Your trained weights |
| `yolo11s.pt` | Official YOLOv11s COCO model | Auto-downloaded by `ultralytics` on first run |
| `depth_anything.onnx` | Depth Anything (ONNX export) for monocular depth estimation | Export from https://github.com/DepthAnything/Depth-Anything-V2 |
| `track_seg.onnx` | Rail/track segmentation model (ONNX export) | Your trained segmentation model, or Roboflow-exported ONNX |

None of these binary weight files are included in this package — only code.
Update `config/settings.yaml` → `paths:` if you store them elsewhere.
