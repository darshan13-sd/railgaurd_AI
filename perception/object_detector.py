"""
=============================================================================
perception/object_detector.py — Dual YOLO Object Detection
=============================================================================

Wraps ultralytics YOLO for:
  - Model A: Custom Fallen Tree detector
  - Model B: COCO pretrained detector (filtered to classes of interest)

Also performs cross-model NMS merging (previously detection_merger.py).
"""

from dataclasses import dataclass
from typing import List

from ultralytics import YOLO

from config import config as cfg


@dataclass
class Detection:
    """Single detection result."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str
    source: str   # "fallen_tree" or "coco"

    @property
    def center(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def compute_iou(det_a: "Detection", det_b: "Detection") -> float:
    """Intersection-over-Union between two detections."""
    ix1 = max(det_a.x1, det_b.x1)
    iy1 = max(det_a.y1, det_b.y1)
    ix2 = min(det_a.x2, det_b.x2)
    iy2 = min(det_a.y2, det_b.y2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    union = det_a.area + det_b.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def merge_detections(fallen_tree_dets: List[Detection], coco_dets: List[Detection]) -> List[Detection]:
    """Merge detections from both models and apply cross-model greedy NMS."""
    all_dets = list(fallen_tree_dets) + list(coco_dets)
    if not all_dets:
        return []

    all_dets.sort(key=lambda d: d.confidence, reverse=True)

    keep = []
    suppressed = set()
    for i, det_i in enumerate(all_dets):
        if i in suppressed:
            continue
        keep.append(det_i)
        for j in range(i + 1, len(all_dets)):
            if j in suppressed:
                continue
            if compute_iou(det_i, all_dets[j]) > cfg.CROSS_MODEL_NMS_IOU_THRESHOLD:
                suppressed.add(j)

    return keep


class ObjectDetector:
    """Manages both YOLO models and provides a unified detection interface."""

    def __init__(self):
        print(f"[ObjectDetector] Loading fallen-tree model: {cfg.FALLEN_TREE_MODEL_PATH}")
        self.fallen_tree_model = YOLO(cfg.FALLEN_TREE_MODEL_PATH)

        print(f"[ObjectDetector] Loading COCO model: {cfg.COCO_MODEL_PATH}")
        self.coco_model = YOLO(cfg.COCO_MODEL_PATH)
        print("[ObjectDetector] Both models loaded.")

    def _run(self, model, frame, classes=None, name_map=None, source_tag="") -> List[Detection]:
        results = model.predict(
            source=frame,
            conf=cfg.YOLO_CONFIDENCE_THRESHOLD,
            iou=cfg.YOLO_NMS_IOU_THRESHOLD,
            device=cfg.YOLO_DEVICE,
            verbose=False,
            max_det=cfg.YOLO_MAX_DETECTIONS,
            imgsz=cfg.YOLO_IMG_SIZE,
            classes=classes,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = name_map.get(cls_id, f"class_{cls_id}") if name_map else result.names.get(cls_id, f"class_{cls_id}")

                detections.append(Detection(
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                    confidence=conf, class_name=cls_name, source=source_tag
                ))
        return detections

    def run_fallen_tree_model(self, frame) -> List[Detection]:
        return self._run(self.fallen_tree_model, frame, source_tag="fallen_tree")

    def run_coco_model(self, frame) -> List[Detection]:
        return self._run(
            self.coco_model, frame,
            classes=list(cfg.COCO_CLASSES_OF_INTEREST.keys()),
            name_map=cfg.COCO_CLASSES_OF_INTEREST,
            source_tag="coco",
        )

    def detect_all(self, frame) -> List[Detection]:
        """Run both models and return merged, deduplicated detections."""
        fallen_tree_dets = self.run_fallen_tree_model(frame)
        coco_dets = self.run_coco_model(frame)
        return merge_detections(fallen_tree_dets, coco_dets)
