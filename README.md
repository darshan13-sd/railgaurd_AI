<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv11-FF6F00?style=for-the-badge&logo=yolo&logoColor=white" />
  <img src="https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</div>

<h1 align="center">🚂 RailGuard-AI</h1>
<p align="center"><b>An end-to-end perception → prediction → alert pipeline for real-time railway track obstacle detection.</b></p>

<p align="center">
  Detects fallen trees, vehicles, animals, and people on railway tracks using a dual-YOLO<br/>
  ensemble, estimates real proximity with monocular depth + object tracking, scores risk,<br/>
  and raises DANGER / WARNING alerts before a human even notices.
</p>

---

## 🚀 The Problem

Trains need enormous stopping distances. If an obstacle — a fallen tree, a stalled vehicle,
wandering livestock, or a person — is on the track, the operator needs to know **immediately**,
not after the train is already too close to react.

Generic object detectors also produce constant false alarms from objects safely *off* the track
(a car on a parallel road, a cow in a nearby field). A usable safety system has to reason about
**where** the object is relative to the rails, **how fast** it's approaching, and **how urgent**
the alert actually is — not just "something was detected somewhere in frame."

## 💡 The Solution

RailGuard-AI is a modular, config-driven CV pipeline split into four clear stages:

**Perception → Prediction → Alert → Visualization**

1. **Perception** — figure out *what* is in the scene and *where* the track is
2. **Prediction** — turn that into a risk score and a time-to-collision estimate
3. **Alert** — fan out DANGER/WARNING events to voice, console, and a persistent log
4. **Visualization** — render a clean HUD so a human operator can verify at a glance

---

## 🏗️ Architecture

```
RailGuard-AI/
├── app.py                        # Entry point — wires all modules together
├── config/
│   ├── config.py                 # Loads settings.yaml into typed constants
│   └── settings.yaml             # ALL tunable parameters live here
├── models/                       # Model weights go here (not included in repo)
├── perception/
│   ├── object_detector.py        # Dual YOLO detection + cross-model NMS merge
│   ├── track_segmenter.py        # ONNX rail/track segmentation → ROI polygon
│   ├── depth_estimator.py        # Monocular depth (Depth-Anything ONNX)
│   ├── object_tracker.py         # Centroid multi-object tracker (persistent IDs)
│   └── scene_understanding.py    # Fuses ROI + tracks + depth into a SceneState
├── prediction/
│   ├── speed_estimator.py        # Screen-space + depth closing-speed estimation
│   ├── collision_predictor.py    # Trajectory extrapolation vs. the track ROI
│   ├── risk_engine.py            # Weighted 0–1 risk scoring + Time-To-Collision
│   └── decision_engine.py        # DANGER / WARNING / SAFE classification
├── alert/
│   ├── voice_alert.py            # Offline TTS spoken alerts (rate-limited)
│   ├── notifier.py               # Fan-out: console + voice + database
│   └── event_logger.py           # Persists alert events to SQLite
├── dashboard/
│   ├── dashboard.py               # Streamlit live viewer + rolling event log
│   └── visualization.py           # HUD drawing (ROI, boxes, status bar, alerts)
├── database/
│   └── sqlite.py                  # SQLite wrapper for event history
├── utils/
│   ├── geometry.py                 # IoU, distance, point-in-polygon helpers
│   └── video_io.py                 # VideoCapture / VideoWriter helpers
└── requirements.txt
```

## 🧠 How It Works

### 1. Dual YOLO Detection
Runs two YOLOv11 models concurrently:
- **Model A** — a custom-trained detector specialized for **fallen trees**, a class no
  public dataset labels
- **Model B** — the official COCO-pretrained model, filtered down to obstacle-relevant
  classes (people, vehicles, animals)

Results are merged with cross-model greedy NMS to remove duplicate/overlapping boxes.

### 2. Track Segmentation
An ONNX segmentation model locates the rail region and outputs a smoothed polygon ROI,
so detections can be checked against the *actual* track geometry instead of a fixed zone —
with a static-trapezoid fallback if no segmentation model is available.

### 3. Depth + Tracking
A monocular depth model estimates relative proximity per pixel, and a lightweight centroid
tracker assigns persistent IDs across frames — giving every object a position *and* velocity
history instead of a single disconnected frame.

### 4. Risk Scoring & Decision
Each tracked object gets a weighted 0–1 risk score from three signals — ROI overlap, proximity,
and closing speed — plus a computed Time-To-Collision. These combine into a final
**DANGER / WARNING / SAFE** classification.

### 5. Alerting
DANGER/WARNING events trigger a spoken voice alert (rate-limited to avoid spamming), a console
log line, and a persistent SQLite record for later review.

### 6. Visualization
A HUD overlay draws the ROI, color-coded bounding boxes, a live status bar, FPS/frame counters,
and a flashing alert banner — viewable either in the rendered output video or live via the
Streamlit dashboard.

---

## ⚙️ Installation

```bash
git clone <your-repo-url> RailGuard-AI
cd RailGuard-AI
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

> If you don't have a CUDA GPU, swap `onnxruntime-gpu` → `onnxruntime` in `requirements.txt`,
> and set `device: cpu` under `yolo:` in `config/settings.yaml`.

### Add your model weights

Drop these into `models/` (see `models/README.md` for details):

| File | Required? | Notes |
|---|---|---|
| `best.pt` | Yes | Custom fallen-tree YOLO model |
| `yolo11s.pt` | No | Auto-downloaded by `ultralytics` on first run |
| `depth_anything.onnx` | Optional | Falls back to a position-based heuristic if missing |
| `track_seg.onnx` | Optional | Falls back to a static trapezoid ROI if missing |

### Configure

Edit `config/settings.yaml` — video paths, detection thresholds, and every other tunable
parameter live here. No Python edits required.

---

## ▶️ Usage

**Run the full pipeline** (writes an annotated output video + logs events to SQLite):
```bash
python app.py
```

**Launch the live dashboard** (real-time preview + rolling alert table):
```bash
streamlit run dashboard/dashboard.py
```

---

## 🛣️ Roadmap

- [ ] Validate against real-world railway footage (beyond synthetic/stock test clips)
- [ ] Tune risk-engine weights per obstacle class (e.g. lower TTC threshold for people)
- [ ] Swap the centroid tracker for ByteTrack/DeepSORT for crowded-scene robustness
- [ ] Explore parallel (rather than sequential) dual-model inference for higher FPS
- [ ] Package as a Docker image for easier deployment

## 📄 License

MIT — see [LICENSE](LICENSE).
