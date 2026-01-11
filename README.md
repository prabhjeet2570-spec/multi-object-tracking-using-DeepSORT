# Multi-Object Tracking with YOLOv11 and DeepSORT

Real-time object detection and tracking using YOLOv11 + DeepSORT.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Download the feature model:
```bash
wget https://github.com/anushkadhiman/ObjectTracking-DeepSORT-YOLOv3-TF2/raw/master/model_data/coco/mars-small128.pb
```

YOLOv11 will download automatically on first run.

## Run

```bash
python tracker.py
```

Change the input/output video paths in `tracker.py` before running.

## Files

- `settings.py` - Configuration
- `detection.py` - Object detection and features
- `kalman.py` - Motion prediction
- `matching.py` - Track-detection matching
- `distance.py` - Appearance similarity
- `track.py` - Track lifecycle
- `visualization.py` - Draw boxes
- `tracker.py` - Main script

## How it works

1. YOLOv11 detects objects
2. Extract appearance features
3. Kalman filter predicts motion
4. Match detections to existing tracks
5. Update or create tracks

## Adjust settings

Edit `settings.py` to change detection threshold, tracking parameters, etc.
