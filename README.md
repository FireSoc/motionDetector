# Motion Detector

A webcam-based motion detection app that records video clips whenever motion is
detected and a recognizable object is present. It combines classic computer
vision (background subtraction + contour detection) with a YOLOv8 object
detector to reduce false positives and to draw labeled bounding boxes on the
recorded footage.

## How it works

1. Each frame from the camera is converted to grayscale and blurred.
2. A MOG2 background subtractor compares the frame against a learned background
   to produce a foreground mask.
3. The mask is thresholded, dilated, and scanned for contours. Contours larger
   than `min_contour_area` count as motion.
4. When motion is found, YOLOv8 runs on the original frame. If it detects any
   object, the app starts recording and draws labeled bounding boxes.
5. Recording continues for `post_motion_frames` frames after the last detection,
   then stops. Each clip is saved to `motion_detection_clips/` with a timestamped
   filename (e.g. `2026-06-13_13-30-00.mp4`).

Press `q` in the preview window to quit.

## Project structure

- `main.py` — core motion detection and recording loop (`run_motion_detection`).
- `config.json` — all tunable parameters (thresholds, colors, FPS, model path).
- `test.py` — convenience script to run detection on a specific camera index.
- `requirements.txt` — Python dependencies.
- `yolov8n.pt` — YOLOv8 nano model weights.
- `motion_detection_clips/` — output folder for recorded clips (auto-created,
  gitignored).

## Setup

Requires Python 3.10+.

```bash
# Clone and enter the project
cd motionDetector

# (Recommended) create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The `yolov8n.pt` weights are included in the repo. If missing, `ultralytics`
will download them automatically on first run.

## Usage

Run with the default camera (index `0`):

```bash
python main.py
```

Run against a different camera (e.g. an external webcam at index `1`):

```bash
python test.py
```

Or call the function directly from your own script:

```python
from main import run_motion_detection

run_motion_detection(camera_index=0)
```

The `motion_detection_clips/` folder is created automatically if it does not
already exist, so no manual setup is needed before the first run.

## Configuration

Tune behavior by editing `config.json`. Key settings:

| Key | Description |
| --- | --- |
| `model_path` | Path to the YOLO weights file. |
| `min_contour_area` | Minimum contour size (pixels) to count as motion. |
| `bg_history`, `bg_var_threshold`, `bg_detect_shadows` | MOG2 background subtractor parameters. |
| `diff_threshold`, `threshold_max`, `dilate_iterations` | Mask thresholding and cleanup. |
| `gaussian_blur_kernel`, `gaussian_blur_sigma` | Pre-processing blur. |
| `bbox_color`, `bbox_thickness`, `label_*` | Bounding box and label styling. |
| `recording_fps`, `video_codec` | Output video frame rate and codec. |
| `post_motion_frames` | Frames to keep recording after motion stops. |

## Notes

- On macOS you may need to grant camera permission to your terminal/IDE the
  first time you run the app.
- Output clips are excluded from version control via `.gitignore`.
