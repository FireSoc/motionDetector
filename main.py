import cv2
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

CLIPS_DIR = Path(__file__).resolve().parent / "motion_detection_clips"
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

with CONFIG_PATH.open() as f:
    CONFIG = json.load(f)

CONFIG["bbox_color"] = tuple(CONFIG["bbox_color"])
CONFIG["label_text_color"] = tuple(CONFIG["label_text_color"])


def run_motion_detection(camera_index: int=0) -> None:
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # video frames with proper detection
    # apply thresholds like blurring and contouring

    cap = cv2.VideoCapture(camera_index)
    model = YOLO(CONFIG["model_path"])
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=CONFIG["bg_history"],
        varThreshold=CONFIG["bg_var_threshold"],
        detectShadows=CONFIG["bg_detect_shadows"],
    )

    recording = False
    detected_motion = False
    out = None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    window_size = (width, height)

    frame_rec_count = 0
    fourcc = cv2.VideoWriter_fourcc(*CONFIG["video_codec"])

    while (True):
        detected_motion = False

        ret, frame = cap.read()
        if not ret:
            break

        
        # transforming to grayscale and blurring
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (CONFIG["gaussian_blur_kernel"], CONFIG["gaussian_blur_kernel"]), CONFIG["gaussian_blur_sigma"])

        # now we use MOG2 (mixture of Gaussians) to comapre the current gray to the background per pixel
        fg_mask = bg_subtractor.apply(gray)

        # thresholding values with pxels above 25 to become 0 or 255
        _, thresh = cv2.threshold(fg_mask, CONFIG["diff_threshold"], CONFIG["threshold_max"], cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=CONFIG["dilate_iterations"])

        # contours, borders around a white region in a binary mask that returns the region's outline
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < CONFIG["min_contour_area"]:
                continue
            motion_boxes.append(cv2.boundingRect(contour))

        # if there is motion, than YOLO runs on the original frame and we alter to show bb
        if motion_boxes:
            results = model(frame, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = result.names[int(box.cls[0])]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), CONFIG["bbox_color"], CONFIG["bbox_thickness"])
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, CONFIG["label_font_scale"], CONFIG["label_font_thickness"])
                    cv2.rectangle(frame, (x1, y1 - text_h - CONFIG["label_bg_padding"]), (x1 + text_w, y1), CONFIG["bbox_color"], -1)
                    cv2.putText(frame, label, (x1, y1 - CONFIG["label_text_y_offset"]), cv2.FONT_HERSHEY_SIMPLEX, CONFIG["label_font_scale"], CONFIG["label_text_color"], CONFIG["label_font_thickness"])
            if len(results[0].boxes) > 0:
                detected_motion = True
            
        if detected_motion or recording: 
            if recording == False:
                print('Motion Detected, no recording in progress. Recording Started')
                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                clip_path = CLIPS_DIR / f'{now}.mp4'
                out = cv2.VideoWriter(str(clip_path), fourcc, CONFIG["recording_fps"], window_size)

                recording = True

            else:
                if detected_motion:
                    frame_rec_count = 0
                if frame_rec_count == CONFIG["post_motion_frames"]:
                    out.release()
                    recording = False
                    frame_rec_count = 0

            print('Adding frame to recording')
            out.write(frame)
            frame_rec_count += 1

        cv2.imshow('frame', frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            if out is not None:
                out.release()
                print('released')
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_motion_detection()
