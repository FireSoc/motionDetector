import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

CLIPS_DIR = Path(__file__).resolve().parent / "motion_detection_clips"


def run_motion_detection(camera_index: int=0) -> None:
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # video frames with proper detection
    # apply thresholds like blurring and contouring

    cap = cv2.VideoCapture(camera_index)
    last_gray = None

    recording = False
    detected_motion = False
    out = None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    window_size = (width, height)

    frame_rec_count = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    while(True):
        detected_motion = False

        ret, frame = cap.read()
        if not ret:
            break

        ''' 
        Can apply a bounding box to frames that have difference in pixels
        '''

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        # runs once at beginning of the recording
        if last_gray is None:
            last_gray = gray
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # thresholding converts to a binary image and isolate motion areas
        diff = cv2.absdiff(last_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # contours, borders around a white region in a binary mask that returns the region's outline
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 2750:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detected_motion = True
            
        if detected_motion or recording: 
            if recording == False:
                print('Motion Detected, no recording in progress. Recording Started')
                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                clip_path = CLIPS_DIR / f'{now}.mp4'
                out = cv2.VideoWriter(str(clip_path), fourcc, 30, window_size)

                recording = True

            else:
                if detected_motion:
                    frame_rec_count = 0
                if frame_rec_count == 300:
                    out.release()
                    recording = False
                    frame_rec_count = 0

            print('Adding frame to recording')
            out.write(frame)
            frame_rec_count += 1

        last_gray = gray
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
