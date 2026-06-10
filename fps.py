import cv2
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Unable to open webcam")

# Get frame size from the webcam
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam frame size: {frame_width}x{frame_height}")

# Variables to store time for FPS calculation
prev_time = 0
new_time = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate real-time FPS
    new_time = time.time()
    fps = 1 / (new_time - prev_time) if (new_time - prev_time) > 0 else 0
    prev_time = new_time

    # Convert FPS to an integer for cleaner display
    fps_text = f"FPS: {int(fps)}"
    frame_size_text = f"{frame_width}x{frame_height}"

    # Overlay the FPS text and frame size onto the live video frame
    cv2.putText(
        frame,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        frame_size_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Display the resulting frame
    cv2.imshow('Webcam Live FPS', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close windows
cap.release()
cv2.destroyAllWindows()