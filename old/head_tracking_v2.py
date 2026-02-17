import cv2
import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import pyautogui
import numpy as np
import os
from evdev import UInput, ecodes as e

print("Creating virtual mouse device...")
capabilities = {
    e.EV_REL: [e.REL_X, e.REL_Y],
    e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
}
try:
    virtual_mouse = UInput(capabilities, name='head-tracking-mouse')
    print("Virtual mouse created successfully!")
except PermissionError:
    print("ERROR: Permission denied. You may need to:")
    print("  1. Add yourself to the 'input' group: sudo usermod -a -G input $USER")
    print("  2. Log out and log back in")
    print("  3. Or run with sudo (not recommended)")
    exit(1)

# Helper function for native mouse movement
last_x, last_y = 0, 0
def move_mouse_native(target_x, target_y):
    global last_x, last_y
    # Calculate relative movement
    dx = int(target_x - last_x)
    dy = int(target_y - last_y)

    if dx != 0:
        virtual_mouse.write(e.EV_REL, e.REL_X, dx)
    if dy != 0:
        virtual_mouse.write(e.EV_REL, e.REL_Y, dy)
    if dx != 0 or dy != 0:
        virtual_mouse.syn()

    last_x = target_x
    last_y = target_y


base_options = python.BaseOptions(model_asset_path=os.path.expanduser('~/face_landmarker.task'))
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)
screen_width, screen_height = pyautogui.size()
print(f"Screen resolution: {screen_width}x{screen_height}")

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame_height, frame_width = frame.shape[:2]
window_name = 'Head Tracking Calibration'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

for countdown in range(3, 0, -1):
    instruction_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    cv2.putText(instruction_frame, "HEAD TRACKING CALIBRATION",
               (screen_width//2 - 400, screen_height//2 - 200),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    cv2.putText(instruction_frame, "Look at this text and keep your head still!",
               (screen_width//2 - 450, screen_height//2 - 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.putText(instruction_frame, f"Starting in {countdown}...",
               (screen_width//2 - 200, screen_height//2 + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)

    cv2.circle(instruction_frame, (screen_width//2, screen_height//2), 20, (0, 0, 255), -1)
    cv2.circle(instruction_frame, (screen_width//2, screen_height//2), 5, (255, 255, 255), -1)

    cv2.imshow(window_name, instruction_frame)
    cv2.waitKey(1000)

calibration_samples = []
frame_timestamp_ms = 0
for i in range(30):
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb_frame)
    results = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    frame_timestamp_ms += 33

    if results.face_landmarks:
        face_landmarks = results.face_landmarks[0]
        nose = face_landmarks[1]
        calibration_samples.append((nose.x, nose.y))

    # Show progress
    instruction_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    cv2.putText(instruction_frame, "CALIBRATING...",
               (screen_width//2 - 200, screen_height//2 - 100),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(instruction_frame, f"Hold still! {i+1}/30",
               (screen_width//2 - 150, screen_height//2),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.circle(instruction_frame, (screen_width//2, screen_height//2), 20, (0, 0, 255), -1)
    cv2.circle(instruction_frame, (screen_width//2, screen_height//2), 5, (255, 255, 255), -1)
    cv2.imshow(window_name, instruction_frame)
    cv2.waitKey(1)

    time.sleep(0.033)

if len(calibration_samples) == 0:
    print("ERROR: No face detected during calibration!")
    exit(1)

center_x = np.mean([s[0] for s in calibration_samples])
center_y = np.mean([s[1] for s in calibration_samples])

# Show success message
success_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
cv2.putText(success_frame, "CALIBRATION COMPLETE!",
           (screen_width//2 - 350, screen_height//2 - 100),
           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
cv2.putText(success_frame, "Starting head tracking...",
           (screen_width//2 - 280, screen_height//2),
           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
cv2.putText(success_frame, "Move your head to control the cursor",
           (screen_width//2 - 350, screen_height//2 + 100),
           cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
cv2.imshow(window_name, success_frame)
cv2.waitKey(2000)

# Close calibration window
cv2.destroyAllWindows()

print(f"\nCalibration complete! Center: ({center_x:.3f}, {center_y:.3f})")
print("Tracking started! Move your head to control the cursor.")
print("Press Ctrl+C to quit\n")

# Smoothing
smooth_x, smooth_y = screen_width / 2, screen_height / 2
smoothing_factor = 0.3

# Sensitivity settings
sensitivity_x = 2.5
sensitivity_y = 2.5

# FPS limit
target_fps = 30
frame_delay = 1.0 / target_fps

# Initialize last position for relative movement
last_x, last_y = screen_width / 2, screen_height / 2

try:
    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb_frame)
        results = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += int((1000 / target_fps))

        if results.face_landmarks:
            face_landmarks = results.face_landmarks[0]
            nose = face_landmarks[1]

            # Calculate offset from center
            offset_x = (nose.x - center_x) * sensitivity_x
            offset_y = (nose.y - center_y) * sensitivity_y

            # Map to screen coordinates
            target_x = (screen_width / 2) + (offset_x * screen_width)
            target_y = (screen_height / 2) + (offset_y * screen_height)

            # Apply smoothing
            smooth_x = smooth_x * (1 - smoothing_factor) + target_x * smoothing_factor
            smooth_y = smooth_y * (1 - smoothing_factor) + target_y * smoothing_factor

            # Clamp to screen bounds
            margin = 10
            smooth_x = max(margin, min(smooth_x, screen_width - margin))
            smooth_y = max(margin, min(smooth_y, screen_height - margin))

            # Move cursor using native evdev
            move_mouse_native(smooth_x, smooth_y)

        # Cap frame rate
        elapsed = time.time() - start_time
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

except KeyboardInterrupt:
    print("\nStopped by user")

# Cleanup
cap.release()
virtual_mouse.close()
print("Done!")
