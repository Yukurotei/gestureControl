from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3
import pyautogui
import subprocess
import time
import cv2
import numpy as np

# Helper function for Wayland mouse movement
def move_mouse_wayland(x, y):
    subprocess.run(['ydotool', 'mousemove', '-a', str(int(x)), str(int(y))],
                   check=False, capture_output=True, timeout=0.1)

# Initialize
gestures = EyeGestures_v3()
cap = VideoCapture(0)
screen_width, screen_height = pyautogui.size()

print(f"Screen resolution: {screen_width}x{screen_height}")
print("\n=== CALIBRATION MODE ===")
print("Look at the RED CIRCLE when it appears on your screen")
print("Keep looking at it until it disappears")
print("Press ESC to quit\n")

# Calibration points (16-point calibration grid for better accuracy)
calibration_points = [
    # Top row
    (screen_width * 0.1, screen_height * 0.1),
    (screen_width * 0.35, screen_height * 0.1),
    (screen_width * 0.65, screen_height * 0.1),
    (screen_width * 0.9, screen_height * 0.1),
    # Upper-middle row
    (screen_width * 0.1, screen_height * 0.35),
    (screen_width * 0.5, screen_height * 0.35),
    (screen_width * 0.9, screen_height * 0.35),
    # Lower-middle row
    (screen_width * 0.1, screen_height * 0.65),
    (screen_width * 0.5, screen_height * 0.65),
    (screen_width * 0.9, screen_height * 0.65),
    # Bottom row
    (screen_width * 0.1, screen_height * 0.9),
    (screen_width * 0.35, screen_height * 0.9),
    (screen_width * 0.65, screen_height * 0.9),
    (screen_width * 0.9, screen_height * 0.9),
    # Add center point last (most important)
    (screen_width * 0.5, screen_height * 0.5),
    (screen_width * 0.5, screen_height * 0.5),  # Repeat center for extra data
]

print("IMPORTANT: Keep your head COMPLETELY STILL during calibration!")
print("Position yourself comfortably before starting.\n")
time.sleep(3)  # Give user time to read and position

# Create fullscreen calibration window
window_name = 'Calibration'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Calibration phase
calibration_duration = 3.5  # seconds per point (longer for better data)
for i, (cal_x, cal_y) in enumerate(calibration_points):
    print(f"Calibrating point {i+1}/{len(calibration_points)}: ({int(cal_x)}, {int(cal_y)})")

    start_time = time.time()
    while time.time() - start_time < calibration_duration:
        ret, frame = cap.read()
        if not ret:
            continue

        # Create fullscreen black canvas
        display_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        # Draw calibration point (large red circle)
        cv2.circle(display_frame, (int(cal_x), int(cal_y)), 40, (0, 0, 255), -1)
        cv2.circle(display_frame, (int(cal_x), int(cal_y)), 10, (255, 255, 255), -1)

        # Draw progress info
        remaining = calibration_duration - (time.time() - start_time)
        cv2.putText(display_frame, f"Point {i+1}/{len(calibration_points)} - Look at the circle",
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, f"{remaining:.1f}s remaining",
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.imshow(window_name, display_frame)

        # Process frame with calibration enabled
        event, cevent = gestures.step(frame, True, cal_x, cal_y, context="calibration")

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    time.sleep(0.5)  # Brief pause between points

print("\n=== CALIBRATION COMPLETE ===")
print("Starting eye tracking mode...")
print("The window will close. Press Ctrl+C in terminal to quit.\n")

# Close fullscreen window
cv2.destroyAllWindows()
time.sleep(0.5)

# Set FPS cap
target_fps = 20
frame_delay = 1.0 / target_fps

# Tracking phase (no window, just mouse control)
try:
    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Use calibration data (calibrate=False means use the calibration we just did)
        event, cevent = gestures.step(frame, False, screen_width, screen_height,
                                      context="calibration")

        if event is not None and event:
            cursor_x, cursor_y = event.point[0], event.point[1]

            # Add boundary margins
            margin = 50
            cursor_x = max(margin, min(cursor_x, screen_width - margin))
            cursor_y = max(margin, min(cursor_y, screen_height - margin))

            # Move mouse
            move_mouse_wayland(cursor_x, cursor_y)

        # Cap frame rate
        elapsed = time.time() - start_time
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

except KeyboardInterrupt:
    print("\nStopped by user")

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Done!")
