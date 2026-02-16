import cv2
import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import pyautogui
import numpy as np
import os
import json
from evdev import UInput, ecodes as e
from collections import deque

# Load config (create with defaults if missing)
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
default_config = {
    "THUMB_INDEX_THRESHOLD": 0.07,
    "THUMB_MIDDLE_THRESHOLD": 0.07,
    "THUMB_PINKIE_THRESHOLD": 0.05,
    "SENSITIVITY_MULTIPLIER": 1.0,
    "FPS": 20
}
if not os.path.exists(config_path):
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=4)
    config = default_config
else:
    with open(config_path) as f:
        config = json.load(f)

THUMB_INDEX_THRESHOLD = config["THUMB_INDEX_THRESHOLD"]
THUMB_MIDDLE_THRESHOLD = config["THUMB_MIDDLE_THRESHOLD"]
THUMB_PINKIE_THRESHOLD = config["THUMB_PINKIE_THRESHOLD"]
SENSITIVITY_MULTIPLIER = config["SENSITIVITY_MULTIPLIER"]
target_fps = config["FPS"]

print("Creating virtual input devices...")

# Virtual mouse
mouse_capabilities = {
    e.EV_REL: [e.REL_X, e.REL_Y],
    e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
}
virtual_mouse = UInput(mouse_capabilities, name='gesture-mouse')

# Virtual keyboard for shortcuts
keyboard_capabilities = {
    e.EV_KEY: [
        e.KEY_LEFTCTRL, e.KEY_LEFTMETA,  # Ctrl, Super
        e.KEY_LEFT, e.KEY_RIGHT, e.KEY_UP, e.KEY_DOWN,
        e.KEY_Q  # For close window (Super+Q)
    ]
}
virtual_keyboard = UInput(keyboard_capabilities, name='gesture-keyboard')
print("Virtual devices created successfully!")

# Helper functions
last_mouse_x, last_mouse_y = 0, 0
def move_mouse_native(target_x, target_y):
    global last_mouse_x, last_mouse_y
    dx = int(target_x - last_mouse_x)
    dy = int(target_y - last_mouse_y)
    if dx != 0:
        virtual_mouse.write(e.EV_REL, e.REL_X, dx)
    if dy != 0:
        virtual_mouse.write(e.EV_REL, e.REL_Y, dy)
    if dx != 0 or dy != 0:
        virtual_mouse.syn()
    last_mouse_x, last_mouse_y = target_x, target_y

def are_fingers_touching(landmark1, landmark2, threshold):
    """Check if two finger landmarks are touching"""
    dist = ((landmark1.x - landmark2.x)**2 +
            (landmark1.y - landmark2.y)**2 +
            (landmark1.z - landmark2.z)**2)**0.5
    return dist < threshold

def is_fist(hand_landmarks):
    """Detect if hand is in a fist pose"""
    wrist = hand_landmarks[0]

    # Fingertips (excluding thumb)
    fingertips = [
        hand_landmarks[8],   # Index
        hand_landmarks[12],  # Middle
        hand_landmarks[16],  # Ring
        hand_landmarks[20],  # Pinky
    ]

    # Finger bases (MCP joints)
    finger_bases = [
        hand_landmarks[5],   # Index MCP
        hand_landmarks[9],   # Middle MCP
        hand_landmarks[13],  # Ring MCP
        hand_landmarks[17],  # Pinky MCP
    ]

    # Check if each fingertip is curled (closer to wrist than base)
    curled_count = 0
    for tip, base in zip(fingertips, finger_bases):
        # Calculate distances to wrist
        tip_dist = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)**0.5
        base_dist = ((base.x - wrist.x)**2 + (base.y - wrist.y)**2)**0.5

        # If tip is closer to wrist than base, finger is curled
        if tip_dist < base_dist * 1.1:  # Small threshold for tolerance
            curled_count += 1

    # If at least 3 out of 4 fingers are curled, it's a fist
    return curled_count >= 3

def press_mouse_button(button='left'):
    """Press and hold mouse button"""
    button_code = e.BTN_LEFT if button == 'left' else e.BTN_RIGHT
    virtual_mouse.write(e.EV_KEY, button_code, 1)
    virtual_mouse.syn()
    print(f"🖱️  Mouse {button} PRESSED")

def release_mouse_button(button='left'):
    """Release mouse button"""
    button_code = e.BTN_LEFT if button == 'left' else e.BTN_RIGHT
    virtual_mouse.write(e.EV_KEY, button_code, 0)
    virtual_mouse.syn()
    print(f"🖱️  Mouse {button} RELEASED")

def send_workspace_switch(direction):
    """Send Hyprland workspace switch shortcut: Ctrl+Super+Arrow"""
    key_map = {
        'left': e.KEY_LEFT,
        'right': e.KEY_RIGHT,
        'up': e.KEY_UP,
        'down': e.KEY_DOWN
    }

    if direction not in key_map:
        return

    arrow_key = key_map[direction]

    # Press Ctrl+Super+Arrow
    virtual_keyboard.write(e.EV_KEY, e.KEY_LEFTCTRL, 1)
    virtual_keyboard.write(e.EV_KEY, e.KEY_LEFTMETA, 1)
    virtual_keyboard.write(e.EV_KEY, arrow_key, 1)
    virtual_keyboard.syn()

    time.sleep(0.05)

    # Release all keys
    virtual_keyboard.write(e.EV_KEY, arrow_key, 0)
    virtual_keyboard.write(e.EV_KEY, e.KEY_LEFTMETA, 0)
    virtual_keyboard.write(e.EV_KEY, e.KEY_LEFTCTRL, 0)
    virtual_keyboard.syn()

    print(f"🚀 Workspace switch: {direction}")

def send_close_window():
    """Send Hyprland close window shortcut: Super+Q"""
    # Press Super+Q
    virtual_keyboard.write(e.EV_KEY, e.KEY_LEFTMETA, 1)
    virtual_keyboard.write(e.EV_KEY, e.KEY_Q, 1)
    virtual_keyboard.syn()

    time.sleep(0.05)

    # Release all keys
    virtual_keyboard.write(e.EV_KEY, e.KEY_Q, 0)
    virtual_keyboard.write(e.EV_KEY, e.KEY_LEFTMETA, 0)
    virtual_keyboard.syn()

    print(f"💥 SNAP - Close window!")

# Initialize MediaPipe Face Landmarker
face_base_options = python.BaseOptions(
    model_asset_path=os.path.expanduser('~/face_landmarker.task')
)
face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

# Initialize MediaPipe Hand Landmarker
hand_base_options = python.BaseOptions(
    model_asset_path=os.path.expanduser('~/hand_landmarker.task')
)
hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# Get screen resolution
screen_width, screen_height = pyautogui.size()
print(f"Screen resolution: {screen_width}x{screen_height}")

# Open webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Calibration window
window_name = 'Gesture Control Calibration'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Show calibration instructions
for countdown in range(3, 0, -1):
    instruction_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    cv2.putText(instruction_frame, "GESTURE CONTROL CALIBRATION",
               (screen_width//2 - 450, screen_height//2 - 200),
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

# Calibration
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
        nose = results.face_landmarks[0][1]
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

# Success message
success_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
cv2.putText(success_frame, "CALIBRATION COMPLETE!",
           (screen_width//2 - 350, screen_height//2 - 150),
           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
cv2.putText(success_frame, "Head tracking: Move head to control cursor",
           (screen_width//2 - 450, screen_height//2 - 50),
           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(success_frame, "Hand gestures: Swipe left/right to switch workspaces",
           (screen_width//2 - 500, screen_height//2 + 50),
           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(success_frame, "Press Ctrl+C to quit",
           (screen_width//2 - 200, screen_height//2 + 150),
           cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
cv2.imshow(window_name, success_frame)
cv2.waitKey(2000)
cv2.destroyAllWindows()

print(f"\nCalibration complete!")
print("Controls:")
print("  - Move your HEAD to control the cursor")
print("  - SWIPE LEFT/RIGHT with your hand to switch workspaces")
print("  - Press Ctrl+C to quit\n")

# Gesture detection state
hand_history = deque(maxlen=10)  # Track hand position over last 10 frames
last_gesture_time = 0
last_gesture_detected = None
last_gesture_display_time = 0
gesture_cooldown = 1.0  # seconds between gestures
gesture_display_duration = 0.8  # seconds to show gesture message

# Click detection state
thumb_index_touching = False
thumb_middle_touching = False
thumb_pinkie_touching = False
left_button_held = False
right_button_held = False

# Snap gesture detection state
last_thumb_middle_touch_time = 0
snap_time_window = 1  # seconds - window to detect snap transition

# Tracking parameters
smooth_x, smooth_y = screen_width / 2, screen_height / 2
smoothing_factor = 0.3
sensitivity_x = 2.5 * SENSITIVITY_MULTIPLIER
sensitivity_y = 2.5 * SENSITIVITY_MULTIPLIER
frame_delay = 1.0 / target_fps
last_mouse_x, last_mouse_y = screen_width / 2, screen_height / 2

# Create debug window
debug_window = 'Gesture Control - Debug View'
cv2.namedWindow(debug_window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(debug_window, 800, 600)

# For drawing hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]

try:
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        debug_frame = frame.copy()  # For visualization
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb_frame)

        # Face tracking for cursor
        face_results = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if face_results.face_landmarks:
            nose = face_results.face_landmarks[0][1]
            offset_x = (nose.x - center_x) * sensitivity_x
            offset_y = (nose.y - center_y) * sensitivity_y
            target_x = (screen_width / 2) + (offset_x * screen_width)
            target_y = (screen_height / 2) + (offset_y * screen_height)
            smooth_x = smooth_x * (1 - smoothing_factor) + target_x * smoothing_factor
            smooth_y = smooth_y * (1 - smoothing_factor) + target_y * smoothing_factor
            move_mouse_native(smooth_x, smooth_y)

            # Draw nose landmark on debug frame
            h, w = debug_frame.shape[:2]
            nose_px = (int(nose.x * w), int(nose.y * h))
            cv2.circle(debug_frame, nose_px, 8, (0, 255, 0), -1)
            cv2.putText(debug_frame, "NOSE", (nose_px[0] + 10, nose_px[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Hand tracking for gestures
        hand_results = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += int((1000 / target_fps))

        gesture_detected = None
        h, w = debug_frame.shape[:2]

        if hand_results.hand_landmarks and len(hand_results.hand_landmarks) > 0:
            # Draw all detected hands
            for hand_idx, hand_landmarks in enumerate(hand_results.hand_landmarks):
                # Draw hand connections
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    start_px = (int(start.x * w), int(start.y * h))
                    end_px = (int(end.x * w), int(end.y * h))
                    cv2.line(debug_frame, start_px, end_px, (255, 200, 0), 2)

                # Draw landmarks
                for idx, landmark in enumerate(hand_landmarks):
                    px = (int(landmark.x * w), int(landmark.y * h))
                    if idx == 0:  # Wrist - larger circle
                        cv2.circle(debug_frame, px, 8, (0, 255, 255), -1)
                    elif idx in [4, 8, 12, 20]:  # Thumb, index, middle, pinkie tips - highlight
                        cv2.circle(debug_frame, px, 6, (0, 255, 0), -1)
                    else:
                        cv2.circle(debug_frame, px, 4, (255, 100, 100), -1)

                # Draw touch indicators
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]
                middle_tip = hand_landmarks[12]
                pinkie_tip = hand_landmarks[20]

                thumb_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_px = (int(index_tip.x * w), int(index_tip.y * h))
                middle_px = (int(middle_tip.x * w), int(middle_tip.y * h))
                pinkie_px = (int(pinkie_tip.x * w), int(pinkie_tip.y * h))

                # Calculate distances for debugging
                thumb_index_dist = ((thumb_tip.x - index_tip.x)**2 +
                                   (thumb_tip.y - index_tip.y)**2 +
                                   (thumb_tip.z - index_tip.z)**2)**0.5
                thumb_middle_dist = ((thumb_tip.x - middle_tip.x)**2 +
                                    (thumb_tip.y - middle_tip.y)**2 +
                                    (thumb_tip.z - middle_tip.z)**2)**0.5
                thumb_pinkie_dist = ((thumb_tip.x - pinkie_tip.x)**2 +
                                    (thumb_tip.y - pinkie_tip.y)**2 +
                                    (thumb_tip.z - pinkie_tip.z)**2)**0.5

                # Check touches and draw indicators
                if are_fingers_touching(thumb_tip, index_tip, THUMB_INDEX_THRESHOLD):
                    # Check if this hand is a fist for correct label
                    hand_is_fist_viz = is_fist(hand_landmarks)

                    if hand_is_fist_viz:
                        # Fist mode: thumb+index = right click
                        cv2.line(debug_frame, thumb_px, index_px, (255, 0, 255), 5)
                        label = "RIGHT HOLDING (FIST)" if right_button_held else "RIGHT CLICK (FIST)"
                        cv2.putText(debug_frame, f"{label} ({thumb_index_dist:.3f})",
                                   (index_px[0] + 10, index_px[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    else:
                        # Open hand mode: thumb+index = left click
                        cv2.line(debug_frame, thumb_px, index_px, (0, 255, 0), 5)
                        label = "LEFT HOLDING" if left_button_held else "LEFT CLICK"
                        cv2.putText(debug_frame, f"{label} ({thumb_index_dist:.3f})",
                                   (index_px[0] + 10, index_px[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # Always show thumb-index distance
                    cv2.putText(debug_frame, f"T-I: {thumb_index_dist:.3f} / {THUMB_INDEX_THRESHOLD:.3f}",
                               (index_px[0] + 10, index_px[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                # Thumb-middle touch indicator (for snap gesture)
                if are_fingers_touching(thumb_tip, middle_tip, THUMB_MIDDLE_THRESHOLD):
                    cv2.line(debug_frame, thumb_px, middle_px, (255, 165, 0), 5)  # Orange
                    cv2.putText(debug_frame, f"SNAP READY ({thumb_middle_dist:.3f})",
                               (middle_px[0] + 10, middle_px[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                else:
                    # Show thumb-middle distance
                    cv2.putText(debug_frame, f"T-M: {thumb_middle_dist:.3f} / {THUMB_MIDDLE_THRESHOLD:.3f}",
                               (middle_px[0] + 10, middle_px[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                if are_fingers_touching(thumb_tip, pinkie_tip, THUMB_PINKIE_THRESHOLD):
                    cv2.line(debug_frame, thumb_px, pinkie_px, (255, 0, 255), 5)
                    label = "RIGHT HOLDING" if right_button_held else "RIGHT CLICK"
                    cv2.putText(debug_frame, f"{label} ({thumb_pinkie_dist:.3f})",
                               (pinkie_px[0] + 10, pinkie_px[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                else:
                    # Always show thumb-pinkie distance
                    cv2.putText(debug_frame, f"T-P: {thumb_pinkie_dist:.3f} / {THUMB_PINKIE_THRESHOLD:.3f}",
                               (pinkie_px[0] + 10, pinkie_px[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                # Draw handedness label (flip since frame is mirrored)
                if hand_results.handedness and hand_idx < len(hand_results.handedness):
                    handedness = hand_results.handedness[hand_idx][0].category_name
                    # Flip handedness because frame is flipped
                    handedness = "Right" if handedness == "Left" else "Left"
                    wrist = hand_landmarks[0]
                    wrist_px = (int(wrist.x * w), int(wrist.y * h))

                    # Check if this hand is a fist
                    is_fist_pose = is_fist(hand_landmarks)
                    hand_label = f"{handedness.upper()} {'FIST' if is_fist_pose else 'OPEN'}"
                    label_color = (0, 255, 0) if is_fist_pose else (255, 255, 0)

                    cv2.putText(debug_frame, hand_label,
                               (wrist_px[0] - 30, wrist_px[1] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
            # Use first detected hand
            first_hand = hand_results.hand_landmarks[0]
            wrist = first_hand[0]  # Landmark 0 is wrist

            # Get finger tips
            thumb_tip = first_hand[4]
            index_tip = first_hand[8]
            middle_tip = first_hand[12]
            pinkie_tip = first_hand[20]

            # Check if hand is in fist pose (check this FIRST)
            hand_is_fist = is_fist(first_hand)

            # Detect finger touches
            thumb_index_now = are_fingers_touching(thumb_tip, index_tip, THUMB_INDEX_THRESHOLD)
            thumb_middle_now = are_fingers_touching(thumb_tip, middle_tip, THUMB_MIDDLE_THRESHOLD)
            thumb_pinkie_now = are_fingers_touching(thumb_tip, pinkie_tip, THUMB_PINKIE_THRESHOLD)

            current_time = time.time()

            # SNAP GESTURE DETECTION (thumb-middle → thumb-index quickly)
            # Track thumb-middle touches
            if thumb_middle_now and not thumb_middle_touching:
                # Thumb-middle just touched
                last_thumb_middle_touch_time = current_time

            # Detect snap: thumb-index touches within window after thumb-middle
            if thumb_index_now and not thumb_index_touching:
                time_since_middle = current_time - last_thumb_middle_touch_time
                if time_since_middle < snap_time_window and time_since_middle > 0.05:
                    # SNAP DETECTED!
                    send_close_window()
                    last_gesture_detected = 'SNAP - CLOSE'
                    last_gesture_display_time = current_time
                    # Reset to prevent re-triggering
                    last_thumb_middle_touch_time = 0

            # Click detection based on hand pose
            if not hand_is_fist:
                # OPEN HAND MODE
                # Left button: thumb + index
                if thumb_index_now and not left_button_held:
                    press_mouse_button('left')
                    left_button_held = True
                    last_gesture_detected = 'LEFT CLICK'
                    last_gesture_display_time = current_time
                elif not thumb_index_now and left_button_held:
                    release_mouse_button('left')
                    left_button_held = False

                # Right button: thumb + pinkie
                if thumb_pinkie_now and not right_button_held:
                    press_mouse_button('right')
                    right_button_held = True
                    last_gesture_detected = 'RIGHT CLICK'
                    last_gesture_display_time = current_time
                elif not thumb_pinkie_now and right_button_held:
                    release_mouse_button('right')
                    right_button_held = False
            else:
                # FIST MODE - thumb+index becomes right click
                # Release left button if it was held (hand changed to fist)
                if left_button_held:
                    release_mouse_button('left')
                    left_button_held = False

                # Right click: thumb + index (when fist)
                if thumb_index_now and not right_button_held:
                    press_mouse_button('right')
                    right_button_held = True
                    last_gesture_detected = 'RIGHT CLICK (FIST)'
                    last_gesture_display_time = current_time
                elif not thumb_index_now and right_button_held:
                    release_mouse_button('right')
                    right_button_held = False

            # Update touch states
            thumb_index_touching = thumb_index_now
            thumb_middle_touching = thumb_middle_now
            thumb_pinkie_touching = thumb_pinkie_now

            # Only track position if hand is fist
            if hand_is_fist:
                hand_history.append((wrist.x, wrist.y, current_time))
            else:
                hand_history.clear()  # Clear history if not a fist

            # Draw motion trail
            if len(hand_history) > 1:
                for i in range(len(hand_history) - 1):
                    pt1 = (int(hand_history[i][0] * w), int(hand_history[i][1] * h))
                    pt2 = (int(hand_history[i+1][0] * w), int(hand_history[i+1][1] * h))
                    alpha = i / len(hand_history)  # Fade effect
                    color = (int(255 * alpha), int(100 * alpha), int(255 * alpha))
                    cv2.line(debug_frame, pt1, pt2, color, 3)

            # Detect swipe gesture
            if len(hand_history) >= 8:  # Need at least 8 frames
                current_time = time.time()

                # Only detect if enough time has passed since last gesture
                if current_time - last_gesture_time > gesture_cooldown:
                    oldest_pos = hand_history[0]
                    newest_pos = hand_history[-1]

                    dx = newest_pos[0] - oldest_pos[0]
                    dy = newest_pos[1] - oldest_pos[1]
                    time_diff = newest_pos[2] - oldest_pos[2]

                    # Calculate velocity
                    if time_diff > 0:
                        vx = dx / time_diff
                        vy = dy / time_diff

                        # Swipe detection thresholds
                        swipe_threshold = 0.5  # velocity threshold (lower = easier to trigger)

                        # Draw velocity vector and values
                        wrist_px = (int(wrist.x * w), int(wrist.y * h))
                        vel_end = (int(wrist_px[0] + vx * 50), int(wrist_px[1] + vy * 50))
                        cv2.arrowedLine(debug_frame, wrist_px, vel_end, (0, 255, 255), 3)

                        # Show velocity values for debugging
                        cv2.putText(debug_frame, f"Vx: {vx:.2f} Vy: {vy:.2f}",
                                   (wrist_px[0] + 10, wrist_px[1] + 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(debug_frame, f"Threshold: {swipe_threshold}",
                                   (wrist_px[0] + 10, wrist_px[1] + 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # Horizontal swipes
                        if abs(vx) > swipe_threshold and abs(vx) > abs(vy) * 1.5:
                            if vx > 0:
                                gesture_detected = 'RIGHT'
                                send_workspace_switch('right')
                            else:
                                gesture_detected = 'LEFT'
                                send_workspace_switch('left')
                            last_gesture_time = current_time
                            last_gesture_detected = gesture_detected
                            last_gesture_display_time = current_time
                            hand_history.clear()
        else:
            hand_history.clear()
            # Reset touch states when no hand detected
            thumb_index_touching = False
            thumb_middle_touching = False
            thumb_pinkie_touching = False
            # Release any held buttons when hand disappears
            if left_button_held:
                release_mouse_button('left')
                left_button_held = False
            if right_button_held:
                release_mouse_button('right')
                right_button_held = False

        # Draw status overlay
        cv2.rectangle(debug_frame, (10, 10), (w - 10, 145), (0, 0, 0), -1)
        cv2.rectangle(debug_frame, (10, 10), (w - 10, 145), (255, 255, 255), 2)

        status_y = 35
        cv2.putText(debug_frame, f"Thresholds - Index: {THUMB_INDEX_THRESHOLD:.3f} | Pinkie: {THUMB_PINKIE_THRESHOLD:.3f}",
                   (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        status_y += 25
        cv2.putText(debug_frame, f"Face Tracking: {'ACTIVE' if face_results.face_landmarks else 'NO FACE'}",
                   (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if face_results.face_landmarks else (0, 0, 255), 2)

        status_y += 25
        hand_count = len(hand_results.hand_landmarks) if hand_results.hand_landmarks else 0
        fist_status = ""
        if hand_count > 0:
            first_hand = hand_results.hand_landmarks[0]
            if is_fist(first_hand):
                fist_status = " - FIST READY!"
        cv2.putText(debug_frame, f"Hands Detected: {hand_count}{fist_status}",
                   (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if hand_count > 0 else (150, 150, 150), 2)

        status_y += 25
        current_time = time.time()
        cooldown_remaining = max(0, gesture_cooldown - (current_time - last_gesture_time))

        # Show gesture message for a duration
        if last_gesture_detected and (current_time - last_gesture_display_time) < gesture_display_duration:
            # Format message based on gesture type
            if 'CLICK' in last_gesture_detected:
                message = f"{last_gesture_detected}!"
            else:
                message = f"SWIPE {last_gesture_detected}!"
            cv2.putText(debug_frame, message,
                       (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif cooldown_remaining > 0:
            cv2.putText(debug_frame, f"Cooldown: {cooldown_remaining:.1f}s",
                       (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        else:
            cv2.putText(debug_frame, "Ready for gesture",
                       (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

        # Show debug frame
        cv2.imshow(debug_window, debug_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Cap frame rate
        elapsed = time.time() - start_time
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

except KeyboardInterrupt:
    print("\nStopped by user")

# Cleanup
cap.release()
cv2.destroyAllWindows()
virtual_mouse.close()
virtual_keyboard.close()
print("Done!")
