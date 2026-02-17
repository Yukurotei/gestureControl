import cv2
import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
import pyautogui
from collections import deque

from PyQt6.QtCore import QThread, pyqtSignal

from tracking.gestures import are_fingers_touching, is_fist, finger_distance

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


class TrackingThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, camera_index, center_x, center_y, config, device_manager, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.center_x = center_x
        self.center_y = center_y
        self._config = config.copy()
        self.device_manager = device_manager
        self._running = False

    def stop(self):
        self._running = False

    def update_config(self, config):
        self._config = config.copy()

    def run(self):
        self._running = True

        # Load MediaPipe models
        face_base_options = python.BaseOptions(
            model_asset_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'face_landmarker.task')
        )
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        hand_base_options = python.BaseOptions(
            model_asset_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'hand_landmarker.task')
        )
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
        )
        hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error.emit(f"Cannot open camera {self.camera_index}")
            return

        screen_width, screen_height = pyautogui.size()

        # Tracking state
        smooth_x, smooth_y = screen_width / 2, screen_height / 2
        smoothing_factor = 0.3
        self.device_manager.last_mouse_x = screen_width / 2
        self.device_manager.last_mouse_y = screen_height / 2

        # Gesture state
        hand_history = deque(maxlen=10)
        last_gesture_time = 0
        last_gesture_detected = None
        last_gesture_display_time = 0
        gesture_cooldown = 1.0
        gesture_display_duration = 0.8

        # Click state
        thumb_index_touching = False
        thumb_middle_touching = False
        thumb_pinkie_touching = False
        left_button_held = False
        right_button_held = False
        last_thumb_middle_touch_time = 0
        snap_last_touch_dist = None  # thumb-middle distance when last touching
        snap_release_time = None     # when thumb-middle released

        frame_timestamp_ms = 0

        while self._running:
            start_time = time.time()

            # Read config atomically
            cfg = self._config
            target_fps = cfg["FPS"]
            sensitivity_x = 2.5 * cfg["SENSITIVITY_MULTIPLIER"]
            sensitivity_y = 2.5 * cfg["SENSITIVITY_MULTIPLIER"]
            thumb_index_thresh = cfg["THUMB_INDEX_THRESHOLD"]
            thumb_index_min = cfg.get("THUMB_INDEX_MIN_THRESHOLD", 0.0)
            thumb_middle_thresh = cfg["THUMB_MIDDLE_THRESHOLD"]
            thumb_middle_min = cfg.get("THUMB_MIDDLE_MIN_THRESHOLD", 0.0)
            thumb_pinkie_thresh = cfg["THUMB_PINKIE_THRESHOLD"]
            thumb_pinkie_min = cfg.get("THUMB_PINKIE_MIN_THRESHOLD", 0.0)
            fist_curled_fingers = cfg["FIST_CURLED_FINGERS_AMOUNT"]
            fist_right_click = cfg["FIST_THUMB_INDEX_RIGHT_CLICK"]
            snap_time_window = cfg["SNAP_TIME_WINDOW_SECONDS"]
            snap_distance_threshold = cfg.get("SNAP_DISTANCE_THRESHOLD", 0.15)
            snap_mode = cfg.get("SNAP_MODE", "thumb")
            frame_delay = 1.0 / target_fps

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            debug_frame = frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb_frame)

            # Face tracking
            face_results = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            face_detected = False

            if face_results.face_landmarks:
                face_detected = True
                nose = face_results.face_landmarks[0][1]
                offset_x = (nose.x - self.center_x) * sensitivity_x
                offset_y = (nose.y - self.center_y) * sensitivity_y
                target_x = (screen_width / 2) + (offset_x * screen_width)
                target_y = (screen_height / 2) + (offset_y * screen_height)
                smooth_x = smooth_x * (1 - smoothing_factor) + target_x * smoothing_factor
                smooth_y = smooth_y * (1 - smoothing_factor) + target_y * smoothing_factor
                self.device_manager.move_mouse(smooth_x, smooth_y)

                h, w = debug_frame.shape[:2]
                nose_px = (int(nose.x * w), int(nose.y * h))
                cv2.circle(debug_frame, nose_px, 8, (0, 255, 0), -1)
                cv2.putText(debug_frame, "NOSE", (nose_px[0] + 10, nose_px[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Hand tracking
            hand_results = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += int(1000 / target_fps)

            gesture_detected = None
            h, w = debug_frame.shape[:2]
            hand_count = len(hand_results.hand_landmarks) if hand_results.hand_landmarks else 0
            is_fist_status = False

            if hand_results.hand_landmarks and len(hand_results.hand_landmarks) > 0:
                # Draw all detected hands
                for hand_idx, hand_landmarks in enumerate(hand_results.hand_landmarks):
                    for connection in HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        start = hand_landmarks[start_idx]
                        end = hand_landmarks[end_idx]
                        start_px = (int(start.x * w), int(start.y * h))
                        end_px = (int(end.x * w), int(end.y * h))
                        cv2.line(debug_frame, start_px, end_px, (255, 200, 0), 2)

                    for idx, landmark in enumerate(hand_landmarks):
                        px = (int(landmark.x * w), int(landmark.y * h))
                        if idx == 0:
                            cv2.circle(debug_frame, px, 8, (0, 255, 255), -1)
                        elif idx in [4, 8, 12, 20]:
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

                    thumb_index_dist = finger_distance(thumb_tip, index_tip)
                    thumb_middle_dist = finger_distance(thumb_tip, middle_tip)
                    thumb_pinkie_dist = finger_distance(thumb_tip, pinkie_tip)

                    if are_fingers_touching(thumb_tip, index_tip, thumb_index_thresh, thumb_index_min):
                        hand_is_fist_viz = is_fist(hand_landmarks, fist_curled_fingers)
                        if hand_is_fist_viz and fist_right_click:
                            cv2.line(debug_frame, thumb_px, index_px, (255, 0, 255), 5)
                            label = "RIGHT HOLDING (FIST)" if right_button_held else "RIGHT CLICK (FIST)"
                            cv2.putText(debug_frame, f"{label} ({thumb_index_dist:.3f})",
                                       (index_px[0] + 10, index_px[1] - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        elif hand_is_fist_viz and not fist_right_click:
                            cv2.line(debug_frame, thumb_px, index_px, (100, 100, 100), 3)
                            cv2.putText(debug_frame, f"TOUCH (FIST - DISABLED) ({thumb_index_dist:.3f})",
                                       (index_px[0] + 10, index_px[1] - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
                        else:
                            cv2.line(debug_frame, thumb_px, index_px, (0, 255, 0), 5)
                            label = "LEFT HOLDING" if left_button_held else "LEFT CLICK"
                            cv2.putText(debug_frame, f"{label} ({thumb_index_dist:.3f})",
                                       (index_px[0] + 10, index_px[1] - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(debug_frame, f"T-I: {thumb_index_dist:.3f} / {thumb_index_thresh:.3f}",
                                   (index_px[0] + 10, index_px[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                    if are_fingers_touching(thumb_tip, middle_tip, thumb_middle_thresh, thumb_middle_min):
                        cv2.line(debug_frame, thumb_px, middle_px, (255, 165, 0), 5)
                        cv2.putText(debug_frame, f"SNAP READY ({thumb_middle_dist:.3f})",
                                   (middle_px[0] + 10, middle_px[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    else:
                        cv2.putText(debug_frame, f"T-M: {thumb_middle_dist:.3f} / {thumb_middle_thresh:.3f}",
                                   (middle_px[0] + 10, middle_px[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                    if are_fingers_touching(thumb_tip, pinkie_tip, thumb_pinkie_thresh, thumb_pinkie_min):
                        cv2.line(debug_frame, thumb_px, pinkie_px, (255, 0, 255), 5)
                        label = "RIGHT HOLDING" if right_button_held else "RIGHT CLICK"
                        cv2.putText(debug_frame, f"{label} ({thumb_pinkie_dist:.3f})",
                                   (pinkie_px[0] + 10, pinkie_px[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    else:
                        cv2.putText(debug_frame, f"T-P: {thumb_pinkie_dist:.3f} / {thumb_pinkie_thresh:.3f}",
                                   (pinkie_px[0] + 10, pinkie_px[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                    # Palm debug
                    palm_px = (int(hand_landmarks[0].x * w), int(hand_landmarks[0].y * h))
                    palm_mid_dist = finger_distance(middle_tip, hand_landmarks[0])
                    cv2.circle(debug_frame, palm_px, 8, (0, 200, 200), -1)
                    cv2.putText(debug_frame, f"Palm-M: {palm_mid_dist:.3f}",
                               (palm_px[0] + 10, palm_px[1] + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
                    if snap_release_time is not None:
                        cv2.line(debug_frame, palm_px, middle_px, (0, 255, 255), 2)

                    # Handedness label
                    if hand_results.handedness and hand_idx < len(hand_results.handedness):
                        handedness = hand_results.handedness[hand_idx][0].category_name
                        handedness = "Right" if handedness == "Left" else "Left"
                        wrist = hand_landmarks[0]
                        wrist_px = (int(wrist.x * w), int(wrist.y * h))
                        is_fist_pose = is_fist(hand_landmarks, fist_curled_fingers)
                        hand_label = f"{handedness.upper()} {'FIST' if is_fist_pose else 'OPEN'}"
                        label_color = (0, 255, 0) if is_fist_pose else (255, 255, 0)
                        cv2.putText(debug_frame, hand_label,
                                   (wrist_px[0] - 30, wrist_px[1] - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

                # Use first detected hand for gestures
                first_hand = hand_results.hand_landmarks[0]
                wrist = first_hand[0]
                palm_center = first_hand[0]  # wrist
                thumb_tip = first_hand[4]
                index_tip = first_hand[8]
                middle_tip = first_hand[12]
                pinkie_tip = first_hand[20]

                hand_is_fist = is_fist(first_hand, fist_curled_fingers)
                is_fist_status = hand_is_fist

                thumb_index_now = are_fingers_touching(thumb_tip, index_tip, thumb_index_thresh, thumb_index_min)
                thumb_middle_now = are_fingers_touching(thumb_tip, middle_tip, thumb_middle_thresh, thumb_middle_min)
                thumb_pinkie_now = are_fingers_touching(thumb_tip, pinkie_tip, thumb_pinkie_thresh, thumb_pinkie_min)

                current_time = time.time()

                # Palm distance (middle tip to palm center)
                palm_middle_dist = finger_distance(middle_tip, palm_center)
                palm_px = (int(palm_center.x * w), int(palm_center.y * h))

                # Snap gesture (only when open hand)
                if not hand_is_fist:
                    if thumb_middle_now:
                        # While holding, store reference distance
                        if snap_mode == "palm":
                            snap_last_touch_dist = palm_middle_dist
                        else:
                            snap_last_touch_dist = thumb_middle_dist # type: ignore
                        snap_release_time = None
                    elif thumb_middle_touching and not thumb_middle_now:
                        # Just released — start timer
                        snap_release_time = current_time

                    # Check snap after release + timer
                    if snap_release_time is not None and snap_last_touch_dist is not None:
                        elapsed = current_time - snap_release_time
                        if elapsed >= snap_time_window:
                            if snap_mode == "palm":
                                # Palm mode: middle finger must get CLOSER to palm
                                dist_diff = snap_last_touch_dist - palm_middle_dist
                            else:
                                # Thumb mode: middle finger must get FARTHER from thumb
                                dist_diff = thumb_middle_dist - snap_last_touch_dist # type: ignore
                            if dist_diff >= snap_distance_threshold:
                                self.device_manager.send_close_window()
                                last_gesture_detected = 'SNAP - CLOSE'
                                last_gesture_display_time = current_time
                            snap_release_time = None
                            snap_last_touch_dist = None

                # Click detection
                if not hand_is_fist:
                    if thumb_index_now and not left_button_held:
                        self.device_manager.press_mouse_button('left')
                        left_button_held = True
                        last_gesture_detected = 'LEFT CLICK'
                        last_gesture_display_time = current_time
                    elif not thumb_index_now and left_button_held:
                        self.device_manager.release_mouse_button('left')
                        left_button_held = False

                    if thumb_pinkie_now and not right_button_held:
                        self.device_manager.press_mouse_button('right')
                        right_button_held = True
                        last_gesture_detected = 'RIGHT CLICK'
                        last_gesture_display_time = current_time
                    elif not thumb_pinkie_now and right_button_held:
                        self.device_manager.release_mouse_button('right')
                        right_button_held = False
                else:
                    if left_button_held:
                        self.device_manager.release_mouse_button('left')
                        left_button_held = False

                    if fist_right_click:
                        if thumb_index_now and not right_button_held:
                            self.device_manager.press_mouse_button('right')
                            right_button_held = True
                            last_gesture_detected = 'RIGHT CLICK (FIST)'
                            last_gesture_display_time = current_time
                        elif not thumb_index_now and right_button_held:
                            self.device_manager.release_mouse_button('right')
                            right_button_held = False

                thumb_index_touching = thumb_index_now
                thumb_middle_touching = thumb_middle_now
                thumb_pinkie_touching = thumb_pinkie_now

                # Swipe tracking (fist only)
                if hand_is_fist:
                    hand_history.append((wrist.x, wrist.y, current_time))
                else:
                    hand_history.clear()

                # Draw motion trail
                if len(hand_history) > 1:
                    for i in range(len(hand_history) - 1):
                        pt1 = (int(hand_history[i][0] * w), int(hand_history[i][1] * h))
                        pt2 = (int(hand_history[i+1][0] * w), int(hand_history[i+1][1] * h))
                        alpha = i / len(hand_history)
                        color = (int(255 * alpha), int(100 * alpha), int(255 * alpha))
                        cv2.line(debug_frame, pt1, pt2, color, 3)

                # Swipe detection
                if len(hand_history) >= 8:
                    current_time = time.time()
                    if current_time - last_gesture_time > gesture_cooldown:
                        oldest_pos = hand_history[0]
                        newest_pos = hand_history[-1]
                        dx = newest_pos[0] - oldest_pos[0]
                        dy = newest_pos[1] - oldest_pos[1]
                        time_diff = newest_pos[2] - oldest_pos[2]

                        if time_diff > 0:
                            vx = dx / time_diff
                            vy = dy / time_diff
                            swipe_threshold = self._config.get('SWIPE_VELOCITY_THRESHOLD', 0.5)

                            wrist_px = (int(wrist.x * w), int(wrist.y * h))
                            vel_end = (int(wrist_px[0] + vx * 50), int(wrist_px[1] + vy * 50))
                            cv2.arrowedLine(debug_frame, wrist_px, vel_end, (0, 255, 255), 3)
                            cv2.putText(debug_frame, f"Vx: {vx:.2f} Vy: {vy:.2f}",
                                       (wrist_px[0] + 10, wrist_px[1] + 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                            if abs(vx) > swipe_threshold and abs(vx) > abs(vy) * 1.5:
                                if vx > 0:
                                    gesture_detected = 'RIGHT'
                                    self.device_manager.send_workspace_switch('right')
                                else:
                                    gesture_detected = 'LEFT'
                                    self.device_manager.send_workspace_switch('left')
                                last_gesture_time = current_time
                                last_gesture_detected = gesture_detected
                                last_gesture_display_time = current_time
                                hand_history.clear()
            else:
                hand_history.clear()
                thumb_index_touching = False
                thumb_middle_touching = False
                thumb_pinkie_touching = False
                if left_button_held:
                    self.device_manager.release_mouse_button('left')
                    left_button_held = False
                if right_button_held:
                    self.device_manager.release_mouse_button('right')
                    right_button_held = False

            # Draw status overlay
            cv2.rectangle(debug_frame, (10, 10), (w - 10, 145), (0, 0, 0), -1)
            cv2.rectangle(debug_frame, (10, 10), (w - 10, 145), (255, 255, 255), 2)

            status_y = 35
            cv2.putText(debug_frame, f"Thresholds - Index: {thumb_index_thresh:.3f} | Pinkie: {thumb_pinkie_thresh:.3f}",
                       (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            status_y += 25
            cv2.putText(debug_frame, f"Face Tracking: {'ACTIVE' if face_detected else 'NO FACE'}",
                       (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 255, 0) if face_detected else (0, 0, 255), 2)

            status_y += 25
            fist_status = ""
            if hand_count > 0 and is_fist_status:
                fist_status = " - FIST READY!"
            cv2.putText(debug_frame, f"Hands Detected: {hand_count}{fist_status}",
                       (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 255, 0) if hand_count > 0 else (150, 150, 150), 2)

            status_y += 25
            current_time = time.time()
            cooldown_remaining = max(0, gesture_cooldown - (current_time - last_gesture_time))

            if last_gesture_detected and (current_time - last_gesture_display_time) < gesture_display_duration:
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

            self.frame_ready.emit(debug_frame)
            self.status_update.emit({
                "face_detected": face_detected,
                "hand_count": hand_count,
                "is_fist": is_fist_status,
                "last_gesture": last_gesture_detected,
            })

            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

        # Cleanup
        if left_button_held:
            self.device_manager.release_mouse_button('left')
        if right_button_held:
            self.device_manager.release_mouse_button('right')
        cap.release()
        face_landmarker.close()
        hand_landmarker.close()
