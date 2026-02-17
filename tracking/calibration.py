import cv2
import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time

from PyQt6.QtCore import QThread, pyqtSignal


class CalibrationThread(QThread):
    progress = pyqtSignal(int, int)            # (current_frame, total_frames)
    frame_ready = pyqtSignal(np.ndarray)       # BGR frame for display
    calibration_done = pyqtSignal(float, float) # (center_x, center_y)
    calibration_failed = pyqtSignal(str)

    def __init__(self, camera_index=0, num_frames=30, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.num_frames = num_frames
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
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

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.calibration_failed.emit(f"Cannot open camera {self.camera_index}")
            return

        calibration_samples = []
        frame_timestamp_ms = 0

        for i in range(self.num_frames):
            if not self._running:
                break

            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb_frame)
            results = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 33

            if results.face_landmarks:
                nose = results.face_landmarks[0][1]
                calibration_samples.append((nose.x, nose.y))

                # Draw nose on frame
                h, w = frame.shape[:2]
                nose_px = (int(nose.x * w), int(nose.y * h))
                cv2.circle(frame, nose_px, 8, (0, 255, 0), -1)
                cv2.putText(frame, "NOSE DETECTED", (nose_px[0] + 10, nose_px[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                h, w = frame.shape[:2]
                cv2.putText(frame, "NO FACE DETECTED", (w // 2 - 100, h // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            self.frame_ready.emit(frame)
            self.progress.emit(i + 1, self.num_frames)
            time.sleep(0.033)

        cap.release()
        face_landmarker.close()

        if len(calibration_samples) == 0:
            self.calibration_failed.emit("No face detected during calibration. Make sure your face is visible.")
        else:
            center_x = float(np.mean([s[0] for s in calibration_samples]))
            center_y = float(np.mean([s[1] for s in calibration_samples]))
            self.calibration_done.emit(center_x, center_y)
