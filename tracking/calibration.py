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
            model_asset_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'face_landmarker.task')
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

            h, w = frame.shape[:2]
            if results.face_landmarks:
                nose = results.face_landmarks[0][1]
                calibration_samples.append((nose.x, nose.y))

                # Reticle: dot + ring + gapped crosshair (cyan)
                _C = (255, 212, 0)   # cyan in BGR
                nx, ny = int(nose.x * w), int(nose.y * h)
                cv2.circle(frame, (nx, ny), 3, _C, -1, cv2.LINE_AA)
                cv2.circle(frame, (nx, ny), 9, _C,  1, cv2.LINE_AA)
                g = 5
                cv2.line(frame, (nx-16, ny), (nx-g, ny), _C, 1, cv2.LINE_AA)
                cv2.line(frame, (nx+g,  ny), (nx+16,ny), _C, 1, cv2.LINE_AA)
                cv2.line(frame, (nx, ny-16), (nx, ny-g), _C, 1, cv2.LINE_AA)
                cv2.line(frame, (nx, ny+g),  (nx, ny+16),_C, 1, cv2.LINE_AA)
                # Small label
                cv2.putText(frame, f"LOCKED  {len(calibration_samples)}/{self.num_frames}",
                            (nx+14, ny-2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.40, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"LOCKED  {len(calibration_samples)}/{self.num_frames}",
                            (nx+14, ny-2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.40, _C, 1, cv2.LINE_AA)
            else:
                msg = "NO FACE DETECTED"
                (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                tx, ty = (w - tw) // 2, h // 2
                cv2.putText(frame, msg, (tx+1, ty+1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(frame, msg, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (85, 34, 255), 1, cv2.LINE_AA)

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
