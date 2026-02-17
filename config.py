import json
import os


class ConfigManager:
    DEFAULTS = {
        "THUMB_INDEX_THRESHOLD": 0.07,
        "THUMB_INDEX_MIN_THRESHOLD": 0.01,
        "THUMB_MIDDLE_THRESHOLD": 0.07,
        "THUMB_MIDDLE_MIN_THRESHOLD": 0.01,
        "THUMB_PINKIE_THRESHOLD": 0.05,
        "THUMB_PINKIE_MIN_THRESHOLD": 0.01,
        "FIST_CURLED_FINGERS_AMOUNT": 3,
        "FIST_THUMB_INDEX_RIGHT_CLICK": True,
        "SNAP_TIME_WINDOW_SECONDS": 1.0,
        "SNAP_DISTANCE_THRESHOLD": 0.15,
        "SNAP_MODE": "thumb",
        "SENSITIVITY_MULTIPLIER": 1.0,
        "FPS": 20,
        "SWIPE_VELOCITY_THRESHOLD": 0.5,
    }

    DESCRIPTIONS = {
        "THUMB_INDEX_THRESHOLD": "Max distance for thumb-to-index touch (left click). Lower = fingers must be closer.",
        "THUMB_INDEX_MIN_THRESHOLD": "Min distance for thumb-to-index touch. Prevents triggering when fingers fully overlap.",
        "THUMB_MIDDLE_THRESHOLD": "Max distance for thumb-to-middle touch (snap initiation). Lower = fingers must be closer.",
        "THUMB_MIDDLE_MIN_THRESHOLD": "Min distance for thumb-to-middle touch. Prevents triggering when fingers fully overlap.",
        "THUMB_PINKIE_THRESHOLD": "Max distance for thumb-to-pinkie touch (right click). Lower = fingers must be closer.",
        "THUMB_PINKIE_MIN_THRESHOLD": "Min distance for thumb-to-pinkie touch. Prevents triggering when fingers fully overlap.",
        "FIST_CURLED_FINGERS_AMOUNT": "Minimum curled fingers (out of 4) to detect a fist pose.",
        "FIST_THUMB_INDEX_RIGHT_CLICK": "Enable right-click via thumb+index while hand is in fist pose.",
        "SNAP_TIME_WINDOW_SECONDS": "Seconds to wait after releasing thumb-middle before measuring snap distance.",
        "SNAP_DISTANCE_THRESHOLD": "Min distance change to trigger snap. Thumb mode: how far from thumb. Palm mode: how close to palm.",
        "SNAP_MODE": "Snap measurement mode. 'thumb' = middle moves away from thumb. 'palm' = middle moves closer to palm.",
        "SENSITIVITY_MULTIPLIER": "Head movement sensitivity multiplier. Higher = faster cursor.",
        "FPS": "Target frames per second for the tracking loop.",
        "SWIPE_VELOCITY_THRESHOLD": "Minimum hand velocity to trigger a workspace switch swipe. Lower = easier to trigger.",
    }

    DISPLAY_NAMES = {
        "THUMB_INDEX_THRESHOLD": "Thumb-Index Max",
        "THUMB_INDEX_MIN_THRESHOLD": "Thumb-Index Min",
        "THUMB_MIDDLE_THRESHOLD": "Thumb-Middle Max",
        "THUMB_MIDDLE_MIN_THRESHOLD": "Thumb-Middle Min",
        "THUMB_PINKIE_THRESHOLD": "Thumb-Pinkie Max",
        "THUMB_PINKIE_MIN_THRESHOLD": "Thumb-Pinkie Min",
        "FIST_CURLED_FINGERS_AMOUNT": "Fist Curled Fingers",
        "FIST_THUMB_INDEX_RIGHT_CLICK": "Fist Right Click",
        "SNAP_TIME_WINDOW_SECONDS": "Snap Time Window",
        "SNAP_DISTANCE_THRESHOLD": "Snap Distance",
        "SNAP_MODE": "Snap Mode",
        "SENSITIVITY_MULTIPLIER": "Sensitivity Multiplier",
        "FPS": "FPS",
        "SWIPE_VELOCITY_THRESHOLD": "Swipe Velocity",
    }

    VALUE_RANGES = {
        "THUMB_INDEX_THRESHOLD": (0.01, 0.20, 0.01),
        "THUMB_INDEX_MIN_THRESHOLD": (0.00, 0.10, 0.005),
        "THUMB_MIDDLE_THRESHOLD": (0.01, 0.20, 0.01),
        "THUMB_MIDDLE_MIN_THRESHOLD": (0.00, 0.10, 0.005),
        "THUMB_PINKIE_THRESHOLD": (0.01, 0.20, 0.01),
        "THUMB_PINKIE_MIN_THRESHOLD": (0.00, 0.10, 0.005),
        "FIST_CURLED_FINGERS_AMOUNT": (1, 4, 1),
        "SNAP_TIME_WINDOW_SECONDS": (0.1, 3.0, 0.1),
        "SNAP_DISTANCE_THRESHOLD": (0.05, 0.5, 0.01),
        "SENSITIVITY_MULTIPLIER": (0.5, 10.0, 0.5),
        "FPS": (5, 120, 5),
        "SWIPE_VELOCITY_THRESHOLD": (0.1, 2.0, 0.1),
    }

    VALUE_OPTIONS = {
        "SNAP_MODE": ["thumb", "palm"],
    }

    def __init__(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        self.path = path
        self._config = self.load()

    def load(self):
        if not os.path.exists(self.path):
            self.save(self.DEFAULTS)
            return dict(self.DEFAULTS)

        with open(self.path) as f:
            config = json.load(f)

        # Merge missing keys from defaults
        changed = False
        for key, value in self.DEFAULTS.items():
            if key not in config:
                config[key] = value
                changed = True
        if changed:
            self.save(config)
        return config

    def save(self, config=None):
        if config is not None:
            self._config = config
        with open(self.path, 'w') as f:
            json.dump(self._config, f, indent=4)

    def get(self, key):
        return self._config.get(key, self.DEFAULTS.get(key))

    def set(self, key, value):
        self._config[key] = value

    def get_all(self):
        return dict(self._config)
