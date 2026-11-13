"""MediaPipe Tasks wrapper.

Note on the API: mediapipe removed `mp.solutions.*` in the 0.10.3x releases.
Anything you find online using `mp.solutions.hands` will not import. This uses
the Tasks API, which needs the .task model bundles on disk. Run
scripts/download_models.sh once.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from config import HAND_TASK, POSE_TASK
from features import frame_features

RunningMode = vision.RunningMode


def _to_array(landmarks):
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)


class LandmarkExtractor:
    """Runs hand + pose landmarkers over frames and returns feature vectors.

    Use running_mode="video" when reading a file (timestamps come from the
    file), "live" when reading a webcam. Both are stateful trackers, so create
    one extractor per stream and do not share it across threads.
    """

    def __init__(self, running_mode="video", num_hands=2, min_conf=0.5):
        mode = RunningMode.VIDEO if running_mode == "video" else RunningMode.LIVE_STREAM
        if running_mode == "live":
            # LIVE_STREAM is callback based. For a Streamlit loop the simpler
            # option is VIDEO mode with a monotonically increasing timestamp.
            mode = RunningMode.VIDEO

        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(HAND_TASK)),
                running_mode=mode,
                num_hands=num_hands,
                min_hand_detection_confidence=min_conf,
                min_tracking_confidence=min_conf,
            )
        )
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(POSE_TASK)),
                running_mode=mode,
                num_poses=1,
                min_pose_detection_confidence=min_conf,
                min_tracking_confidence=min_conf,
            )
        )

        # MediaPipe rejects a timestamp that does not exceed the previous one
        # seen by this landmarker. Each video restarts its own clock at zero,
        # so the extractor keeps its own monotonic counter.
        self._last_ts = -1

    def close(self):
        self.hands.close()
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def frame(self, bgr, timestamp_ms):
        """One BGR frame in, one (282,) feature vector out."""
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_res = self.hands.detect_for_video(image, timestamp_ms)
        pose_res = self.pose.detect_for_video(image, timestamp_ms)

        left = right = None
        for i, hand in enumerate(hand_res.hand_landmarks):
            label = hand_res.handedness[i][0].category_name  # "Left" / "Right"
            if label not in ("Left", "Right"):
                # Fail loudly. A changed label would silently swap the hands
                # across a whole extraction run and the cause would be
                # invisible in the training curves.
                raise RuntimeError(
                    f"unexpected handedness label {label!r}; expected 'Left' or 'Right'"
                )
            arr = _to_array(hand)
            if label == "Left":
                left = arr
            else:
                right = arr

        pose = _to_array(pose_res.pose_landmarks[0]) if pose_res.pose_landmarks else None
        return frame_features(pose=pose, left=left, right=right)

    def video(self, path, stride=1):
        """Whole video file in, (T, 282) array out."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                out.append(self.frame(frame, int(i / fps * 1000)))
            i += 1
        cap.release()
        if not out:
            raise ValueError(f"no frames read from {path}")
        return np.stack(out)
