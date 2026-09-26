import mediapipe as mp
import cv2

class UpperBodyPoseDetector:
    """Detect upper-body landmarks in a camera frame."""

    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmark_names = {
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_hip": 23,
            "right_hip": 24,
        }

    def detect(self, frame):
        """Return upper-body landmarks from an OpenCV camera frame."""

        # OpenCV uses BGR; MediaPipe expects RGB.
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb_frame)

        if results.pose_landmarks is None:
            return None

        landmarks = results.pose_landmarks.landmark

        return {
            name: {
                "x": landmarks[index].x,
                "y": landmarks[index].y,
                "z": landmarks[index].z,
                "visibility": landmarks[index].visibility,
            }
            for name, index in self.landmark_names.items()
        }

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()