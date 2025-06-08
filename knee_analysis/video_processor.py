import cv2
import mediapipe as mp
from pathlib import Path
from typing import Tuple, Optional

class VideoProcessor:
    def __init__(self, target_fps: float):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.video_name = ""
        self.output_dir = Path("output_files")
        self.output_dir.mkdir(exist_ok=True)
        self.target_fps = target_fps

    def get_video_properties(self, cap: cv2.VideoCapture) -> Tuple[int, int, int]:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        return width, height, fps

    def create_video_writer(self, video_path: str, width: int, height: int) -> cv2.VideoWriter:
        self.video_name = Path(video_path).stem
        output_video_path = self.output_dir / f"{self.video_name}_with_skeleton.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(str(output_video_path), fourcc, self.target_fps, (width, height))

    def get_landmarks(self, results) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        if not results.pose_landmarks:
            return None
        left_hip = (
            results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
            results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y
        )
        left_knee = (
            results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
            results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y
        )
        left_ankle = (
            results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
            results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y
        )
        right_hip = (
            results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
            results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y
        )
        right_knee = (
            results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].x,
            results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].y
        )
        right_ankle = (
            results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x,
            results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y
        )
        return left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle

    def process_frame(self, frame, angle_calculator) -> Tuple[cv2.Mat, float, float]:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        left_angle = 0.0
        right_angle = 0.0
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            landmarks = self.get_landmarks(results)
            if landmarks:
                left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle = landmarks
                left_angle = angle_calculator.calculate_angle(left_hip, left_knee, left_ankle)
                right_angle = angle_calculator.calculate_angle(right_hip, right_knee, right_ankle)
                cv2.putText(frame, f"Left Knee: {left_angle:.1f}°", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Right Knee: {right_angle:.1f}°", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self._draw_knee_lines(frame, left_hip, left_knee, left_ankle, (0, 255, 0))
                self._draw_knee_lines(frame, right_hip, right_knee, right_ankle, (0, 255, 0))
        return frame, left_angle, right_angle

    def _draw_knee_lines(self, frame, hip, knee, ankle, color):
        height, width = frame.shape[:2]
        hip_point = (int(hip[0] * width), int(hip[1] * height))
        knee_point = (int(knee[0] * width), int(knee[1] * height))
        ankle_point = (int(ankle[0] * width), int(ankle[1] * height))
        cv2.line(frame, hip_point, knee_point, color, 2)
        cv2.line(frame, knee_point, ankle_point, color, 2) 