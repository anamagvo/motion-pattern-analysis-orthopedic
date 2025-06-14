import cv2
import mediapipe as mp
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
from .angle_calculator import AngleCalculator

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

    def process_frame(self, frame: np.ndarray, angle_calculator: AngleCalculator) -> Tuple[np.ndarray, Optional[List[Tuple[float, float]]]]:
        """Process a single frame and detect landmarks."""
        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        # Convert back to BGR for OpenCV
        processed_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        landmarks = []
        if results.pose_landmarks:
            # Extract landmarks
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                landmarks.append((x, y))
            
            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                processed_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Calculate angles
            if len(landmarks) >= 25:  # MediaPipe Pose has 33 landmarks
                left_hip = landmarks[23]
                left_knee = landmarks[25]
                left_ankle = landmarks[27]
                right_hip = landmarks[24]
                right_knee = landmarks[26]
                right_ankle = landmarks[28]
                
                left_angle = angle_calculator.calculate_angle(left_hip, left_knee, left_ankle)
                right_angle = angle_calculator.calculate_angle(right_hip, right_knee, right_ankle)
                
                # Create overlay on the left side
                h, w = frame.shape[:2]
                overlay = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Add text to overlay
                cv2.putText(overlay, f"Left Knee 2D: {left_angle:.1f}°", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(overlay, f"Right Knee 2D: {right_angle:.1f}°", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add overlay to the left side of the frame
                processed_frame = cv2.addWeighted(processed_frame, 1, overlay, 0.7, 0)
        
        return processed_frame, landmarks

    def _draw_knee_lines(self, frame, hip, knee, ankle, color):
        height, width = frame.shape[:2]
        hip_point = (int(hip[0] * width), int(hip[1] * height))
        knee_point = (int(knee[0] * width), int(knee[1] * height))
        ankle_point = (int(ankle[0] * width), int(ankle[1] * height))
        cv2.line(frame, hip_point, knee_point, color, 2)
        cv2.line(frame, knee_point, ankle_point, color, 2) 