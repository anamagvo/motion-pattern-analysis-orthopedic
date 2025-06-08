import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional, List
import mediapipe as mp
from scipy.signal import find_peaks

from .angle_calculator import AngleCalculator

class DualVideoProcessor:
    def __init__(self, target_fps: float):
        self.second_cap: Optional[cv2.VideoCapture] = None
        self.second_video_name: str = ""
        self.output_dir = Path("output_files")
        self.output_dir.mkdir(exist_ok=True)
        self.target_fps = target_fps
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def initialize_second_video(self, video_path: str) -> bool:
        """Initialize the second video capture."""
        self.second_cap = cv2.VideoCapture(video_path)
        if not self.second_cap.isOpened():
            return False
        
        self.second_video_name = Path(video_path).stem
        return True

    def create_combined_video_writer(self, width: int, height: int) -> cv2.VideoWriter:
        """Create a video writer for the combined output."""
        output_path = self.output_dir / f"combined_analysis_{self.second_video_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(str(output_path), fourcc, self.target_fps, (width * 2, height))

    def detect_movement_onset(self, cap: cv2.VideoCapture) -> int:
        """Detect the frame where movement starts."""
        frame_count = 0
        movement_scores = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame to detect pose
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            
            if results.pose_landmarks:
                # Calculate movement score based on keypoint velocities
                landmarks = self._get_landmarks(results, self.mp_pose)
                if landmarks:
                    left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle = landmarks
                    # Calculate knee movement
                    knee_movement = np.sqrt(
                        (left_knee[0] - right_knee[0])**2 + 
                        (left_knee[1] - right_knee[1])**2
                    )
                    movement_scores.append(knee_movement)
            
            frame_count += 1
            
        # Reset video capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Find movement onset using peak detection
        if len(movement_scores) > 10:  # Need enough frames for reliable detection
            movement_scores = np.array(movement_scores)
            # Normalize scores
            movement_scores = (movement_scores - np.min(movement_scores)) / (np.max(movement_scores) - np.min(movement_scores))
            # Find peaks
            peaks, _ = find_peaks(movement_scores, height=0.5, distance=10)
            if len(peaks) > 0:
                return peaks[0]  # Return first significant movement
        
        return 0  # Return 0 if no clear movement detected

    def synchronize_videos(self, cap1: cv2.VideoCapture) -> Tuple[int, int]:
        """Synchronize videos based on movement onset."""
        print("Detecting movement onset in first video...")
        onset1 = self.detect_movement_onset(cap1)
        print("Detecting movement onset in second video...")
        onset2 = self.detect_movement_onset(self.second_cap)
        
        # Reset video captures
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.second_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return onset1, onset2

    def process_dual_frame(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        angle_calculator: AngleCalculator
    ) -> Tuple[np.ndarray, float, float]:
        """Process frames from both videos and combine them."""
        # Process first frame
        processed_frame1, left_angle1, right_angle1 = self._process_single_frame(
            frame1, angle_calculator, "Left Video")
        
        # Process second frame
        processed_frame2, left_angle2, right_angle2 = self._process_single_frame(
            frame2, angle_calculator, "Right Video")
        
        # Combine frames horizontally
        combined_frame = np.hstack((processed_frame1, processed_frame2))
        
        # Calculate average angles
        left_angle = (left_angle1 + left_angle2) / 2
        right_angle = (right_angle1 + right_angle2) / 2
        
        return combined_frame, left_angle, right_angle

    def _process_single_frame(
        self,
        frame: np.ndarray,
        angle_calculator: AngleCalculator,
        label: str
    ) -> Tuple[np.ndarray, float, float]:
        """Process a single frame and add labels."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        left_angle = 0.0
        right_angle = 0.0
        processed_frame = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                processed_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            # Extract landmarks
            landmarks = self._get_landmarks(results, self.mp_pose)
            if landmarks:
                left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle = landmarks
                left_angle = angle_calculator.calculate_angle(left_hip, left_knee, left_ankle)
                right_angle = angle_calculator.calculate_angle(right_hip, right_knee, right_ankle)
                cv2.putText(processed_frame, f"Left Knee: {left_angle:.1f}°", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Right Knee: {right_angle:.1f}°", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self._draw_knee_lines(processed_frame, left_hip, left_knee, left_ankle, (0, 255, 0))
                self._draw_knee_lines(processed_frame, right_hip, right_knee, right_ankle, (0, 255, 0))
        # Add video label
        cv2.putText(
            processed_frame,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        return processed_frame, left_angle, right_angle

    def _get_landmarks(self, results, mp_pose):
        if not results.pose_landmarks:
            return None
        left_hip = (
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
        )
        left_knee = (
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
        )
        left_ankle = (
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
        )
        right_hip = (
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y
        )
        right_knee = (
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y
        )
        right_ankle = (
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y
        )
        return left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle

    def _draw_knee_lines(self, frame, hip, knee, ankle, color):
        height, width = frame.shape[:2]
        hip_point = (int(hip[0] * width), int(hip[1] * height))
        knee_point = (int(knee[0] * width), int(knee[1] * height))
        ankle_point = (int(ankle[0] * width), int(ankle[1] * height))
        cv2.line(frame, hip_point, knee_point, color, 2)
        cv2.line(frame, knee_point, ankle_point, color, 2)

    def analyze_dual_video(self, video_path1: str, video_path2: str, angle_calculator):
        """Process two videos simultaneously and analyze knee angles."""
        # Initialize first video
        cap1 = cv2.VideoCapture(video_path1)
        if not cap1.isOpened():
            print(f"Error: Could not open first video: {video_path1}")
            return
        
        # Initialize second video
        if not self.initialize_second_video(video_path2):
            print(f"Error: Could not open second video: {video_path2}")
            cap1.release()
            return
        
        # Get video properties and create video writer
        width, height, fps = self.get_video_properties(cap1)
        out = self.create_combined_video_writer(width, height)
        
        while cap1.isOpened() and self.second_cap.isOpened():
            ret1, frame1 = cap1.read()
            ret2, frame2 = self.second_cap.read()
            
            if not ret1 or not ret2:
                break
            
            # Process frames
            processed_frame, left_angle, right_angle = self.process_dual_frame(
                frame1, frame2, angle_calculator)
            
            # Write the processed frame
            out.write(processed_frame)
            
            # Display the frame
            cv2.imshow('Dual Video Knee Angle Analysis', processed_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap1.release()
        self.second_cap.release()
        out.release()
        cv2.destroyAllWindows() 