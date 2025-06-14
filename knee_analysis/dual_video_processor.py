import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional, List
import mediapipe as mp
from scipy.signal import find_peaks

from .angle_calculator import AngleCalculator
from .pose_3d_estimator import Pose3DEstimator
from .visualization_3d import SkeletonVisualizer3D

class DualVideoProcessor:
    def __init__(self, video_path1: str, video_path2: Optional[str] = None, framerate: float = 30.0):
        self.video_path1 = video_path1
        self.video_path2 = video_path2
        self.framerate = framerate
        self.cap1 = cv2.VideoCapture(video_path1)
        self.second_cap = None
        if video_path2 is not None and isinstance(video_path2, str) and video_path2.strip() != "":
            self.second_cap = cv2.VideoCapture(video_path2)
            if not self.second_cap.isOpened():
                print(f"Warning: Could not open second video: {video_path2}. Proceeding in single video mode.")
                self.second_cap = None
        self.pose_3d_estimator = None
        self.visualizer_3d = None
        self.landmarks1 = []
        self.landmarks2 = []
        self.landmarks_3d = []
        self.onset_frame1 = 0
        self.onset_frame2 = 0
        self.output_dir = Path("output_files")
        self.output_dir.mkdir(exist_ok=True)
        if self.second_cap:
            self.second_video_name = Path(video_path2).stem
        else:
            self.second_video_name = ""
        self.target_fps = framerate
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize 3D visualization
        self.visualizer_3d = SkeletonVisualizer3D()
        
        # Initialize 3D pose estimator with default camera parameters
        # These should be calibrated for your specific setup
        camera_matrix1 = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
        camera_matrix2 = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
        dist_coeffs1 = np.zeros(5, dtype=np.float32)
        dist_coeffs2 = np.zeros(5, dtype=np.float32)
        rvec = np.array([0, 0, 0], dtype=np.float32)
        tvec = np.array([0.5, 0, 0], dtype=np.float32)
        
        self.pose_3d = Pose3DEstimator(
            camera_matrix1, camera_matrix2,
            dist_coeffs1, dist_coeffs2,
            rvec, tvec
        )

    def initialize_second_video(self, video_path: str) -> bool:
        """Initialize the second video capture."""
        if self.second_cap is not None:
            self.second_cap.release()
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

    def process_dual_frame(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        angle_calculator: AngleCalculator
    ) -> Tuple[np.ndarray, float, float]:
        """Process frames from both videos and combine them."""
        # Process first frame
        processed_frame1, landmarks1 = self._process_single_frame(frame1, "Left Video")
        
        # Process second frame
        processed_frame2, landmarks2 = self._process_single_frame(frame2, "Right Video")
        
        # Combine frames horizontally
        combined_frame = np.hstack((processed_frame1, processed_frame2))
        
        # If we have landmarks from both cameras, estimate 3D pose
        if landmarks1 and landmarks2 and self.pose_3d_estimator:
            landmarks_3d = self.pose_3d_estimator.estimate_3d_pose(landmarks1, landmarks2)
            
            # Update 3D visualization
            self.visualizer_3d.draw_skeleton(landmarks_3d)
            self.visualizer_3d.handle_input()
            
            # Calculate angles in 3D space
            left_angle = angle_calculator.calculate_angle_3d(
                landmarks_3d[0], landmarks_3d[1], landmarks_3d[2])
            right_angle = angle_calculator.calculate_angle_3d(
                landmarks_3d[3], landmarks_3d[4], landmarks_3d[5])
        else:
            left_angle = 0.0
            right_angle = 0.0
        
        return combined_frame, left_angle, right_angle

    def _process_single_frame(
        self,
        frame: np.ndarray,
        label: str
    ) -> Tuple[np.ndarray, Optional[List[Tuple[float, float]]]]:
        """Process a single frame and return landmarks."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        processed_frame = frame.copy()
        landmarks = None
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                processed_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Extract landmarks
            landmarks = self._get_landmarks(results, self.mp_pose)
            
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
        return processed_frame, landmarks

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
        return [left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]

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
        
        # Collect landmarks for movement detection
        landmarks1_list = []
        landmarks2_list = []
        
        while cap1.isOpened() and self.second_cap.isOpened():
            ret1, frame1 = cap1.read()
            ret2, frame2 = self.second_cap.read()
            
            if not ret1 or not ret2:
                break
            
            # Process frames and collect landmarks
            _, landmarks1 = self._process_single_frame(frame1, "Left Video")
            _, landmarks2 = self._process_single_frame(frame2, "Right Video")
            
            if landmarks1:
                landmarks1_list.append(landmarks1)
            if landmarks2:
                landmarks2_list.append(landmarks2)
        
        # Reset video captures
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.second_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Detect movement onset
        onset1, onset2 = self.pose_3d.detect_movement_onset(landmarks1_list, landmarks2_list)
        print(f"\nSynchronization points:")
        print(f"First video: frame {onset1}")
        print(f"Second video: frame {onset2}")
        
        # Set video positions to movement onset
        cap1.set(cv2.CAP_PROP_POS_FRAMES, onset1)
        self.second_cap.set(cv2.CAP_PROP_POS_FRAMES, onset2)
        
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
        self.visualizer_3d.cleanup()
        cv2.destroyAllWindows()

    def get_video_properties(self, cap: cv2.VideoCapture) -> Tuple[int, int, int]:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        return width, height, fps

    def set_pose_3d_estimator(self, pose_3d_estimator: Pose3DEstimator):
        self.pose_3d_estimator = pose_3d_estimator

    def set_visualizer_3d(self, visualizer_3d: SkeletonVisualizer3D):
        self.visualizer_3d = visualizer_3d

    def synchronize_videos(self) -> Tuple[int, int]:
        """
        Synchronize videos based on movement onset.
        Returns the frame numbers where movement starts in each video.
        """
        if not self.pose_3d_estimator:
            raise ValueError("Pose3DEstimator not set")
        if not self.second_cap:
            raise ValueError("Second video not initialized")
            
        # Collect landmarks for movement detection
        landmarks1_list = []
        landmarks2_list = []
        
        # Reset video positions
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.second_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while self.cap1.isOpened() and self.second_cap.isOpened():
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.second_cap.read()
            
            if not ret1 or not ret2:
                break
            
            # Process frames and collect landmarks
            _, landmarks1 = self._process_single_frame(frame1, "Left Video")
            _, landmarks2 = self._process_single_frame(frame2, "Right Video")
            
            if landmarks1:
                landmarks1_list.append(landmarks1)
            if landmarks2:
                landmarks2_list.append(landmarks2)
        
        # Reset video positions
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.second_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Detect movement onset
        self.onset_frame1, self.onset_frame2 = self.pose_3d_estimator.detect_movement_onset(
            landmarks1_list, landmarks2_list)
        return self.onset_frame1, self.onset_frame2

    def process_frames(self):
        while True:
            ret1, frame1 = self.cap1.read()
            if not ret1:
                break
            # Process frame1 and update landmarks1
            # ... (existing processing code) ...
            if self.second_cap:
                ret2, frame2 = self.second_cap.read()
                if not ret2:
                    break
                # Process frame2 and update landmarks2
                # ... (existing processing code) ...
            # ...
            pass

    def release(self):
        """Release all resources."""
        if self.cap1:
            self.cap1.release()
        if self.second_cap:
            self.second_cap.release()
        if self.visualizer_3d:
            self.visualizer_3d.cleanup()
        cv2.destroyAllWindows() 