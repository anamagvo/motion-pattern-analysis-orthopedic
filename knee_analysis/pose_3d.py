import numpy as np
from typing import List, Tuple, Optional
import cv2

class Pose3DEstimator:
    def __init__(self, camera_matrix1: np.ndarray, camera_matrix2: np.ndarray,
                 dist_coeffs1: np.ndarray, dist_coeffs2: np.ndarray,
                 rvec: np.ndarray, tvec: np.ndarray):
        """
        Initialize the 3D pose estimator with camera parameters.
        
        Args:
            camera_matrix1: 3x3 camera matrix for first camera
            camera_matrix2: 3x3 camera matrix for second camera
            dist_coeffs1: Distortion coefficients for first camera
            dist_coeffs2: Distortion coefficients for second camera
            rvec: Rotation vector between cameras
            tvec: Translation vector between cameras
        """
        self.camera_matrix1 = camera_matrix1
        self.camera_matrix2 = camera_matrix2
        self.dist_coeffs1 = dist_coeffs1
        self.dist_coeffs2 = dist_coeffs2
        # Ensure rvec and tvec are float32 for OpenCV
        self.rvec = np.array(rvec, dtype=np.float32)
        self.tvec = np.array(tvec, dtype=np.float32)
        # Compute projection matrices (assuming rvec, tvec are from cam1 to cam2)
        self.P1 = np.hstack((self.camera_matrix1, np.zeros((3, 1), dtype=np.float32)))
        R, _ = cv2.Rodrigues(self.rvec)
        Rt = np.hstack((R, self.tvec.reshape(3, 1)))
        self.P2 = self.camera_matrix2 @ Rt
        
    def estimate_3d_pose(self, landmarks1: List[Tuple[float, float]], 
                        landmarks2: List[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
        """
        Estimate 3D pose from 2D landmarks from both cameras.
        
        Args:
            landmarks1: List of (x, y) coordinates from first camera
            landmarks2: List of (x, y) coordinates from second camera
            
        Returns:
            List of (x, y, z) coordinates in 3D space
        """
        if not landmarks1 or not landmarks2 or len(landmarks1) != len(landmarks2):
            return []
        points1 = np.array(landmarks1, dtype=np.float32).T  # shape (2, N)
        points2 = np.array(landmarks2, dtype=np.float32).T  # shape (2, N)
        if points1.shape[1] == 0 or points2.shape[1] == 0:
            return []
        # Triangulate points
        points_4d = cv2.triangulatePoints(self.P1, self.P2, points1, points2)
        
        # Convert to 3D points
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.T.tolist()
        
    def detect_movement_onset(self, landmarks1: List[List[Tuple[float, float]]],
                            landmarks2: List[List[Tuple[float, float]]]) -> Tuple[int, int]:
        """
        Detect the frame where movement starts in both videos.
        
        Args:
            landmarks1: List of frames, each containing list of (x, y) coordinates from first camera
            landmarks2: List of frames, each containing list of (x, y) coordinates from second camera
            
        Returns:
            Tuple of (frame1, frame2) where movement starts
        """
        def calculate_movement_score(landmarks):
            if not landmarks:
                return 0
            # Calculate velocity of key points
            velocities = []
            for i in range(1, len(landmarks)):
                prev = np.array(landmarks[i-1])
                curr = np.array(landmarks[i])
                velocity = np.linalg.norm(curr - prev)
                velocities.append(velocity)
            return np.mean(velocities) if velocities else 0
            
        # Calculate movement scores for both videos
        scores1 = [calculate_movement_score(frame) for frame in landmarks1]
        scores2 = [calculate_movement_score(frame) for frame in landmarks2]
        
        # Find first significant movement
        threshold = 0.1  # Adjust based on your needs
        onset1 = next((i for i, score in enumerate(scores1) if score > threshold), 0)
        onset2 = next((i for i, score in enumerate(scores2) if score > threshold), 0)
        
        return onset1, onset2 