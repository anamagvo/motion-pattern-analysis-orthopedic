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
        self.rvec = rvec
        self.tvec = tvec
        
        # Convert rotation vector to rotation matrix
        self.rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Create projection matrices
        # First camera is at origin
        self.projection_matrix1 = np.hstack((self.camera_matrix1, np.zeros((3, 1))))
        
        # Second camera is rotated and translated
        rt_matrix = np.hstack((self.rotation_matrix, self.tvec.reshape(3, 1)))
        self.projection_matrix2 = self.camera_matrix2 @ rt_matrix
        
        # Anatomical constraints
        self.joint_constraints = {
            'hip_width': 0.3,  # meters
            'knee_width': 0.1,  # meters
            'ankle_width': 0.1,  # meters
            'thigh_length': 0.4,  # meters
            'shank_length': 0.4,  # meters
            'max_knee_angle': 160,  # degrees
            'min_knee_angle': 0,  # degrees
        }
        
    def estimate_3d_pose_anatomical(self, landmarks1: dict, landmarks2: dict) -> dict:
        """
        Estimate 3D pose using anatomical constraints and two synchronized 2D views.
        
        Args:
            landmarks1: Dictionary of (x, y) coordinates from first camera
            landmarks2: Dictionary of (x, y) coordinates from second camera
            
        Returns:
            Dictionary of (x, y, z) coordinates in 3D space
        """
        if not landmarks1 or not landmarks2:
            return None
            
        # Convert landmarks to normalized coordinates (0-1)
        def normalize_landmarks(landmarks):
            return {k: (v[0], v[1]) for k, v in landmarks.items()}
        
        norm1 = normalize_landmarks(landmarks1)
        norm2 = normalize_landmarks(landmarks2)
        
        # Estimate depth using anatomical constraints
        def estimate_depth(p1, p2, width):
            # Use the width constraint to estimate depth
            # If points are closer in one view, they must be further in the other
            dx1 = abs(p1[0] - p2[0])
            dx2 = abs(p1[1] - p2[1])
            # Use the larger difference to estimate depth
            depth = width / max(dx1, dx2)
            return depth
        
        # Create 3D points using anatomical constraints
        points_3d = {}
        
        # Estimate hip width and position
        hip_width = self.joint_constraints['hip_width']
        left_hip_depth = estimate_depth(norm1['left_hip'], norm2['left_hip'], hip_width)
        right_hip_depth = estimate_depth(norm1['right_hip'], norm2['right_hip'], hip_width)
        
        # Set hip position as origin
        points_3d['hip'] = (0, 0, 0)
        points_3d['left_hip'] = (-hip_width/2, 0, 0)
        points_3d['right_hip'] = (hip_width/2, 0, 0)
        
        # Estimate knee positions
        knee_width = self.joint_constraints['knee_width']
        left_knee_depth = estimate_depth(norm1['left_knee'], norm2['left_knee'], knee_width)
        right_knee_depth = estimate_depth(norm1['right_knee'], norm2['right_knee'], knee_width)
        
        # Calculate knee positions using thigh length and angles
        left_thigh_angle = self._calculate_angle(norm1['left_hip'], norm1['left_knee'])
        right_thigh_angle = self._calculate_angle(norm1['right_hip'], norm1['right_knee'])
        
        points_3d['left_knee'] = self._calculate_joint_position(
            points_3d['left_hip'],
            self.joint_constraints['thigh_length'],
            left_thigh_angle,
            left_knee_depth
        )
        points_3d['right_knee'] = self._calculate_joint_position(
            points_3d['right_hip'],
            self.joint_constraints['thigh_length'],
            right_thigh_angle,
            right_knee_depth
        )
        
        # Estimate ankle positions
        ankle_width = self.joint_constraints['ankle_width']
        left_ankle_depth = estimate_depth(norm1['left_ankle'], norm2['left_ankle'], ankle_width)
        right_ankle_depth = estimate_depth(norm1['right_ankle'], norm2['right_ankle'], ankle_width)
        
        # Calculate ankle positions using shank length and angles
        left_shank_angle = self._calculate_angle(norm1['left_knee'], norm1['left_ankle'])
        right_shank_angle = self._calculate_angle(norm1['right_knee'], norm1['right_ankle'])
        
        points_3d['left_ankle'] = self._calculate_joint_position(
            points_3d['left_knee'],
            self.joint_constraints['shank_length'],
            left_shank_angle,
            left_ankle_depth
        )
        points_3d['right_ankle'] = self._calculate_joint_position(
            points_3d['right_knee'],
            self.joint_constraints['shank_length'],
            right_shank_angle,
            right_ankle_depth
        )
        
        return points_3d
    
    def _calculate_angle(self, p1, p2):
        """Calculate angle between two points in 2D."""
        import math
        return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    
    def _calculate_joint_position(self, parent_joint, length, angle, depth):
        """Calculate 3D position of a joint based on parent joint, length, and angle."""
        import math
        angle_rad = math.radians(angle)
        x = parent_joint[0] + length * math.cos(angle_rad)
        y = parent_joint[1] + length * math.sin(angle_rad)
        z = depth
        return (x, y, z)
    
    def estimate_3d_pose(self, landmarks1: List[Tuple[float, float]], 
                        landmarks2: List[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
        """
        Estimate 3D pose from 2D landmarks from two cameras.
        
        Args:
            landmarks1: List of (x, y) coordinates from first camera
            landmarks2: List of (x, y) coordinates from second camera
            
        Returns:
            List of (x, y, z) coordinates in 3D space
        """
        if not landmarks1 or not landmarks2:
            return []
            
        # Convert landmarks to numpy arrays
        points1 = np.array(landmarks1, dtype=np.float32)
        points2 = np.array(landmarks2, dtype=np.float32)
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(
            self.projection_matrix1, self.projection_matrix2,
            points1.T, points2.T
        )
        
        # Convert to 3D points
        points_3d = points_4d[:3] / points_4d[3]
        return list(map(tuple, points_3d.T))
        
    def detect_movement_onset(self, landmarks1: List[dict], landmarks2: List[dict]) -> Tuple[int, int]:
        """
        Detect the frame where movement starts in both videos.
        Args:
            landmarks1: List of frames, each containing dictionary of landmarks from first camera
            landmarks2: List of frames, each containing dictionary of landmarks from second camera
        Returns:
            Tuple of (frame1, frame2) where movement starts
        """
        def calculate_movement_score(landmarks):
            if not landmarks:
                return []
            velocities = []
            for i in range(1, len(landmarks)):
                prev = landmarks[i-1]
                curr = landmarks[i]
                joint_velocities = []
                for joint in ['left_knee', 'right_knee']:
                    if joint in prev and joint in curr:
                        prev_pos = np.array(prev[joint])
                        curr_pos = np.array(curr[joint])
                        velocity = np.linalg.norm(curr_pos - prev_pos)
                        joint_velocities.append(velocity)
                if joint_velocities:
                    velocities.append(np.mean(joint_velocities))
            return velocities

        # Calculate movement scores for both videos
        scores1 = calculate_movement_score(landmarks1)
        scores2 = calculate_movement_score(landmarks2)

        # Find first significant movement
        threshold = 0.1  # Adjust based on your needs
        onset1 = next((i for i, score in enumerate(scores1) if score > threshold), 0)
        onset2 = next((i for i, score in enumerate(scores2) if score > threshold), 0)

        return onset1, onset2 