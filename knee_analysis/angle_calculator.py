import numpy as np
from typing import Tuple

class AngleCalculator:
    def calculate_angle(self, hip: Tuple[float, float], knee: Tuple[float, float], ankle: Tuple[float, float]) -> float:
        """
        Calculate the angle between three points in 2D space.
        
        Args:
            hip: (x, y) coordinates of the hip
            knee: (x, y) coordinates of the knee
            ankle: (x, y) coordinates of the ankle
            
        Returns:
            Angle in degrees
        """
        # Convert to numpy arrays for easier calculation
        hip = np.array(hip)
        knee = np.array(knee)
        ankle = np.array(ankle)
        
        # Calculate vectors
        v1 = hip - knee
        v2 = ankle - knee
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure value is in valid range
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle

    def calculate_angle_3d(self, hip: Tuple[float, float, float], knee: Tuple[float, float, float], ankle: Tuple[float, float, float]) -> float:
        """
        Calculate the angle between three points in 3D space.
        
        Args:
            hip: (x, y, z) coordinates of the hip
            knee: (x, y, z) coordinates of the knee
            ankle: (x, y, z) coordinates of the ankle
            
        Returns:
            Angle in degrees
        """
        # Convert to numpy arrays for easier calculation
        hip = np.array(hip)
        knee = np.array(knee)
        ankle = np.array(ankle)
        
        # Calculate vectors
        v1 = hip - knee
        v2 = ankle - knee
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure value is in valid range
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle 