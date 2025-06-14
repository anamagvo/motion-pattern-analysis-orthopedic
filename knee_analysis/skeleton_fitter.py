import numpy as np
from typing import List, Tuple, Dict, Optional
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class JointConstraint:
    min_angle: float
    max_angle: float
    preferred_angle: float
    weight: float = 1.0

class SkeletonFitter:
    def __init__(self, trc_file_path: str):
        """
        Initialize the skeleton fitter with a reference TRC file.
        
        Args:
            trc_file_path: Path to the TRC file containing neutral pose data
        """
        self.joint_constraints = self._load_joint_constraints()
        self.reference_skeleton = self._load_trc_file(trc_file_path)
        self.joint_hierarchy = {
            'hip': ['hip', 'knee', 'ankle'],
            'knee': ['knee', 'ankle'],
            'ankle': ['ankle']
        }
        
    def _load_joint_constraints(self) -> Dict[str, JointConstraint]:
        """Load joint angle constraints."""
        return {
            'hip': JointConstraint(min_angle=-30, max_angle=120, preferred_angle=0),
            'knee': JointConstraint(min_angle=0, max_angle=140, preferred_angle=0),
            'ankle': JointConstraint(min_angle=-20, max_angle=50, preferred_angle=0)
        }
        
    def _load_trc_file(self, trc_file_path: str) -> Dict[str, np.ndarray]:
        """
        Load and parse a TRC file.
        
        Args:
            trc_file_path: Path to the TRC file
            
        Returns:
            Dictionary mapping joint names to their 3D coordinates
        """
        # Read the TRC file
        df = pd.read_csv(trc_file_path, skiprows=3, delimiter='\t')
        
        # Extract joint positions
        skeleton = {}
        for joint in ['hip', 'knee', 'ankle']:
            x_col = f'{joint}_X'
            y_col = f'{joint}_Y'
            z_col = f'{joint}_Z'
            
            if all(col in df.columns for col in [x_col, y_col, z_col]):
                skeleton[joint] = np.column_stack((
                    df[x_col].values,
                    df[y_col].values,
                    df[z_col].values
                ))
                
        return skeleton
        
    def calculate_joint_angles(self, skeleton: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate joint angles from skeleton positions.
        
        Args:
            skeleton: Dictionary mapping joint names to their 3D coordinates
            
        Returns:
            Dictionary mapping joint names to their angles
        """
        angles = {}
        
        # Calculate hip angle
        if all(joint in skeleton for joint in ['hip', 'knee', 'ankle']):
            hip_vec = skeleton['knee'] - skeleton['hip']
            knee_vec = skeleton['ankle'] - skeleton['knee']
            angles['hip'] = self._calculate_angle(hip_vec, knee_vec)
            
        # Calculate knee angle
        if all(joint in skeleton for joint in ['knee', 'ankle']):
            knee_vec = skeleton['ankle'] - skeleton['knee']
            angles['knee'] = self._calculate_angle(knee_vec, np.array([0, 0, 1]))
            
        # Calculate ankle angle
        if all(joint in skeleton for joint in ['ankle']):
            ankle_vec = skeleton['ankle'] - skeleton['knee']
            angles['ankle'] = self._calculate_angle(ankle_vec, np.array([0, 0, 1]))
            
        return angles
        
    def _calculate_angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
        
    def optimize_skeleton(self, detected_landmarks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Optimize skeleton fit to detected landmarks while respecting constraints.
        
        Args:
            detected_landmarks: Dictionary mapping joint names to their detected 3D coordinates
            
        Returns:
            Dictionary mapping joint names to their optimized 3D coordinates
        """
        def objective_function(x):
            # Reshape x into joint positions
            positions = x.reshape(-1, 3)
            skeleton = {joint: pos for joint, pos in zip(detected_landmarks.keys(), positions)}
            
            # Calculate error terms
            detection_error = self._calculate_detection_error(skeleton, detected_landmarks)
            constraint_error = self._calculate_constraint_error(skeleton)
            smoothness_error = self._calculate_smoothness_error(skeleton)
            
            return detection_error + constraint_error + smoothness_error
            
        # Initial guess from detected landmarks
        x0 = np.concatenate([pos.flatten() for pos in detected_landmarks.values()])
        
        # Define bounds for optimization
        bounds = []
        for joint in detected_landmarks.keys():
            if joint in self.joint_constraints:
                constraint = self.joint_constraints[joint]
                bounds.extend([(None, None)] * 3)  # Position bounds
                
        # Optimize
        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        # Reshape result into joint positions
        optimized_positions = result.x.reshape(-1, 3)
        return {joint: pos for joint, pos in zip(detected_landmarks.keys(), optimized_positions)}
        
    def _calculate_detection_error(self, skeleton: Dict[str, np.ndarray],
                                 detected_landmarks: Dict[str, np.ndarray]) -> float:
        """Calculate error between skeleton and detected landmarks."""
        error = 0.0
        for joint in skeleton:
            if joint in detected_landmarks:
                error += np.sum((skeleton[joint] - detected_landmarks[joint]) ** 2)
        return error
        
    def _calculate_constraint_error(self, skeleton: Dict[str, np.ndarray]) -> float:
        """Calculate error from joint angle constraints."""
        error = 0.0
        angles = self.calculate_joint_angles(skeleton)
        
        for joint, angle in angles.items():
            if joint in self.joint_constraints:
                constraint = self.joint_constraints[joint]
                if angle < constraint.min_angle:
                    error += (constraint.min_angle - angle) ** 2 * constraint.weight
                elif angle > constraint.max_angle:
                    error += (angle - constraint.max_angle) ** 2 * constraint.weight
                error += (angle - constraint.preferred_angle) ** 2 * constraint.weight
                
        return error
        
    def _calculate_smoothness_error(self, skeleton: Dict[str, np.ndarray]) -> float:
        """Calculate smoothness error between consecutive segments."""
        error = 0.0
        children = list(self.joint_hierarchy.keys())
        
        # Only calculate smoothness for segments that have enough points
        for i in range(len(children) - 2):  # -2 because we need i+1 and i+2
            if i + 2 >= len(children):
                break
                
            vec1 = skeleton[children[i+1]] - skeleton[children[i]]
            vec2 = skeleton[children[i+2]] - skeleton[children[i+1]]
            
            # Normalize vectors
            vec1_norm = np.linalg.norm(vec1)
            vec2_norm = np.linalg.norm(vec2)
            
            if vec1_norm > 0 and vec2_norm > 0:
                vec1 = vec1 / vec1_norm
                vec2 = vec2 / vec2_norm
                
                # Calculate angle between vectors
                dot_product = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
                angle = np.arccos(dot_product)
                
                # Add to error (penalize large angles)
                error += angle * angle
        
        return error 