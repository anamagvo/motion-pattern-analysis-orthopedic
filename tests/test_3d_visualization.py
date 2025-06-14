import pytest
import numpy as np
from knee_analysis.visualization_3d import SkeletonVisualizer3D
from knee_analysis.pose_3d import Pose3DEstimator
from knee_analysis.angle_calculator import AngleCalculator

@pytest.fixture
def sample_landmarks_3d():
    """Sample 3D landmarks for testing."""
    return [
        (0.0, 0.0, 0.0),    # left hip
        (0.0, -0.5, 0.0),   # left knee
        (0.0, -1.0, 0.0),   # left ankle
        (0.5, 0.0, 0.0),    # right hip
        (0.5, -0.5, 0.0),   # right knee
        (0.5, -1.0, 0.0),   # right ankle
    ]

@pytest.fixture
def sample_landmarks_2d():
    """Sample 2D landmarks from two cameras for testing."""
    # Camera 1 landmarks (front view)
    landmarks1 = [
        (0.5, 0.5),    # left hip
        (0.5, 0.7),    # left knee
        (0.5, 0.9),    # left ankle
        (0.7, 0.5),    # right hip
        (0.7, 0.7),    # right knee
        (0.7, 0.9),    # right ankle
    ]
    
    # Camera 2 landmarks (side view)
    landmarks2 = [
        (0.5, 0.5),    # left hip
        (0.5, 0.7),    # left knee
        (0.5, 0.9),    # left ankle
        (0.7, 0.5),    # right hip
        (0.7, 0.7),    # right knee
        (0.7, 0.9),    # right ankle
    ]
    
    return landmarks1, landmarks2

@pytest.fixture
def pose_3d_estimator():
    """Create a Pose3DEstimator instance with test camera parameters."""
    camera_matrix1 = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
    camera_matrix2 = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
    dist_coeffs1 = np.zeros(5)
    dist_coeffs2 = np.zeros(5)
    rvec = np.array([0, 0, 0])
    tvec = np.array([0.5, 0, 0])
    
    return Pose3DEstimator(
        camera_matrix1, camera_matrix2,
        dist_coeffs1, dist_coeffs2,
        rvec, tvec
    )

def test_3d_visualization_initialization():
    """Test initialization of 3D visualizer."""
    visualizer = SkeletonVisualizer3D()
    assert visualizer.display == (800, 600)
    assert visualizer.joint_color == (1, 0, 0)
    assert visualizer.bone_color == (0, 1, 0)

def test_3d_visualization_draw_skeleton(sample_landmarks_3d):
    """Test skeleton drawing functionality."""
    visualizer = SkeletonVisualizer3D()
    # Note: This is a basic test that just checks if the function runs without errors
    # Actual rendering tests would require more complex setup
    visualizer.draw_skeleton(sample_landmarks_3d)
    assert True  # If we get here, no exceptions were raised

def test_pose_3d_estimation(pose_3d_estimator, sample_landmarks_2d):
    """Test 3D pose estimation from 2D landmarks."""
    landmarks1, landmarks2 = sample_landmarks_2d
    landmarks_3d = pose_3d_estimator.estimate_3d_pose(landmarks1, landmarks2)
    
    assert len(landmarks_3d) == 6  # Should have 6 landmarks
    for landmark in landmarks_3d:
        assert len(landmark) == 3  # Each landmark should have x, y, z coordinates

def test_movement_onset_detection(pose_3d_estimator):
    """Test movement onset detection."""
    # Create sample landmark sequences
    landmarks1 = [
        [(0.5, 0.5)] * 10,  # No movement
        [(0.5 + i*0.01, 0.5) for i in range(10)]  # Movement
    ]
    landmarks2 = [
        [(0.5, 0.5)] * 10,  # No movement
        [(0.5 + i*0.01, 0.5) for i in range(10)]  # Movement
    ]
    
    onset1, onset2 = pose_3d_estimator.detect_movement_onset(landmarks1, landmarks2)
    assert onset1 >= 0
    assert onset2 >= 0

def test_3d_angle_calculation(sample_landmarks_3d):
    """Test 3D angle calculation."""
    calculator = AngleCalculator()
    
    # Test left knee angle
    left_angle = calculator.calculate_angle_3d(
        sample_landmarks_3d[0],  # left hip
        sample_landmarks_3d[1],  # left knee
        sample_landmarks_3d[2]   # left ankle
    )
    
    # Test right knee angle
    right_angle = calculator.calculate_angle_3d(
        sample_landmarks_3d[3],  # right hip
        sample_landmarks_3d[4],  # right knee
        sample_landmarks_3d[5]   # right ankle
    )
    
    assert 0 <= left_angle <= 180
    assert 0 <= right_angle <= 180

def test_camera_controls():
    """Test camera control functionality."""
    visualizer = SkeletonVisualizer3D()
    
    # Test initial rotation values
    assert visualizer.rot_x == 0
    assert visualizer.rot_y == 0
    
    # Simulate key presses
    # Note: This is a basic test that just checks if the function runs without errors
    # Actual key press simulation would require more complex setup
    visualizer.handle_input()
    assert True  # If we get here, no exceptions were raised

def test_error_handling(pose_3d_estimator):
    """Test error handling for missing or invalid landmarks."""
    # Test with empty landmarks
    landmarks_3d = pose_3d_estimator.estimate_3d_pose([], [])
    assert len(landmarks_3d) == 0
    
    # Test with invalid landmarks (should raise exception)
    with pytest.raises(Exception):
        pose_3d_estimator.estimate_3d_pose([(0, 0)], [(0, 0, 0)])

if __name__ == '__main__':
    pytest.main(['-v', 'test_3d_visualization.py']) 