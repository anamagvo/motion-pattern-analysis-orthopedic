import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt
from knee_analysis.angle_calculator import calculate_knee_angle
from knee_analysis.dual_camera_processor import DualCameraProcessor
from knee_analysis.skeleton_fitter import SkeletonFitter
from knee_analysis.plotter import plot_angles
from knee_analysis.trc_writer import write_trc_file
from knee_analysis.constraints import load_joint_constraints
from knee_analysis.skeleton import Skeleton
import sys
sys.path.append('/Users/avoegele/Dev/videopose3d/VideoPose3D')
from common.model import TemporalModel
from common.generators import UnchunkedGenerator

# ... existing code ... 