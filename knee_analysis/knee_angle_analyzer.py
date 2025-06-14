import cv2
import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

from .video_processor import VideoProcessor
from .dual_video_processor import DualVideoProcessor
from .angle_calculator import AngleCalculator
from .plotter import Plotter
from .pose_3d_estimator import Pose3DEstimator
from .standard_skeleton import Skeleton  

class KneeAngleAnalyzer:
    def __init__(self, target_fps: float):
        self.video_processor = VideoProcessor(target_fps)
        self.dual_video_processor = None
        self.angle_calculator = AngleCalculator()
        self.plotter = Plotter()
        self.target_fps = target_fps
        
        # Initialize lists to store angle data
        self.left_knee_angles: List[float] = []
        self.right_knee_angles: List[float] = []
        self.left_knee_angles_3d: List[float] = []
        self.right_knee_angles_3d: List[float] = []
        self.left_knee_angles_2d_proj: List[float] = []
        self.right_knee_angles_2d_proj: List[float] = []
        
        # Store last valid 3D skeleton
        self.last_valid_skeleton = None

    def analyze_video(self, video_path: str, second_video_path: Optional[str] = None, trc_file_path: Optional[str] = None):
        """Process the video(s) and analyze knee angles."""
        if second_video_path:
            self.dual_video_processor = DualVideoProcessor(video_path, second_video_path, self.target_fps)
            # Initialize 3D pose estimator with default camera parameters
            camera_matrix1 = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
            camera_matrix2 = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
            dist_coeffs1 = np.zeros(5, dtype=np.float32)
            dist_coeffs2 = np.zeros(5, dtype=np.float32)
            rvec = np.array([0, 0, 0], dtype=np.float32)
            tvec = np.array([0.5, 0, 0], dtype=np.float32)
            
            pose_3d_estimator = Pose3DEstimator(
                camera_matrix1, camera_matrix2,
                dist_coeffs1, dist_coeffs2,
                rvec, tvec
            )
            self.dual_video_processor.set_pose_3d_estimator(pose_3d_estimator)
            
            # Initialize skeleton fitter if TRC file is provided
            if trc_file_path:
                self.dual_video_processor.set_skeleton_fitter(trc_file_path)
            
            self._analyze_dual_video(video_path, second_video_path)
        else:
            self._analyze_single_video(video_path)

    def _analyze_single_video(self, video_path: str):
        """Process a single video and analyze knee angles."""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties and create video writer
        width, height, _ = self.video_processor.get_video_properties(cap)
        out = self.video_processor.create_video_writer(video_path, width, height)
        
        # Initialize 3D pose estimator with default camera parameters
        camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros(5, dtype=np.float32)
        pose_3d_estimator = Pose3DEstimator(camera_matrix, camera_matrix, dist_coeffs, dist_coeffs, np.zeros(3), np.zeros(3))
        
        # Initialize skeleton fitter if TRC file is provided
        skeleton_fitter = None
        if hasattr(self, 'trc_file_path') and self.trc_file_path:
            skeleton_fitter = SkeletonFitter(self.trc_file_path)
        
        # Create figure for angle plots
        plt.figure(figsize=(20, 5))
        
        # 2D angle plot
        ax_2d = plt.subplot(141)
        ax_2d.set_title('2D Knee Angles')
        ax_2d.set_xlabel('Frame')
        ax_2d.set_ylabel('Angle (degrees)')
        ax_2d.grid(True)
        
        # 3D views
        ax_front = plt.subplot(142, projection='3d')
        ax_side = plt.subplot(143, projection='3d')
        ax_top = plt.subplot(144, projection='3d')
        
        # Set initial views
        ax_front.view_init(elev=0, azim=0)  # Front view
        ax_side.view_init(elev=0, azim=90)  # Side view
        ax_top.view_init(elev=90, azim=0)   # Top view
        
        # Set titles
        ax_front.set_title('Front View')
        ax_side.set_title('Side View')
        ax_top.set_title('Top View')
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame, landmarks = self.video_processor.process_frame(frame, self.angle_calculator)
            
            # Calculate 2D angles from landmarks
            if landmarks:
                left_angle = self.angle_calculator.calculate_angle(
                    landmarks['left_hip'], landmarks['left_knee'], landmarks['left_ankle'])
                right_angle = self.angle_calculator.calculate_angle(
                    landmarks['right_hip'], landmarks['right_knee'], landmarks['right_ankle'])
                self.left_knee_angles.append(left_angle)
                self.right_knee_angles.append(right_angle)
            
            # If landmarks are detected and skeleton fitter is available, do 3D fitting
            if landmarks and skeleton_fitter:
                # Estimate 3D pose using default camera model
                landmarks_3d = pose_3d_estimator.estimate_3d_pose(landmarks, landmarks)
                
                # Fit skeleton using inverse kinematics
                fitted_landmarks = skeleton_fitter.optimize_skeleton(landmarks_3d)
                self.last_valid_skeleton = fitted_landmarks  # Store the last valid skeleton
                
                # Calculate angles from fitted 3D skeleton
                left_angle_3d = self.angle_calculator.calculate_angle_3d(
                    fitted_landmarks['hip'], fitted_landmarks['knee'], fitted_landmarks['ankle'])
                right_angle_3d = self.angle_calculator.calculate_angle_3d(
                    fitted_landmarks['hip'], fitted_landmarks['knee'], fitted_landmarks['ankle'])
                
                self.left_knee_angles_3d.append(left_angle_3d)
                self.right_knee_angles_3d.append(right_angle_3d)
            
            # Update plots
            frame_count += 1
            frames = range(frame_count)
            
            # Update 2D angle plot
            ax_2d.clear()
            if self.left_knee_angles:
                ax_2d.plot(frames, self.left_knee_angles, 'g-', label='Left Knee 2D')
            if self.right_knee_angles:
                ax_2d.plot(frames, self.right_knee_angles, 'b-', label='Right Knee 2D')
            if self.left_knee_angles_3d:
                ax_2d.plot(frames, self.left_knee_angles_3d, 'g--', label='Left Knee 3D')
            if self.right_knee_angles_3d:
                ax_2d.plot(frames, self.right_knee_angles_3d, 'b--', label='Right Knee 3D')
            ax_2d.set_title('Knee Angles')
            ax_2d.set_xlabel('Frame')
            ax_2d.set_ylabel('Angle (degrees)')
            ax_2d.grid(True)
            ax_2d.legend()
            
            # Update 3D views if we have a valid skeleton
            if self.last_valid_skeleton:
                for ax in [ax_front, ax_side, ax_top]:
                    ax.clear()
                    # Plot joints
                    for landmark in self.last_valid_skeleton.values():
                        ax.scatter(landmark[0], landmark[1], landmark[2], c='r', marker='o', s=100)
                    
                    # Plot connections
                    ax.plot([self.last_valid_skeleton['hip'][0], self.last_valid_skeleton['knee'][0]],
                           [self.last_valid_skeleton['hip'][1], self.last_valid_skeleton['knee'][1]],
                           [self.last_valid_skeleton['hip'][2], self.last_valid_skeleton['knee'][2]], 'b-', linewidth=2)
                    ax.plot([self.last_valid_skeleton['knee'][0], self.last_valid_skeleton['ankle'][0]],
                           [self.last_valid_skeleton['knee'][1], self.last_valid_skeleton['ankle'][1]],
                           [self.last_valid_skeleton['knee'][2], self.last_valid_skeleton['ankle'][2]], 'b-', linewidth=2)
                    
                    # Set labels
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_box_aspect([1,1,1])
            
            plt.draw()
            plt.pause(0.001)
            
            # Write frame to output video
            out.write(processed_frame)
            
            # Display frame
            cv2.imshow('Knee Angle Analysis', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Save the final plot
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        plt.savefig(f'output_files/{video_name}_knee_angle_analysis.png', dpi=300, bbox_inches='tight')
        plt.close('all')
        
        # Plot results and get statistics, only if there is data
        if self.left_knee_angles_3d or self.left_knee_angles_2d_proj:
            self.plotter.plot_knee_angles_compare(
                self.video_processor.video_name,
                self.left_knee_angles_3d,
                self.left_knee_angles_2d_proj,
                self.target_fps
            )
        else:
            print("No knee angle data was collected. Please check your video and ensure the person is fully visible.")

    def _analyze_dual_video(self, video_path1: str, video_path2: str):
        """Process two videos simultaneously and analyze knee angles."""
        # Initialize first video
        cap1 = cv2.VideoCapture(video_path1)
        if not cap1.isOpened():
            print(f"Error: Could not open first video: {video_path1}")
            return
        
        # Initialize second video
        if not self.dual_video_processor.initialize_second_video(video_path2):
            print(f"Error: Could not open second video: {video_path2}")
            cap1.release()
            return
        
        # Synchronize videos based on movement onset
        onset_frame1, onset_frame2 = self.dual_video_processor.synchronize_videos()
        print(f"\nSynchronization points:")
        print(f"First video: frame {onset_frame1}")
        print(f"Second video: frame {onset_frame2}")
        
        # Set video positions to movement onset
        cap1.set(cv2.CAP_PROP_POS_FRAMES, onset_frame1)
        self.dual_video_processor.second_cap.set(cv2.CAP_PROP_POS_FRAMES, onset_frame2)
        
        # Get video properties and create video writer
        width, height, _ = self.video_processor.get_video_properties(cap1)
        out = self.dual_video_processor.create_combined_video_writer(width, height)
        
        while cap1.isOpened() and self.dual_video_processor.second_cap.isOpened():
            ret1, frame1 = cap1.read()
            ret2, frame2 = self.dual_video_processor.second_cap.read()
            
            if not ret1 or not ret2:
                break
            
            # Process frames
            processed_frame, left_angle, right_angle = self.dual_video_processor.process_dual_frame(
                frame1, frame2, self.angle_calculator)
            
            # Store angles
            self.left_knee_angles.append(left_angle)
            self.right_knee_angles.append(right_angle)
            
            # Write the processed frame
            out.write(processed_frame)
            
            # Display the frame
            cv2.imshow('Dual Video Knee Angle Analysis', processed_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap1.release()
        self.dual_video_processor.second_cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Plot results and get statistics
        self.plotter.plot_knee_angles(
            f"{self.video_processor.video_name}_{self.dual_video_processor.second_video_name}",
            self.left_knee_angles,
            self.right_knee_angles,
            self.target_fps
        )

def main():
    # Get user input for framerate
    target_fps = float(input("Enter desired framerate (e.g., 30.0): "))
    
    # Initialize the analyzer with the target framerate
    analyzer = KneeAngleAnalyzer(target_fps=target_fps)
    
    # Get user input for video folder
    print("\nAvailable folders:")
    folders = [f for f in os.listdir("input_files") if os.path.isdir(os.path.join("input_files", f))]
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder}")
    
    folder_idx = int(input("\nEnter the number of the folder containing your videos: ")) - 1
    folder_path = os.path.join("input_files", folders[folder_idx])
    
    # Get user input for video files
    videos = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'))]
    print(f"\nVideos found in {folders[folder_idx]}:")
    for i, video in enumerate(videos, 1):
        print(f"{i}. {video}")
    
    video_idx1 = int(input("\nEnter the number of the first video to analyze: ")) - 1
    video_path1 = os.path.join(folder_path, videos[video_idx1])
    
    # Ask if user wants to analyze a second video
    use_second_video = input("\nDo you want to analyze a second video? (yes/no): ").lower() == 'yes'
    video_path2 = None
    if use_second_video:
        video_idx2 = int(input("Enter the number of the second video to analyze: ")) - 1
        video_path2 = os.path.join(folder_path, videos[video_idx2])
    
    # Get user input for TRC file
    trc_file_path = input("\nEnter the path to the TRC file (or press Enter to use default): ").strip()
    if not trc_file_path:
        # Use default TRC file
        trc_file_path = "input_files/default_skeleton.trc"
        print(f"Using default TRC file: {trc_file_path}")
    
    # Analyze the video(s)
    if video_path2:
        analyzer.analyze_video(video_path1, video_path2, trc_file_path)
    else:
        analyzer.analyze_video(video_path1, None, trc_file_path)

if __name__ == "__main__":
    main() 