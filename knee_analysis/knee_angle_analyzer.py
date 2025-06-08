import cv2
import os
from pathlib import Path
from typing import List, Optional

from .video_processor import VideoProcessor
from .dual_video_processor import DualVideoProcessor
from .angle_calculator import AngleCalculator
from .plotter import Plotter

class KneeAngleAnalyzer:
    def __init__(self, target_fps: float):
        self.video_processor = VideoProcessor(target_fps)
        self.dual_video_processor = DualVideoProcessor(target_fps)
        self.angle_calculator = AngleCalculator()
        self.plotter = Plotter()
        self.target_fps = target_fps
        
        # Initialize lists to store angle data
        self.left_knee_angles: List[float] = []
        self.right_knee_angles: List[float] = []

    def analyze_video(self, video_path: str, second_video_path: Optional[str] = None):
        """Process the video(s) and analyze knee angles."""
        if second_video_path:
            self._analyze_dual_video(video_path, second_video_path)
        else:
            self._analyze_single_video(video_path)

    def _analyze_single_video(self, video_path: str):
        """Process a single video and analyze knee angles."""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties and create video writer
        width, height, _ = self.video_processor.get_video_properties(cap)
        out = self.video_processor.create_video_writer(video_path, width, height)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame, left_angle, right_angle = self.video_processor.process_frame(
                frame, self.angle_calculator)
            
            # Store angles
            self.left_knee_angles.append(left_angle)
            self.right_knee_angles.append(right_angle)
            
            # Write the processed frame to output video
            out.write(processed_frame)
            
            # Display the frame
            cv2.imshow('Knee Angle Analysis', processed_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Plot results and get statistics
        self.plotter.plot_knee_angles(
            self.video_processor.video_name,
            self.left_knee_angles,
            self.right_knee_angles,
            self.target_fps
        )

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
        onset1, onset2 = self.dual_video_processor.synchronize_videos(cap1)
        print(f"\nSynchronization points:")
        print(f"First video: frame {onset1}")
        print(f"Second video: frame {onset2}")
        
        # Set video positions to movement onset
        cap1.set(cv2.CAP_PROP_POS_FRAMES, onset1)
        self.dual_video_processor.second_cap.set(cv2.CAP_PROP_POS_FRAMES, onset2)
        
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
    # Print supported video formats
    print("Supported video formats: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm")
    print("Place your video files in a folder within the 'input_files' directory.")
    
    # Get desired framerate
    try:
        fps = float(input("\nEnter desired framerate (e.g., 30.0): "))
        if fps <= 0:
            print("Framerate must be positive.")
            return
    except ValueError:
        print("Please enter a valid number.")
        return
    
    # Check if input_files directory exists
    input_dir = "input_files"
    if not os.path.exists(input_dir):
        print(f"Error: input_files directory not found")
        print("Please create the directory and place your video folder(s) there.")
        return
    
    # List available folders
    folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    if not folders:
        print("No folders found in the input_files directory.")
        return
    
    print("\nAvailable folders:")
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder}")
    
    # Get folder selection from user
    try:
        selection = int(input("\nEnter the number of the folder containing your videos: "))
        if selection < 1 or selection > len(folders):
            print("Invalid selection.")
            return
        selected_folder = folders[selection - 1]
        folder_path = os.path.join(input_dir, selected_folder)
        
        # List videos in selected folder
        video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'))]
        
        if not video_files:
            print(f"No video files found in {selected_folder}.")
            return
            
        if len(video_files) < 2:
            print(f"Found only {len(video_files)} video(s) in {selected_folder}. Need at least 2 videos for analysis.")
            return
        
        print(f"\nVideos found in {selected_folder}:")
        for i, video in enumerate(video_files, 1):
            print(f"{i}. {video}")
        
        # Get video selections
        selection1 = int(input("\nEnter the number of the first video to analyze: "))
        if selection1 < 1 or selection1 > len(video_files):
            print("Invalid selection.")
            return
            
        selection2 = int(input("Enter the number of the second video to analyze: "))
        if selection2 < 1 or selection2 > len(video_files) or selection2 == selection1:
            print("Invalid selection.")
            return
            
        video_path1 = os.path.join(folder_path, video_files[selection1 - 1])
        video_path2 = os.path.join(folder_path, video_files[selection2 - 1])
        
        # Initialize the analyzer with desired framerate
        analyzer = KneeAngleAnalyzer(target_fps=fps)
        
        # Analyze the videos
        analyzer.analyze_video(video_path1, video_path2)
        
    except ValueError:
        print("Please enter a valid number.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 