import cv2
import os
from pathlib import Path
from typing import List

from .video_processor import VideoProcessor
from .angle_calculator import AngleCalculator
from .plotter import Plotter

class KneeAngleAnalyzer:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.angle_calculator = AngleCalculator()
        self.plotter = Plotter()
        
        # Initialize lists to store angle data
        self.left_knee_angles: List[float] = []
        self.right_knee_angles: List[float] = []

    def analyze_video(self, video_path: str):
        """Process the video and analyze knee angles."""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties and create video writer
        width, height, fps = self.video_processor.get_video_properties(cap)
        out = self.video_processor.create_video_writer(video_path, width, height, fps)
        
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
            fps
        )

def main():
    # Print supported video formats
    print("Supported video formats: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm")
    print("Place your video file in the 'input_files' directory.")
    
    # Get video filename from user
    video_filename = input("Enter the video filename (e.g., myvideo.mp4): ")
    video_path = os.path.join("input_files", video_filename)
    
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return
    
    # Initialize the analyzer
    analyzer = KneeAngleAnalyzer()
    
    # Analyze the video
    analyzer.analyze_video(video_path)

if __name__ == "__main__":
    main() 