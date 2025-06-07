import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import os
from scipy.signal import argrelextrema

class KneeAngleAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize lists to store angle data
        self.left_knee_angles: List[float] = []
        self.right_knee_angles: List[float] = []
        self.frame_count = 0

    def calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        # Return 180 - angle to follow orthopedic convention: straight knee is 0 degrees, bent knee is the flexion angle
        return 180 - angle

    def process_frame(self, frame):
        """Process a single frame and calculate knee angles."""
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect the pose
        results = self.pose.process(image)
        
        if results.pose_landmarks:
            # Draw the pose landmarks
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Get landmarks for left knee
            left_hip = (
                results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y
            )
            left_knee = (
                results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y
            )
            left_ankle = (
                results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y
            )
            
            # Get landmarks for right knee
            right_hip = (
                results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y
            )
            right_knee = (
                results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].x,
                results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].y
            )
            right_ankle = (
                results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y
            )
            
            # Calculate angles
            left_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            
            # Store angles
            self.left_knee_angles.append(left_angle)
            self.right_knee_angles.append(right_angle)
            
            # Display angles on frame
            cv2.putText(frame, f"Left Knee: {left_angle:.1f}°", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Right Knee: {right_angle:.1f}°", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            self.frame_count += 1
            
        return frame

    def analyze_video(self, video_path: str):
        """Process the video and analyze knee angles."""
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display the frame
            cv2.imshow('Knee Angle Analysis', processed_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        self.plot_results()

    def plot_results(self):
        """Plot the knee angles over time and calculate step durations based on local minima/maxima."""
        plt.figure(figsize=(12, 6))
        
        window_size = 5
        left_smoothed = np.convolve(self.left_knee_angles, np.ones(window_size)/window_size, mode='valid')
        right_smoothed = np.convolve(self.right_knee_angles, np.ones(window_size)/window_size, mode='valid')
        
        # Plot left knee angles
        plt.subplot(1, 2, 1)
        plt.plot(self.left_knee_angles, label='Left Knee', alpha=0.5)
        plt.plot(left_smoothed, label='Left Knee (Smoothed)', color='red')
        plt.title('Left Knee Angles Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        
        # Plot right knee angles
        plt.subplot(1, 2, 2)
        plt.plot(self.right_knee_angles, label='Right Knee', alpha=0.5)
        plt.plot(right_smoothed, label='Right Knee (Smoothed)', color='red')
        plt.title('Right Knee Angles Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        
        # Calculate and display statistics
        left_min = min(self.left_knee_angles)
        left_max = max(self.left_knee_angles)
        right_min = min(self.right_knee_angles)
        right_max = max(self.right_knee_angles)
        
        left_amplitude = left_max - left_min
        right_amplitude = right_max - right_min

        fps = 30  # Assuming 30 fps
        # Find local minima and maxima for step duration
        def get_step_durations(angle_array):
            arr = np.array(angle_array)
            # Find local minima and maxima
            minima = argrelextrema(arr, np.less)[0]
            maxima = argrelextrema(arr, np.greater)[0]
            # Combine and sort
            extrema = np.sort(np.concatenate((minima, maxima)))
            if len(extrema) < 2:
                return 0.0, []
            # Calculate durations between consecutive extrema
            durations = np.diff(extrema) / fps
            avg_duration = np.mean(durations) if len(durations) > 0 else 0.0
            return avg_duration, durations

        left_step_duration, left_durations = get_step_durations(self.left_knee_angles)
        right_step_duration, right_durations = get_step_durations(self.right_knee_angles)

        print("\nKnee Angle Statistics:")
        print(f"Left Knee - Min: {left_min:.1f}°, Max: {left_max:.1f}°, Amplitude: {left_amplitude:.1f}°, Step Duration: {left_step_duration:.2f}s")
        print(f"Right Knee - Min: {right_min:.1f}°, Max: {right_max:.1f}°, Amplitude: {right_amplitude:.1f}°, Step Duration: {right_step_duration:.2f}s")
        
        plt.tight_layout()
        plt.show()

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