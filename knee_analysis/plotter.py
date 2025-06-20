import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import argrelextrema
from typing import List

class Plotter:
    def __init__(self):
        self.output_dir = Path("output_files")
        self.output_dir.mkdir(exist_ok=True)

    def plot_knee_angles(self, video_name: str, left_angles: List[float], right_angles: List[float], fps: int = 30):
        plt.figure(figsize=(12, 6))
        window_size = 5
        left_smoothed = np.convolve(left_angles, np.ones(window_size)/window_size, mode='valid')
        right_smoothed = np.convolve(right_angles, np.ones(window_size)/window_size, mode='valid')
        plt.subplot(1, 2, 1)
        plt.plot(left_angles, label='Left Knee', alpha=0.5)
        plt.plot(left_smoothed, label='Left Knee (Smoothed)', color='red')
        plt.title(f'Left Knee Angles Over Time - {video_name}')
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(right_angles, label='Right Knee', alpha=0.5)
        plt.plot(right_smoothed, label='Right Knee (Smoothed)', color='red')
        plt.title(f'Right Knee Angles Over Time - {video_name}')
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        stats = self._calculate_statistics(left_angles, right_angles, fps)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{video_name}_knee_angles.png")
        plt.show()
        return stats

    def _calculate_statistics(self, left_angles: List[float], right_angles: List[float], fps: int) -> dict:
        left_min = min(left_angles)
        left_max = max(left_angles)
        right_min = min(right_angles)
        right_max = max(right_angles)
        left_amplitude = left_max - left_min
        right_amplitude = right_max - right_min
        def get_step_durations(angle_array):
            arr = np.array(angle_array)
            minima = argrelextrema(arr, np.less)[0]
            maxima = argrelextrema(arr, np.greater)[0]
            extrema = np.sort(np.concatenate((minima, maxima)))
            if len(extrema) < 2:
                return 0.0, []
            durations = np.diff(extrema) / fps
            avg_duration = np.mean(durations) if len(durations) > 0 else 0.0
            return avg_duration, durations
        left_step_duration, left_durations = get_step_durations(left_angles)
        right_step_duration, right_durations = get_step_durations(right_angles)
        stats = {
            'left': {
                'min': left_min,
                'max': left_max,
                'amplitude': left_amplitude,
                'step_duration': left_step_duration
            },
            'right': {
                'min': right_min,
                'max': right_max,
                'amplitude': right_amplitude,
                'step_duration': right_step_duration
            }
        }
        print("\nKnee Angle Statistics:")
        print(f"Left Knee - Min: {left_min:.1f}°, Max: {left_max:.1f}°, Amplitude: {left_amplitude:.1f}°, Step Duration: {left_step_duration:.2f}s")
        print(f"Right Knee - Min: {right_min:.1f}°, Max: {right_max:.1f}°, Amplitude: {right_amplitude:.1f}°, Step Duration: {right_step_duration:.2f}s")
        return stats 