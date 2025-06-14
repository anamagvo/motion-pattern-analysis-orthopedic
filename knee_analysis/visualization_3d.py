import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SkeletonVisualizer3D:
    def __init__(self):
        """Initialize the 3D visualizer with matplotlib."""
        self.fig = plt.figure(figsize=(20, 5))
        
        # Create subplots for 2D and 3D views
        self.ax_2d = self.fig.add_subplot(141)
        self.ax_front = self.fig.add_subplot(142, projection='3d')
        self.ax_side = self.fig.add_subplot(143, projection='3d')
        self.ax_top = self.fig.add_subplot(144, projection='3d')
        
        # Set titles and labels
        self.ax_2d.set_title('2D Knee Angles')
        self.ax_2d.set_xlabel('Frame')
        self.ax_2d.set_ylabel('Angle (degrees)')
        
        self.ax_front.set_title('Front View')
        self.ax_side.set_title('Side View')
        self.ax_top.set_title('Top View')
        
        # Set initial view angles
        self.ax_front.view_init(elev=0, azim=0)
        self.ax_side.view_init(elev=0, azim=90)
        self.ax_top.view_init(elev=90, azim=0)
        
        # Initialize data storage
        self.frame_count = 0
        self.left_angles_2d = []
        self.right_angles_2d = []
        self.left_angles_3d = []
        self.right_angles_3d = []
        
        # Show the plot
        plt.ion()
        plt.show()
    
    def update(self, left_angle_2d=None, right_angle_2d=None, left_angle_3d=None, right_angle_3d=None):
        """Update the visualization with new angle data."""
        self.frame_count += 1
        
        # Store angles
        if left_angle_2d is not None:
            self.left_angles_2d.append(left_angle_2d)
        if right_angle_2d is not None:
            self.right_angles_2d.append(right_angle_2d)
        if left_angle_3d is not None:
            self.left_angles_3d.append(left_angle_3d)
        if right_angle_3d is not None:
            self.right_angles_3d.append(right_angle_3d)
        
        # Clear previous plots
        self.ax_2d.clear()
        self.ax_front.clear()
        self.ax_side.clear()
        self.ax_top.clear()
        
        # Plot 2D angles
        frames = range(len(self.left_angles_2d))
        self.ax_2d.plot(frames, self.left_angles_2d, 'b-', label='Left Knee')
        self.ax_2d.plot(frames, self.right_angles_2d, 'r-', label='Right Knee')
        self.ax_2d.set_title('2D Knee Angles')
        self.ax_2d.set_xlabel('Frame')
        self.ax_2d.set_ylabel('Angle (degrees)')
        self.ax_2d.legend()
        
        # Create simple 3D visualization using 2D angles
        if self.left_angles_2d or self.right_angles_2d:
            # Use the last angle for visualization
            left_angle = self.left_angles_2d[-1] if self.left_angles_2d else 0
            right_angle = self.right_angles_2d[-1] if self.right_angles_2d else 0
            
            # Create simple skeleton visualization
            self._plot_simple_skeleton(self.ax_front, left_angle, right_angle, view='front')
            self._plot_simple_skeleton(self.ax_side, left_angle, right_angle, view='side')
            self._plot_simple_skeleton(self.ax_top, left_angle, right_angle, view='top')
        
        # Update the plot
        plt.draw()
        plt.pause(0.001)
    
    def _plot_simple_skeleton(self, ax, left_angle, right_angle, view='front'):
        """Plot a simple skeleton visualization using 2D angles."""
        # Create points for a simple skeleton
        points = {
            'hip': (0, 0, 0),
            'left_knee': (-0.5, -0.5, 0),
            'right_knee': (0.5, -0.5, 0),
            'left_ankle': (-0.5, -1, 0),
            'right_ankle': (0.5, -1, 0)
        }
        
        # Apply angle-based transformations
        if view == 'front':
            # Rotate knees based on angles
            left_knee = self._rotate_point(points['left_knee'], points['hip'], left_angle)
            right_knee = self._rotate_point(points['right_knee'], points['hip'], right_angle)
            points['left_knee'] = left_knee
            points['right_knee'] = right_knee
        elif view == 'side':
            # Show side view with depth
            points = {k: (v[0], v[1], v[2] + 0.5) for k, v in points.items()}
        else:  # top view
            # Show top view
            points = {k: (v[0], v[2], v[1]) for k, v in points.items()}
        
        # Plot points
        for point in points.values():
            ax.scatter(*point, c='r', marker='o')
        
        # Plot connections
        ax.plot([points['hip'][0], points['left_knee'][0]],
                [points['hip'][1], points['left_knee'][1]],
                [points['hip'][2], points['left_knee'][2]], 'b-')
        ax.plot([points['hip'][0], points['right_knee'][0]],
                [points['hip'][1], points['right_knee'][1]],
                [points['hip'][2], points['right_knee'][2]], 'b-')
        ax.plot([points['left_knee'][0], points['left_ankle'][0]],
                [points['left_knee'][1], points['left_ankle'][1]],
                [points['left_knee'][2], points['left_ankle'][2]], 'b-')
        ax.plot([points['right_knee'][0], points['right_ankle'][0]],
                [points['right_knee'][1], points['right_ankle'][1]],
                [points['right_knee'][2], points['right_ankle'][2]], 'b-')
        
        # Set view
        if view == 'front':
            ax.view_init(elev=0, azim=0)
        elif view == 'side':
            ax.view_init(elev=0, azim=90)
        else:  # top view
            ax.view_init(elev=90, azim=0)
        
        ax.set_title(f'{view.title()} View')
    
    def _rotate_point(self, point, center, angle_degrees):
        """Rotate a point around a center by an angle in degrees."""
        import math
        angle_rad = math.radians(angle_degrees)
        x, y, z = point
        cx, cy, cz = center
        
        # Translate point to origin
        x -= cx
        y -= cy
        
        # Rotate point
        new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        
        # Translate back
        return (new_x + cx, new_y + cy, z)
    
    def cleanup(self):
        """Clean up resources."""
        plt.close(self.fig)
        
    def handle_input(self):
        """Handle keyboard input for view control."""
        for event in plt.get_current_fig_manager().canvas.events:
            if event.type == 'close':
                plt.close('all')
                return False
        return True 