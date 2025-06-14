import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from typing import List, Tuple

class SkeletonVisualizer3D:
    def __init__(self):
        """Initialize the 3D visualization window."""
        pygame.init()
        display = (800, 600)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        
        # Set up the perspective
        gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)
        
        # Initialize rotation angles
        self.rot_x = 0
        self.rot_y = 0
        
        # Set up lighting
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set up light
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 1, 0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.5, 0.5, 0.5, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))

    def draw_joint(self, position: Tuple[float, float, float], size: float = 0.1):
        """Draw a joint as a sphere at the given position."""
        glPushMatrix()
        glTranslatef(position[0], position[1], position[2])
        
        # Create a sphere
        quad = gluNewQuadric()
        gluSphere(quad, size, 32, 32)
        
        glPopMatrix()

    def draw_bone(self, start: Tuple[float, float, float], end: Tuple[float, float, float], width: float = 0.05):
        """Draw a bone as a cylinder between two points."""
        # Calculate the direction vector
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction)
        
        if length == 0:
            return
            
        # Normalize the direction
        direction = direction / length
        
        # Calculate the rotation axis and angle
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction)
        rotation_angle = np.arccos(np.dot(z_axis, direction)) * 180 / np.pi
        
        glPushMatrix()
        
        # Move to the start position
        glTranslatef(start[0], start[1], start[2])
        
        # Rotate to align with the direction
        if np.any(rotation_axis):
            glRotatef(rotation_angle, rotation_axis[0], rotation_axis[1], rotation_axis[2])
        
        # Draw the cylinder
        quad = gluNewQuadric()
        gluCylinder(quad, width, width, length, 32, 1)
        
        glPopMatrix()

    def draw_skeleton(self, landmarks_3d: List[Tuple[float, float, float]]):
        """Draw the 3D skeleton using the provided landmarks."""
        # Clear the screen and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Apply rotation
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        
        # Draw joints
        for landmark in landmarks_3d:
            self.draw_joint(landmark)
            
        # Draw bones (connections between joints)
        # Left leg
        self.draw_bone(landmarks_3d[0], landmarks_3d[1])  # Hip to knee
        self.draw_bone(landmarks_3d[1], landmarks_3d[2])  # Knee to ankle
        
        # Right leg
        self.draw_bone(landmarks_3d[3], landmarks_3d[4])  # Hip to knee
        self.draw_bone(landmarks_3d[4], landmarks_3d[5])  # Knee to ankle
        
        pygame.display.flip()
        
    def handle_input(self):
        """Handle keyboard and mouse input for camera control."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.rot_y += 5
                elif event.key == pygame.K_RIGHT:
                    self.rot_y -= 5
                elif event.key == pygame.K_UP:
                    self.rot_x += 5
                elif event.key == pygame.K_DOWN:
                    self.rot_x -= 5
        return True
        
    def cleanup(self):
        """Clean up resources."""
        pygame.quit() 