import numpy as np

class Skeleton:
    def __init__(self):
        # Knochenlängen in Metern (Beispielwerte, anpassbar)
        self.bone_lengths = {
            "femur_left": 0.45,
            "femur_right": 0.45,
            "tibia_left": 0.43,
            "tibia_right": 0.43,
            "spine": 0.60,
            # ... weitere Knochen
        }
        # Gelenk-Constraints (in Grad, als Beispiele, Typen anpassbar)
        self.joint_constraints = {
            "knee_flexion_min": 0,      # 
            "knee_flexion_max": 140,    # maximale Beugung
            "knee_extension_max": -10,    # 0 Grad volle Streckung, -10 starke Überstreckung
            "hip_flexion_max": 120,
            "hip_extension_max": 30,
            # ... weitere Gelenke
        }
        # Optionale Flexibilität für Wirbelsäule, von Domain-Experten festzulegen
        self.spine_flexibility = 45  # maximaler Beugewinkel in Grad

    def get_bone_length(self, bone):
        return self.bone_lengths.get(bone, None)

    def get_joint_constraint(self, joint):
        return self.joint_constraints.get(joint, None)

    def is_knee_angle_valid(self, angle):
        """Prüft, ob der Kniewinkel biomechanisch plausibel ist."""
        return (self.joint_constraints["knee_flexion_min"] <= angle <= self.joint_constraints["knee_flexion_max"])