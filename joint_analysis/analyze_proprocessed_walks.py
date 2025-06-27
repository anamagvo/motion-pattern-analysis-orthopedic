from collections import defaultdict
import os
import pandas as pd
import matplotlib.pyplot as plt
import re


base_data_dir = os.path.expanduser(os.path.join("~", "dev", "opencap-core-local", "Data"))

# --- User input: set your session and trial ---
session_id = '47b0704f-1ff8-4c17-a6b0-284f1dc2930d' #Kristian
#'e90ce2ce-6c8a-4712-ad25-7e8027757811' #Björn
#

trial_name = 'Kristian_walk_back'  # or any other trial name you have

trial_mot = '23bc052a-ea07-49e4-ba05-2dd31b420d3f.mot' #Kristian walk back knee
#'f2dc9838-6b9f-4871-8966-b8ce55d6ae8d.mot'  # Kristian walk normal
mot_file = os.path.join(
    base_data_dir,
    session_id,
    "OpenSimData",
    "Kinematics",
    trial_mot
)
# adjust filename trial_sto (this one is walk normal for Kristian)

print(f"Analyzing trial: {trial_name} in session: {session_id}")
print(f"Using .sto file: {mot_file}")

# --- Read .mot file (skip OpenSim headers) ---
with open(mot_file) as f:
    lines = f.readlines()

# Find where the data starts (after "endheader")
for i, line in enumerate(lines):
    if 'endheader' in line:
        start = i + 1
        break

df = pd.read_csv(mot_file, sep='\t', skiprows=start)

print("Available columns:", list(df.columns))

# # --- Choose joint angle column to plot ---
# joint = 'knee_angle_l'  # or the exact column name from your file

# if joint not in df.columns:
#     raise ValueError(f"Joint angle '{joint}' not found in columns.")

# plt.figure(figsize=(10, 5))
# plt.plot(df['time'], df[joint], label=joint)
# plt.xlabel('Time (s)')
# plt.ylabel('Angle (deg)')
# plt.title(f'{joint} over time ({trial_name})')
# plt.legend()
# plt.grid(True)
# plt.show()


# --- Plot all 3D joint angles over time ---
# Find all unique joint names (before the first underscore)
output_dir = os.path.join("output", session_id, trial_name)
os.makedirs(output_dir, exist_ok=True)

angle_cols = [col for col in df.columns if col != 'time']
n = len(angle_cols)


joint_groups = defaultdict(list)
for col in angle_cols: 
    prefix = col.split('_')[0]  # Get the joint name prefix
    joint_groups[prefix].append(col)

for joint, cols in joint_groups.items():
    n_joint_groups = len(cols)
    if n_joint_groups > 1:
        print(f"Joint '{joint}' has multiple columns: {cols}")
        fig, axes = plt.subplots(n_joint_groups, 1, figsize=(10, 3 * n_joint_groups), sharex=True)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 3))
        axes = [axes]

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(n_joint_groups)]   

    for i, col in enumerate(cols):
        axes[i].plot(df['time'], df[col], label=col, color=colors[i])
        axes[i].set_ylabel('Angle (deg)')
        axes[i].set_title(f'{col} over time ({trial_name})')
        axes[i].grid(True)
        axes[i].legend()
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{joint}_angles.png'))
    #plt.show()
    plt.close(fig)
    print(f"Saved plot for joint '{joint}' with {n_joint_groups} columns to {output_dir}/{joint}_angles.png")

