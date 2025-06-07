# Motion Pattern Analysis Orthopedic

This script analyzes video footage of human movement to track and measure knee angles over time. It uses computer vision and pose estimation to detect knee positions and calculate angles, which can be useful for analyzing knee injuries or movement patterns.

## Features

- Real-time knee angle detection and visualization
- Tracking of both left and right knee angles
- Calculation of minimum and maximum angles
- Graphical representation of angle changes over time
- Pose visualization on video

## Requirements

- Python 3.9+
- Poetry (Python package manager)

## Installation

1. Clone this repository
2. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies using Poetry:
```bash
poetry install
```

## Usage

1. Activate the Poetry environment:
```bash
poetry shell
```

2. Run the script:
```bash
python knee_angle_analyzer.py
```

3. When prompted, enter the video filename (e.g., `myvideo.mp4`) from the `input_files` directory.
4. The script will:
   - Show the video with pose detection and angle measurements
   - Display real-time knee angles on the video
   - Generate plots of angle changes over time
   - Print minimum and maximum angles for both knees

5. Press 'q' to stop the video analysis and view the results

## Notes

- The video should show the person's full body, especially the legs
- Good lighting and clear visibility of the person are important for accurate detection
- The person should be wearing clothing that allows for clear distinction of body parts
- For best results, the camera should be positioned to capture the person from the side or front

## Output

The script provides:
- Real-time visualization of pose detection and angles
- Graphs showing angle changes over time for both knees
- Minimum and maximum angle measurements for both knees
- Statistical analysis of the movement patterns

## Development

To add new dependencies:
```bash
poetry add package-name
```

To update dependencies:
```bash
poetry update
```
