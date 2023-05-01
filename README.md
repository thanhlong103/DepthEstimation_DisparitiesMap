# 3D Reconstruction using Stereo Vision

This project is a demonstration of 3D reconstruction using stereo vision. It uses a pair of calibrated cameras to capture two images of the same scene from slightly different perspectives. The disparity map is computed from these images, which is then used to create a 3D model of the scene.

## Prerequisites
- Python 3.6+
- OpenCV
- NumPy
- Open3D

## Getting started

1. Clone this repository.

2. Install the required dependencies using the following command:

```bash
pip install opencv-python numpy open3d
```

3. Download two videos of the same scene captured by two different cameras and place them in the same directory as the Python file.

4. Run the following command to start the program:

```bash
python main.py
```

5. Click on any point in the left image to get the distance of that point from the camera. If the distance is less than 50 cm, it will show a warning.

6. After processing is complete, the 3D model will be saved as a `.ply` file in the same directory as the Python file.

## Acknowledgments

- This project was inspired by [Ankur Handa's work on 3D reconstruction using stereo vision](https://github.com/ankurhanda/stereo-vision).
- The `write_ply` function used in this project is based on [this code snippet](https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/examples/depth-to-3d-ply.ipynb) from Intel RealSense.
