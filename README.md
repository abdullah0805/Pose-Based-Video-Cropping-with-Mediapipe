# Pose-Based-Video-Cropping-with-Mediapipe
## A single-person video cropping script using Mediapipes pose detection

This project processes input videos, detects poses using Mediapipe Pose, and crops the videos around the detected poses. The output videos are resized to a specified resolution, ensuring consistent dimensions for further analysis or use.

## Features
Pose Detection: Uses Mediapipe to detect human poses in video frames.
Dynamic Cropping: Automatically crops the region around the detected pose with adjustable margins.
Configurable Output: Allows customization of the output video resolution and margin around the cropped area.
Graceful Fallback: Writes a black frame when no pose is detected in a frame.

## How It Works

### Pose Detection:
Mediapipe's Pose solution identifies landmarks representing body joints.
A bounding box is calculated to encompass all detected landmarks.

### Dynamic Cropping:
The bounding box is slightly expanded using a margin to include space around the pose.
The cropped region is centered and resized to a fixed resolution (output_width × output_height).

### Output Video:
The cropped frames are saved as a new video in .mp4 format with the specified resolution.
Fallback for Missing Pose:

If no pose is detected in a frame, a black frame of the target resolution is written to the output.
