import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize MediaPipe Pose
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

def process_video(input_path, output_path, output_resolution, margin):
    """Processes a single video, crops around detected poses, and saves the output."""
    output_width, output_height = output_resolution
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        
        if results.pose_landmarks:
            # Calculate bounding box for the pose
            x_min, x_max, y_min, y_max = w, 0, h, 0
            for lm in results.pose_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, cx)
                x_max = max(x_max, cx)
                y_min = min(y_min, cy)
                y_max = max(y_max, cy)
            
            # Expand the bounding box slightly
            x_min = max(x_min - margin, 0)
            x_max = min(x_max + margin, w)
            y_min = max(y_min - margin, 0)
            y_max = min(y_max + margin, h)

            # Center the ROI within the fixed output dimensions
            roi_center_x = (x_min + x_max) // 2
            roi_center_y = (y_min + y_max) // 2

            # Calculate the cropping box
            crop_x_min = max(roi_center_x - output_width // 2, 0)
            crop_x_max = crop_x_min + output_width
            crop_y_min = max(roi_center_y - output_height // 2, 0)
            crop_y_max = crop_y_min + output_height

            # Adjust if crop goes out of bounds
            if crop_x_max > w:
                crop_x_min = w - output_width
                crop_x_max = w
            if crop_y_max > h:
                crop_y_min = h - output_height
                crop_y_max = h

            # Ensure crop stays within the frame
            crop_x_min = max(crop_x_min, 0)
            crop_x_max = min(crop_x_max, w)
            crop_y_min = max(crop_y_min, 0)
            crop_y_max = min(crop_y_max, h)

            # Extract and write the cropped frame
            cropped_frame = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            out.write(cropped_frame)
    
    cap.release()
    out.release()

def main(input_dir, output_dir, output_resolution=(384, 288), margin=30):
    """
    Main function to process videos from an input directory and save them to an output directory.
    
    Args:
        input_dir (str): Path to the directory containing input videos.
        output_dir (str): Path to save the processed videos.
        output_resolution (tuple): Output video resolution as (width, height).
        margin (int): Margin to expand the bounding box around the detected pose.
    """
    os.makedirs(output_dir, exist_ok=True)
    for video_file in os.listdir(input_dir):
        if video_file.endswith('.mp4'):
            input_path = os.path.join(input_dir, video_file)
            output_path = os.path.join(output_dir, f'Cropped_{video_file}')
            print(f'Processing {video_file}...')
            process_video(input_path, output_path, output_resolution, margin)
    print("Processing complete. Cropped videos saved in", output_dir)

if __name__ == "__main__":
    # Example usage
    input_dir = './Input_Videos'
    output_dir = './Cropped_Videos'
    output_resolution = (384, 288)  # Change this as needed
    margin = 50  # Adjust the margin as needed
    main(input_dir, output_dir, output_resolution, margin)
