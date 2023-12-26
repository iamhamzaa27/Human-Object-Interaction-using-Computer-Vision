import cv2
import os

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the number of frames to skip to get 15 frames per second
    frame_skip = int(fps / 15)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Check if the current frame number is a multiple of frame_skip
        if frame_count % frame_skip == 0:
            # Save the frame
            output_path = os.path.join(output_folder, f"frame7_{frame_count}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved frame {frame_count} at {output_path}")

    cap.release()

# Usage: Replace 'input_video.mp4' with the path to your video file
# and 'output_folder' with the path to the folder where you want to save frames
extract_frames("D:\Semester 7\ML\dataset/false4.mp4", "D:\Semester 7\ML\dataset\False")
