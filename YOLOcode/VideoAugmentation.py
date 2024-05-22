import os
import cv2
import numpy as np

# Function to rotate video frames
def rotate_video(input_path, output_path, angle):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        # Rotate the frame
        rotated_frame = cv2.rotate(frame, angle)
        out.write(rotated_frame)

    cap.release()
    out.release()

# Function to flip video frames
def flip_video(input_path, output_path, flip_code):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        # Flip the frame
        flipped_frame = cv2.flip(frame, flip_code)
        out.write(flipped_frame)

    cap.release()
    out.release()

# Function to crop video frames
def crop_video(input_path, output_path, x, y, w, h):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        # Crop the frame
        cropped_frame = frame[y:y+h, x:x+w]
        out.write(cropped_frame)

    cap.release()
    out.release()

# Create the output directory if it doesn't exist
output_dir = "/Users/hagedorn/Desktop/synched_actions/synched-grooming/LA/LA_Grooming_Augmented_Fragments"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Augment each video in the original_videos folder
input_dir = "/Users/hagedorn/Desktop/synched_actions/synched-grooming/LA/LA_Grooming_Fragments"
for filename in os.listdir(input_dir):
    if filename.endswith(".mp4"):
        input_path = os.path.join(input_dir, filename)

        print(f"Processing {input_path}")
        
        # Rotate
        # rotate_video(input_path, os.path.join(output_dir, filename[:-4] + "_rotated.mp4"), angle=cv2.ROTATE_90_CLOCKWISE)
        # rotate_video(input_path, os.path.join(output_dir, filename[:-4] + "_rotated180.mp4"), angle=cv2.ROTATE_180)
        print("Flipping...")
        # Flip
        flip_video(input_path, os.path.join(output_dir, filename[:-4] + "_flipped.mp4"), flip_code=1)  # Horizontal flip
        flip_video(input_path, os.path.join(output_dir, filename[:-4] + "_flipped_vertical.mp4"), flip_code=0)  # Vertical flip
        # print("Cropping...")
        # # Crop
        # crop_video(input_path, os.path.join(output_dir, filename[:-4] + "_cropped.mp4"), x=100, y=100, w=100, h=100)

print("Video Augmentation done!")
print(f"Original numer of videos: {len(os.listdir(input_dir))}")
print(f"New number of videos: {len(os.listdir(output_dir))}")
