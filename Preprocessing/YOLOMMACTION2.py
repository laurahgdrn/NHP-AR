"""This code performs 2D skeleton based action recognition on unseen video data. 
First, the pretrained YOLO pose estimation model is applied to each frame to extract the 
poses of the macaques. The key points are collected over a number of frames (num_frames_per_action_rec)
and then fed to the pretrained ST-GCN to perform action recognition."""


import cv2
import numpy as np
from mmaction.apis import inference_skeleton, init_recognizer
import os

from ultralytics import YOLO

model = YOLO("train101-best.pt")
video_path = "WholeNewVideos/extracted_detections/J1 gk4 la-20240402-110000_segment_1_segment_42.mp4"

# Function to get label based on class index
def get_label(class_index):
    return "grooming" if class_index == 0 else "playing"

# Paths
config_path = "C:/Users/hagedorn/mmaction2/configs/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py"
checkpoint_path = "C:/Users/hagedorn/mmaction2/work_dirs/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/best_acc_top1_epoch_20.pth"
output_video_path = "test_prediction.mp4"  # Path to save the output video

# Function to detect poses using YOLO model
def detect_poses(frame):

    results = model.predict(frame)
    keypoints = results[0].keypoints.xy.cpu()
    keypoint_data = np.zeros(((2, 1, 17, 2)))
    keypoint_score = np.ones((2,1, 17))
    for id in range(2): 
        for kp in range(17): 
            keypoint_data[id, 0, kp, 0] = keypoints[id, kp, 0]
            keypoint_data[id, 0, kp, 0] = keypoints[id, kp, 0]
    return keypoint_data, keypoint_score


# Initialize skeleton-based action recognition model
model = init_recognizer(config_path, checkpoint_path, device="cpu")

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Failed to open video file.")
    exit()

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

num_frames_per_action_rec = 100
# Process video frames
frame_number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1
    keypoints_over_frames = []
    scores_over_frames = []
    # Perform pose estimation every 30 frames
    if frame_number % num_frames_per_action_rec == 0:
        # Detect poses using YOLO model
        keypoint_data, keypoint_score = detect_poses(frame)
        keypoints_over_frames.append(keypoint_data.tolist())
        scores_over_frames.append(keypoint_score.tolist())

        # Perform action recognition using skeleton data
        result = inference_skeleton(model, skeleton_data)

        # Overlay action label on frame
        action_label = get_label(result['class'])
        cv2.putText(frame, action_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

    # Display frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
