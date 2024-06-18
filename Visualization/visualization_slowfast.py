import cv2
import numpy as np
from mmaction.apis import (inference_skeleton,inference_recognizer,init_recognizer)
import pickle
import os
import mmcv
import mmaction
import mmengine

# Function to get label based on class index
def get_label(class_index):
    return "grooming" if class_index == 0 else "playing"

# Paths
config_path = "C:/Users/hagedorn/mmaction2/work_dirs/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py"
checkpoint_path = "C:/Users/hagedorn/mmaction2/work_dirs/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/best_acc_top1_epoch_20.pth"
video_path = "C:/Users/hagedorn/Desktop/data/macaques_data/all_videos/la-grooming2_fragment_13.mp4" 
output_video_path = "slowfast_prediction.mp4"  # Path to save the output video

cap = cv2.VideoCapture(video_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

model = init_recognizer(config_path, checkpoint_path, device="cpu")  # device can be 'cuda:0'
result = inference_recognizer(model, video_path)

# print(result)
pred_label = result.pred_label.item()
pred_score = result.pred_score[pred_label]
label_str = get_label(pred_label)

fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
font_color = (255, 255, 255)  
text_thickness = 3

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    label_text = f"{label_str} {pred_score:.4f}"

    # Get the text size for centering and background box
    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)
    x_position = (frame.shape[1] - text_width) // 2  # Center horizontally
    y_position = 60  # Adjusted lower position
    # Define the background box coordinates
    box_coords = ((x_position, y_position - text_height - baseline), 
                (x_position + text_width, y_position + baseline))

    # Draw the black background rectangle
    cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 255), cv2.FILLED)

    # Add label text at the top of the frame
    cv2.putText(frame, label_text,  (x_position, y_position), font, font_scale, font_color, text_thickness)

    # Write the frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()

print("Output video saved at:", output_video_path)