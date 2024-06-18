import cv2
import numpy as np
from mmaction.apis import (inference_skeleton,
                           init_recognizer)
import pickle
import os
import mmcv
import mmaction
import mmengine

# Function to get label based on class index
def get_label(class_index):
    return "grooming" if class_index == 0 else "playing"

# Paths
config_path = "C:/Users/hagedorn/mmaction2/configs/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py"
checkpoint_path = "C:/Users/hagedorn/mmaction2/work_dirs/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/best_acc_top1_epoch_200.pth"
# video_path = "C:/Users/hagedorn/Desktop/data/macaques_data/all_videos/la-grooming2_fragment_13.mp4" 
video_path = "C:/Users/hagedorn/mmaction2/runs/pose/predict2/la-grooming2_fragment_13.avi"
output_video_path = "2d-st-gcn_prediction.mp4"  # Path to save the output video

with open("C:/Users/hagedorn/Desktop/data/macaques_skeleton_2d_train2.pkl","rb") as f: 
    data = pickle.load(f)

annos = data["annotations"]
videoname = os.path.basename(video_path)[:-4]
ann = next((anno for anno in annos if anno["frame_dir"] == videoname))

# Original image shape
img_shape = ann['img_shape']

cap = cv2.VideoCapture(video_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

pose_results = []
for frame_keypoints, frame_scores in zip(ann['keypoint'], ann['keypoint_score']):
    print("Keypoints shape: ", frame_keypoints.shape)
    print("Scores shape: ", frame_scores.shape)
    frame_result = {
        'keypoints': frame_keypoints,
        'keypoint_scores': np.ones((frames, 17))
        
    }
    pose_results.append(frame_result)

# # print(ann)
model = init_recognizer(config_path, checkpoint_path, device="cpu")  
result = inference_skeleton(model, pose_results, img_shape)

print(result)

# # print(result)
# pred_label = result.pred_label.item()
# pred_score = result.pred_score[pred_label]
# label_str = get_label(pred_label)

# fps = int(cap.get(cv2.CAP_PROP_FPS))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

# frame_count = 0

# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 2
# font_color = (255, 255, 255)  
# text_thickness = 3

# while True:
#     # Read the next frame
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video reached.")
#         break
    
#     # # Check if the frame index exceeds the number of frames for which key points are available
#     # if frame_count >= len(ann['keypoint'][0]):
#     #     print("Frame index exceeds total number of frames with key points.")
#     #     break
    
#     # # Get the corresponding key points from the annotation file
#     # monk1_keypoints = ann['keypoint'][0][frame_count]
#     # monk2_keypoints = ann['keypoint'][1][frame_count]

#     # # Draw key points for monk1 on the frame
#     # for x, y in monk1_keypoints:
#     #     cv2.circle(frame, (int(x), int(y)), 8, (0,0,255), -1)

#     # # Draw key points for monk2 on the frame
#     # for x, y in monk2_keypoints:
#     #     cv2.circle(frame, (int(x), int(y)), 8, (0,255,0), -1)

#     label_text = f"{label_str} {pred_score:.4f}"

#     # Get the text size for centering and background box
#     (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)
#     x_position = (frame.shape[1] - text_width) // 2  # Center horizontally
#     y_position = 60  # Adjusted lower position
#     # Define the background box coordinates
#     box_coords = ((x_position, y_position - text_height - baseline), 
#                 (x_position + text_width, y_position + baseline))

#     # Draw the black background rectangle
#     cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 255), cv2.FILLED)

#     # Add label text at the top of the frame
#     cv2.putText(frame, label_text,  (x_position, y_position), font, font_scale, font_color, text_thickness)

#     # Write the frame to the output video
#     out.write(frame)

#     frame_count += 1

# # Release resources
# cap.release()
# out.release()

# print("Output video saved at:", output_video_path)