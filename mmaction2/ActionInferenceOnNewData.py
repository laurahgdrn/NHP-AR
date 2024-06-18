"""This code performs 2D skeleton based action recognition on several unseen videos. 
The 2D skeletons are extracted by the pretrained YOLO model and then fed to the ST-GCN.""" 

import cv2
import numpy as np
from mmaction.apis import inference_skeleton, init_recognizer
import torch
from ultralytics import YOLO
import os

def get_class(file): 
    file = os.path.basename(file)
    if file[3:6] == 'gro': 
        return 0 
    else: 
        return 1 
    
# Initialize mmaction2 model
config_path = "work_dirs/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py"
checkpoint_path = "work_dirs/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/best_acc_top1_epoch_20.pth"
mmaction_model = init_recognizer(config_path, checkpoint_path, device="cuda:0") # or "cpu"

predictions_list = []
ground_truth_list = []

# Function to get pose result using YOLO
def get_pose_result(frame, model):
    results = model(frame)
    keypoints = results[0].keypoints.xy.cpu().numpy()  # Assuming the results contain keypoints

    num_persons = len(results[0].boxes.xyxy.tolist())
    keypoint_data = np.zeros((num_persons, 1, 17, 2))
    keypoint_score = np.ones((num_persons, 1, 17))

    for person_id in range(num_persons):
        for keypoint_id in range(17):
            keypoint_data[person_id, 0, keypoint_id, 0] = keypoints[person_id, keypoint_id, 0]
            keypoint_data[person_id, 0, keypoint_id, 1] = keypoints[person_id, keypoint_id, 1]

    return {'keypoints': keypoint_data, 'keypoint_scores': keypoint_score}

# Process video and perform action recognition
def process_video(video_path, yolo_model, mmaction_model):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames in video:", total_frames)

    ground_truth = get_class(video_path) # get ground truth based on filename
    pose_results = []
    frame_count = 0
    action_label = "unknown"
    action_score = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get pose result for the current frame
        pose_result = get_pose_result(frame, yolo_model)
        pose_results.append(pose_result)
        
        frame_count += 1
        
        # Perform action recognition at the end of the video (or every 50, 100, 200 frames)
        if len(pose_results) == total_frames:
            formatted_pose_results = []
            for pose in pose_results:
                keypoints = pose['keypoints']
                keypoint_scores = pose['keypoint_scores']
                num_persons = keypoints.shape[0]

                for person_id in range(num_persons):
                    formatted_pose_results.append({
                        'keypoints': keypoints[person_id].reshape(1, 17, 2),
                        'keypoint_scores': keypoint_scores[person_id].reshape(1, 17)
                    })

            action_result = inference_skeleton(mmaction_model, formatted_pose_results, (height, width))
            
            pred_label = action_result.pred_label.item()
            pred_score = action_result.pred_score[pred_label]
            action_label = "grooming" if pred_label == 0 else "playing"
            action_score = pred_score
            print(f"Prediction: {action_label}, Score: {action_score}")
            return pred_label, ground_truth 

    pose_results = []

    cap.release()
    cv2.destroyAllWindows()
    return ground_truth_list, predictions_list
# # Load YOLO model
yolo_model = YOLO("/Users/hagedorn/runs/pose/train2/weights/best.pt")
video_dir = "/Users/hagedorn/mmaction2/tools/data/macaques/val" # test set directory containing new/unseen videos
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

for video in video_files: 
    video_path = os.path.join(video_dir, video)
    pred_label, ground_truth = process_video(video_path, yolo_model, mmaction_model)
    predictions_list.append(pred_label)
    ground_truth_list.append(ground_truth)
print(ground_truth_list, predictions_list)
    

