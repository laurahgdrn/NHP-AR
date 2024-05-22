import os
import cv2
import json
import numpy as np
import pickle
from ultralytics import YOLO

def get_class_from_filename(file): 
    if os.path.basename(file[3:6]) == "gro": 
        return 0
    else: 
        return 1
    
def process_video(video_path, model, label):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    keypoints_over_frames = []
    scores_over_frames = []

    detected_frames = 0  # Counter for frames with detections

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print(f"Processing {os.path.basename(video_path)}")
        results = model.predict(frame)
        num = len(results[0].boxes.xyxy.tolist()) 

        if num >= 2: 
            detected_frames += 1  # Increment detected frame count
            
            frame_dir =  os.path.basename(video_path)[:-4]
            if frame_dir: 
                print(f"frame dir: {frame_dir}")
            else: 
                print("Frame dir is None")
                return None
            keypoints = results[0].keypoints.xy.cpu()
            
            keypoint_data = np.zeros((2, 1, 17, 2))
            keypoint_score = np.ones((2, 1, 17))
            
            for person_id in range(2):
                for keypoint_id in range(17):
                    keypoint_data[person_id, 0, keypoint_id, 0] = keypoints[person_id, keypoint_id, 0]
                    keypoint_data[person_id, 0, keypoint_id, 1] = keypoints[person_id, keypoint_id, 1]
            keypoints_over_frames.append(keypoint_data.tolist())
            scores_over_frames.append(keypoint_score.tolist())

    print(f"Processing video: {os.path.basename(video_path)}")
    print(f"Total frames: {total_frames}")
    print(f"Detected frames: {detected_frames}")
    if detected_frames > 0: 
        annotation = {
            "frame_dir": frame_dir,
            "total_frames": detected_frames,
            "img_shape": (frame_height, frame_width),
            "original_shape": (frame_height, frame_width),
            "label": label, 
            "keypoint": np.concatenate(keypoints_over_frames, axis=1),
            "keypoint_score": np.concatenate(scores_over_frames, axis=1)
        }
    else: 
        return None 
    
    assert isinstance(annotation["frame_dir"], str), f"frame_dir should be an int and is a {type(os.path.basename(video_path))}"
    assert isinstance(annotation["img_shape"], tuple) and all(isinstance(x, int) for x in annotation["img_shape"]), f"img_shape should be a tuple[int] and is a {type(annotation['img_shape'])}"
    assert isinstance(annotation["label"], int), f"label should be an int and is a {type(label)}"
    assert isinstance(annotation["keypoint"], np.ndarray) and keypoint_data.ndim == 4 and keypoint_data.shape[2:] == (17, 2), f"keypoint should be a np.ndarray with shape [num_persons, total_frames, 17, 2] and is a {type(keypoint_data)} and shape {keypoint_data.shape}"
    assert isinstance(annotation["keypoint_score"], np.ndarray) and keypoint_score.ndim == 3 and keypoint_score.shape[2] == 17, f"keypoint_score should be a np.ndarray with shape [num_persons, total_frames, 17] and is a {type(keypoint_score)} and shape {keypoint_score.shape}"

    cap.release()

    return annotation 

def main():
    videos_train = "/Users/hagedorn/Desktop/YOLO/train/RA"
    videos_val = "/Users/hagedorn/Desktop/YOLO/val/RA"

    model=YOLO("/Users/hagedorn/runs/pose/train2/weights/best.pt") 

    print("Initiating the model...")

    dataset_annotations = []
    split_annotations = {"xsub_train": [], "xsub_val": []}
 
    for folder, f in [(videos_train, "train"), (videos_val, "val")]:
        for video_file in os.listdir(folder):
            if video_file.endswith(".mp4"):
                print(f"Processing {video_file}")
                video_path = os.path.join(folder, video_file)
                label = get_class_from_filename(video_file)
                annotation = process_video(video_path, model, label)
                if annotation:
                    dataset_annotations.append(annotation)
                    split_annotations["xsub_train" if f == "train" else "xsub_val"].append(os.path.splitext(video_file)[0])\
                    
    # Save dataset annotations as a pickle file
    output_pkl_file = "macaques_skeleton_2d_onlyRA.pkl"
    with open(output_pkl_file, "wb") as f:
        pickle.dump({"split": split_annotations, "annotations": dataset_annotations}, f)
    print("Pickle file saved for the action recognition dataset")

if __name__ == "__main__":
    main()
