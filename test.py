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
        
        results = model.predict(frame)
        num = len(results[0].boxes.xyxy.tolist()) 
        if num == 2: 
            detected_frames += 1  # Increment detected frame count
            
            frame_dir =  os.path.basename(video_path)
            keypoints = results[0].keypoints.xy.cpu()
            
            keypoint_data = np.zeros((num, 1, 17, 2))
            keypoint_score = np.ones((num, 1, 17))
            
            for person_id in range(num):
                for keypoint_id in range(17):
                    keypoint_data[person_id, 0, keypoint_id, 0] = keypoints[person_id, keypoint_id, 0]
                    keypoint_data[person_id, 0, keypoint_id, 1] = keypoints[person_id, keypoint_id, 1]
            keypoints_over_frames.append(keypoint_data.tolist())
            scores_over_frames.append(keypoint_score.tolist())

    print(len(keypoints_over_frames))
    print(f"Total frames is {total_frames}")
    print(f"Detected frames is {detected_frames}")

    annotation = {
        "frame_dir": frame_dir,
        "total_frames": detected_frames,
        "img_shape": (frame_height, frame_width),
        "original_shape": (frame_height, frame_width),
        "label": label, 
        "keypoint": np.concatenate(keypoints_over_frames, axis=1),
        "keypoint_score": np.concatenate(scores_over_frames, axis=1)
    }
    
    # assert isinstance(os.path.basename(video_path), str), f"frame_dir should be an int and is a {type(os.path.basename(video_path))}"
    # assert isinstance(annotation["img_shape"], tuple) and all(isinstance(x, int) for x in annotation["img_shape"]), f"img_shape should be a tuple[int] and is a {type(annotation['img_shape'])}"
    # assert isinstance(label, int), f"label should be an int and is a {type(label)}"
    # assert isinstance(keypoint_data, np.ndarray) and keypoint_data.ndim == 4 and keypoint_data.shape[2:] == (17, 2), f"keypoint should be a np.ndarray with shape [num_persons, total_frames, 17, 2] and is a {type(keypoint_data)} and shape {keypoint_data.shape}"
    # assert isinstance(keypoint_score, np.ndarray) and keypoint_score.ndim == 3 and keypoint_score.shape[2] == 17, f"keypoint_score should be a np.ndarray with shape [num_persons, total_frames, 17] and is a {type(keypoint_score)} and shape {keypoint_score.shape}"

    cap.release()

    return annotation
def main():
    # videos_train = "/Users/hagedorn/mmaction2/tools/data/macaques/train60"
    # videos_val = "/Users/hagedorn/mmaction2/tools/data/macaques/val60"
    model_path = "/Users/hagedorn/runs/pose/train3/weights/best.pt"

    model = YOLO(model_path)

    video_path = "/Users/hagedorn/mmaction2/tools/data/macaques/train60/la-grooming1_fragment_1.mp4"
    dataset_annotations = process_video(video_path, model, 0)
    print(f"Keypoint annotation shape: ", dataset_annotations["keypoint"].shape)
    # output_json_file = "small_annotations_2.json"
    # with open(output_json_file, "w") as f:
    #     json.dump({"annotations": dataset_annotations}, f, indent=4)
    # print("Json file saved for the action recognition dataset")

if __name__ == "__main__":
    main()
