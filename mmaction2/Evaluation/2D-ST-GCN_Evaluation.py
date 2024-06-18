"""This code explores additional evaluation metrics than top-1-accuracy. 
This is achieved by manually iterating through the validation data set and 
comparing the ground truth to the model's predictions. 

Evaluation metrics are accuracy_score, precision_score, recall_score, average_precision_score from sklearn. """"

import pickle
import numpy as np
from mmaction.apis import inference_skeleton, init_recognizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score

# Load the pickle file
pickle_file_path = "/Users/hagedorn/mmaction2/macaques_skeleton_2d_train2.pkl"
config_path = "/Users/hagedorn/mmaction2/mmaction2/work_dirs/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py"
checkpoint_path = "/Users/hagedorn/mmaction2/mmaction2/work_dirs/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/best_acc_top1_epoch_10.pth"
mmaction_model = init_recognizer(config_path, checkpoint_path, device="cpu")

dim = 2 # for 2D 

with open(pickle_file_path, "rb") as f:
    data = pickle.load(f)

# Extract validation set annotations
val_videos = data["split"]["xsub_val"]
annotations = data["annotations"]

# Function to get annotations for a specific frame_dir
def get_annotation_for_frame_dir(frame_dir, annotations):
    for annotation in annotations:
        if annotation["frame_dir"] == frame_dir:
            return annotation
    return None

# Extract 3D skeleton data for validation set and perform action recognition
predictions_list = []
ground_truth_list = []
all_pred_scores = []

for frame_dir in val_videos:
    annotation = get_annotation_for_frame_dir(frame_dir, annotations)
    if annotation is not None:
        keypoints = annotation["keypoint"]  # Shape: [M x T x V x C]
        if dim == 2: 
            keypoint_scores = annotation["keypoint_score"]  # Shape: [M x T x V]

        # Format keypoints and keypoint_scores for inference
        num_persons, num_frames, num_keypoints, num_coords = keypoints.shape
        formatted_pose_results = []

        for person_id in range(num_persons):
            if dim == 2: 

                formatted_pose_results.append({
                    'keypoints': keypoints[person_id].reshape(num_frames, num_keypoints, num_coords),
                    'keypoint_scores': keypoint_scores[person_id].reshape(num_frames, num_keypoints)
                })
            if dim == 3: 
                formatted_pose_results.append({
                    'keypoints': keypoints[person_id].reshape(num_frames, num_keypoints, num_coords)
                })

        # Perform action recognition
        height, width = annotation["img_shape"]
        action_result = inference_skeleton(mmaction_model, formatted_pose_results, (height, width))

        pred_label = action_result.pred_label.item()
        pred_score = action_result.pred_score.numpy()  # Get scores for all classes

        predictions_list.append(pred_label)
        ground_truth_list.append(annotation["label"])
        all_pred_scores.append(pred_score)

# Calculate metrics
accuracy = accuracy_score(ground_truth_list, predictions_list)
precision = precision_score(ground_truth_list, predictions_list, average='weighted')
recall = recall_score(ground_truth_list, predictions_list, average='weighted')

# Prepare the true labels and prediction scores for mAP calculation
num_classes = len(set(ground_truth_list))
true_labels = np.zeros((len(ground_truth_list), num_classes))
for idx, label in enumerate(ground_truth_list):
    true_labels[idx, label] = 1

all_pred_scores = np.array(all_pred_scores)
mAP = average_precision_score(true_labels, all_pred_scores, average='macro')

# Print results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"mAP: {mAP}")
