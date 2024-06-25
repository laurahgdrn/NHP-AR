# inference 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, average_precision_score
from mmaction.apis import inference_recognizer, init_recognizer
import os
import numpy as np 

def get_class(file): 
    file = os.path.basename(file)
    if file[3:6] == 'gro': 
        return 0 
    else: 
        return 1 
    
config_path = '/Users/hagedorn/mmaction2/mmaction2/work_dirs/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py'
checkpoint_path = '/Users/hagedorn/mmaction2/mmaction2/work_dirs/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/best_acc_top1_epoch_20.pth' # can be a local path

# build the model from a config file and a checkpoint file
model = init_recognizer(config_path, checkpoint_path, device="cpu")  # device can be 'cuda:0'
# test a single image

# Paths to the validation videos
video_dir = '/Users/hagedorn/mmaction2/tools/data/macaques/val60'
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

# Extract 3D skeleton data for validation set and perform action recognition
predictions_list = []
ground_truth_list = []
all_pred_scores = []

for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    result = inference_recognizer(model, video_path)

    ground_truth = get_class(video_file)

    pred_label = result[0][0]
    pred_score = result[0][1]

    predictions_list.append(pred_label)
    ground_truth_list.append(ground_truth)
    all_pred_scores.append(pred_score)

# Calculate metrics
accuracy = accuracy_score(ground_truth_list, predictions_list)
precision = precision_score(ground_truth_list, predictions_list, average='weighted')
recall = recall_score(ground_truth_list, predictions_list, average='weighted')

# Print results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
