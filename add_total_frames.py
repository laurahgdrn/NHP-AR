import cv2

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

# Path to your annotation file
annotation_file = '/Users/hagedorn/mmaction2/tools/data/macaques/val_labels.txt'
new_annotation_file = '/Users/hagedorn/mmaction2/tools/data/macaques/val_labels_new.txt'

with open(annotation_file, 'r') as f:
    lines = f.readlines()

with open(new_annotation_file, 'w') as f:
    for line in lines:
        video_name, label = line.strip().split()
        video_path = f'/Users/hagedorn/mmaction2/tools/data/macaques/val/{video_name}'
        total_frames = count_frames(video_path)
        f.write(f'{video_name} {total_frames} {label}\n')
