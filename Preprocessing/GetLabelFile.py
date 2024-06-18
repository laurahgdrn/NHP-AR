
import os

train_folder = "C:/Users/hagedorn/Desktop/data/macaques_data/all_videos/train/"
val_folder = "C:/Users/hagedorn/Desktop/data/macaques_data/all_videos/test/"

train_labels = "C:/Users/hagedorn/Desktop/data/macaques_data/all_videos/train_labels.txt"
val_labels = "C:/Users/hagedorn/Desktop/data/macaques_data/all_videos/test_labels.txt"

with open(val_labels, "w") as val_labels_file: 
    for file in os.listdir(val_folder): 
        if file.endswith(".mp4"):
            if file[3:6] == "gro": 
                label = 0 
            else: 
                label = 1
            # Write to train_labels
            val_labels_file.write(f"{file} {label}\n")
