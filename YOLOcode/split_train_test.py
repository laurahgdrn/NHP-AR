#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 08:28:21 2024

@author: hagedorn
"""

import os
import pandas as pd
import torch
import shutil

# Define paths
data_dir = "/Users/hagedorn/Desktop/data/"
path_images = os.path.join(data_dir, "images")
path_labels = os.path.join(data_dir, "labels")
path_pose = os.path.join(data_dir, "pose")

# Read CSV
df = pd.read_csv(os.path.join(data_dir, "annotations.csv"))

# Move/split/copy files to train, val, test
full_dataset = [f[:-4] for f in os.listdir(path_images)]
train_size = int(0.8 * len(full_dataset))
test_size = int((len(full_dataset) - train_size) / 2)
valid_size = len(full_dataset) - train_size - test_size
train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size, valid_size])

def copy_split(dataset, set_type_path):
    for fname in dataset:
        img_path = os.path.join(path_images, fname + ".jpg")
        label_path = os.path.join(path_labels, fname + ".txt")
        try:
            if os.path.exists(img_path):
                shutil.copy(img_path, os.path.join(path_pose, "images", set_type_path, fname + ".jpg"))
            else:
                print(f"Image - {fname} not found!")

            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(path_pose, "labels", set_type_path, fname + ".txt"))
            else:
                print(f"Pose label - {fname} not found!")
        except FileNotFoundError as e:
            print(f"Error: {e}")

copy_split(train_dataset, "train/")
copy_split(test_dataset, "test/")
copy_split(valid_dataset, "valid/")

# Automate config file
conf_kpt_ls = ["# Automatically created config file from converting macaquepose_v1\n",
               f"path: ./pose/images # Dataset root directory\n",
               "train: ./train/  # Train images (relative to 'path')\n",
               "val: ./valid/  # Validation images (relative to 'path')\n",
               "test: ./test/  # Test images (relative to 'path')\n",
               "\n",
               "# Keypoints\n",
               "kpt_shape: [17, 3]  # Number of keypoints, number of dimensions (2 for x,y or 3 for x,y,visible)\n",
               "\n",
               "# Classes\n",
               "names:\n",
                   "0: macaque\n"]

with open("/Users/hagedorn/Desktop/data/macaque_pose.yaml", 'w') as f:
    for line in conf_kpt_ls:
        f.write(line)


