"""
This code transforms CSV annotation files to YOLO format.  
"""

import os
import pandas as pd
import json
import numpy as np
import cv2
import itertools
# import torch
import shutil

draw_labeled_images = False

df = pd.read_csv("/Users/hagedorn/Desktop/data/annotations.csv")
path_images = "/Users/hagedorn/Desktop/data/images/"
path_labels = "/Users/hagedorn/Desktop/data/labels/"
    
all_keypoint_names = set([kd['name'] for r in df.keypoinbts for l in json.loads(r) for kd in l])

keypoint_name_dict = {i: n for i, n in enumerate(all_keypoint_names)}
keypoint_num_dict = {n: i for i, n in enumerate(all_keypoint_names)}

# df = df[:10]
def get_bounding_box(segment, iw, ih):
    
    # Extract the segment coordinates
    coordinates = segment

    # Find minimum and maximum x and y coordinates
    min_x = min(coord[0] for coord in coordinates)
    max_x = max(coord[0] for coord in coordinates)
    min_y = min(coord[1] for coord in coordinates)
    max_y = max(coord[1] for coord in coordinates)

    # Compute width, height, central_x, and central_y
    width = max_x - min_x
    height = max_y - min_y
    central_x = (min_x + max_x) / 2
    central_y = (min_y + max_y) / 2

    # Normalize the values by image width and height
    x_norm = central_x / iw
    y_norm = central_y / ih
    width_norm = width / iw
    height_norm = height / ih

    return '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(0, x_norm, y_norm, width_norm, height_norm)

def normalize_keypoints(dic, iw, ih):
    positions = []
    for item in dic:
        position = item["position"]
        if position is None:
            # If key point is None, append 0 (not labeled, not visible)
            positions.append([0, 0, 0])
        else:
            # Normalize positions by the width and height of the input image
            position = [position[0] / iw, position[1] / ih, 2]  # Append 2 for labeled and visible
            positions.append(position)
    # Flatten the positions list and join the elements as a string
    return " ".join(['{:.6f}'.format(coord) for pos in positions for coord in pos])

for i, r in df.iterrows():
    fname = r['image file name']
    print("Processing ", fname)
    
    ih, iw, ic = cv2.imread(path_images + fname).shape
    fname = fname[:-4]
    output_filename = path_labels + f"{fname}.txt"
    with open(output_filename, "w") as file:
        for num_an in range(len(json.loads(r.segmentation))):
            for q in range(len(json.loads(r.segmentation)[num_an])):
                seg = json.loads(r.segmentation)[num_an][q]['segment']
                if not seg: 
                    print("Skipping ", fname)
                    continue 
                    
                bbox = get_bounding_box(seg, iw, ih)

                file.write(f"{bbox}")
                
                dict_keypoints = json.loads(r.keypoinbts)
                for kp in dict_keypoints:
                    normalized_kp = normalize_keypoints(kp, iw, ih)
                    file.write(f" {normalized_kp}\n")

                file.write("\n")  # Start new object on a new line
