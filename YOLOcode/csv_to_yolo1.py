#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:48:38 2024

@author: hagedorn
"""

import os
import pandas as pd

def normalize_points(points, width, height):
    normalized_points = []
    for point in points:
        x, y = point
        normalized_x = x / width
        normalized_y = y / height
        normalized_points.append((normalized_x, normalized_y))
    return normalized_points

def normalize_coordinates(xtl, ytl, xbr, ybr, width, height):
    # Calculate center coordinates
    center_x = (xtl + xbr) / (2 * width)
    center_y = (ytl + ybr) / (2 * height)
    # Calculate width and height
    normalized_w = (xbr - xtl) / width
    normalized_h = (ybr - ytl) / height
    return center_x, center_y, normalized_w, normalized_h

# Read CSV file
df = pd.read_csv('')

# Create directory to store the text files in YOLO format
out_dir = './labels/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Iterate over rows in the DataFrame
for index, row in df.iterrows():
    image_name = row['image file name']
    width = row['image width']
    height = row['image height']

    label_file_path = os.path.join(out_dir, image_name[:-4] + '.txt')
    label_file = open(label_file_path, 'w')

    # Assuming 'class_label', 'xtl', 'ytl', 'xbr', 'ybr' columns in the CSV
    class_label = row['class_label']
    xtl = row['xtl']
    ytl = row['ytl']
    xbr = row['xbr']
    ybr = row['ybr']

    # Normalize bounding box coordinates
    center_x, center_y, normalized_w, normalized_h = normalize_coordinates(xtl, ytl, xbr, ybr, width, height)

    # Write the class label and normalized coordinates to the file
    label_file.write('{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(class_label, center_x, center_y, normalized_w, normalized_h))

    # Assuming 'keypoint' columns in the CSV containing x,y pairs separated by a delimiter
    keypoints = [(float(x), float(y)) for x, y in zip(row['keypoint_x'].split(';'), row['keypoint_y'].split(';'))]
    normalized_keypoints = normalize_points(keypoints, width, height)

    for p in normalized_keypoints:
        label_file.write(' {:.6f} {:.6f}'.format(p[0], p[1]))

    label_file.write('\n')
    label_file.close()
