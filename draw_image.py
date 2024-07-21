#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:57:45 2024

@author: hagedorn
"""

import cv2

def draw_boxes(image, labels_file):
    with open(labels_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().split()
        class_id = int(line[0])
        center_x, center_y, width, height = map(float, line[1:5])
        keypoints = list(map(float, line[5:]))  # Extract keypoints
        image_h, image_w, _ = image.shape

        # Convert from YOLO format to pixel coordinates
        x = int((center_x - width/2) * image_w)
        y = int((center_y - height/2) * image_h)
        w = int(width * image_w)
        h = int(height * image_h)

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw keypoints
        for i in range(0, len(keypoints), 2):
            kp_x = int(keypoints[i] * image_w)
            kp_y = int(keypoints[i + 1] * image_h)
            cv2.circle(image, (kp_x, kp_y), 3, (0, 0, 255), -1)

    return image

# Load image
image_path = "/Users/hagedorn/Desktop/data/images/0257df98b4bddcbf.jpg"
image = cv2.imread(image_path)

# Draw bounding boxes and keypoints
labeled_image = draw_boxes(image.copy(), "/Users/hagedorn/Desktop/data/labels_test/0257df98b4bddcbf.txt")

output_image_path = "/Users/hagedorn/Desktop/data/labeled_images/0257df98b4bddcbf_labeled.jpg"
cv2.imwrite(output_image_path, labeled_image)

