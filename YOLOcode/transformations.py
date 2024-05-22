#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code performs data augmentation on images and label files in YOLO format. 
"""

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random

random.seed(7)
KEYPOINT_COLOR = (0, 255, 0) # Green

def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=15):
    image = image.copy()

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    return image

def cvat_to_yolo(bbox_cvat, image_width, image_height):

    xtl, ytl, xbr, ybr = bbox_cvat

    # Calculate YOLO coordinates
    center_x = (xtl + xbr) / (2.0 * image_width)
    center_y = (ytl + ybr) / (2.0 * image_height)
    width = (xbr - xtl) / image_width
    height = (ybr - ytl) / image_height

    return [center_x, center_y, width, height]

def normalize_keypoints(keypoints, image_width, image_height):
    normalized_keypoints = [(x / image_width, y / image_height) for x, y in keypoints]
    return normalized_keypoints

def parse_object(line):
    # Split the line into space-separated values
    values = line.split()

    # Extract label, bounding box coordinates, and key points
    label = int(values[0])
    bbox = list(map(float, values[1:5]))  # Convert to float
    keypoints = [tuple(map(float, values[i:i+2])) for i in range(5, len(values), 2)]

    return label, bbox, keypoints

def flip(bbox, keypoints, image):
    transform = A.Compose([
        A.HorizontalFlip(p=1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']), keypoint_params=A.KeypointParams(format="xy"))
    class_labels=['macaque']
   
    transformed_flipped = transform(image=image, bboxes=bbox, keypoints=keypoints, class_labels=class_labels)
   
    rotated_boxes = transformed_flipped['bboxes']
    rotated_keypoints = transformed_flipped['keypoints']
    rotated_image = transformed_flipped['image']
    return rotated_boxes, rotated_keypoints, rotated_image

def rotate(bbox, keypoints, image):
    transform = A.Compose([
        A.Rotate(p=0.9),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']), keypoint_params=A.KeypointParams(format="xy"))
    class_labels=['macaque']
    transformed_rotated = transform(image=image, bboxes=bbox, keypoints=keypoints, class_labels=class_labels)
   
    rotated_boxes = transformed_rotated['bboxes']
    rotated_keypoints = transformed_rotated['keypoints']
    rotated_image = transformed_rotated['image']
    return rotated_boxes, rotated_keypoints, rotated_image

def crop(image, bbox, keypoints):
    width, height, _ = image.shape
    transform = A.Compose([
        A.RandomCrop(width-200, height-200),
    ], bbox_params=A.BboxParams(format='yolo'), keypoint_params=A.KeypointParams(format="xy"))
   
    transformed_cropped = transform(image=image, bboxes=bbox, keypoints=keypoints)
   
    rotated_boxes = transformed_cropped['bboxes']
    rotated_keypoints = transformed_cropped['keypoints']
    rotated_image = transformed_cropped['image']
    return rotated_boxes, rotated_keypoints, rotated_image

input_images_folder = "/Users/hagedorn/Desktop/synched_actions/LARA_All_Frames_Merged"
input_annotations_folder = "/Users/hagedorn/Desktop/YOLO/YOLO_labels/yolo_labels_unnormalized"

# Set the output folders for rotated images and text files
transformed_images_folder = "/Users/hagedorn/Desktop/YOLO/transformed_images"
transformed_annotations_folder = "/Users/hagedorn/Desktop/YOLO/transformed_labels"

# Create output directories if they don't exist
os.makedirs(transformed_images_folder, exist_ok=True)
os.makedirs(transformed_annotations_folder, exist_ok=True)

for filename in os.listdir(input_annotations_folder):
    if filename.endswith(".txt"):
        annotation_filepath = os.path.join(input_annotations_folder, filename)

        with open(annotation_filepath, "r") as textfile:
            content = textfile.read().strip().split('\n')
        
        image_filename = os.path.splitext(filename)[0] + ".jpg"
        image_filepath = os.path.join(input_images_folder, image_filename)

        image = cv2.imread(image_filepath)
        height, width, _ = image.shape
        
        transformed_objects = []  
        for obj in content:
            label, bbox, keypoints = parse_object(obj)
            
            yolo_bbox = cvat_to_yolo(bbox, width, height)
            
            # yolo_bbox.append(str(label))
            
            transformed_boxes, transformed_keypoints, transformed_image = flip([yolo_bbox], keypoints, image)
            transf_height, transf_width, _ = transformed_image.shape
            normalized_keypoints = normalize_keypoints(transformed_keypoints, width, height)

            transformed_objects.append({
                'label': label,
                'transformed_boxes': transformed_boxes,
                'transformed_keypoints': normalized_keypoints
            })
            
            base_filename = filename.split('.')[0]  # Get the base filename without extension
            output_image_filepath = os.path.join(transformed_images_folder, f"{base_filename}_flipped.jpg")
            cv2.imwrite(output_image_filepath, transformed_image)

        # Create a new text file to store rotated object information in the "transformed_annotations" folder
            output_txt_filepath = os.path.join(transformed_annotations_folder, f"{os.path.splitext(filename)[0]}_flipped.txt")
            with open(output_txt_filepath, "w") as output_txt:
                
                for obj_info in transformed_objects: 
                    output_txt.write(f"{obj_info['label']} {obj_info['transformed_boxes'][0][0]} {obj_info['transformed_boxes'][0][1]} {obj_info['transformed_boxes'][0][2]} {obj_info['transformed_boxes'][0][3]}")
                    for point in obj_info['transformed_keypoints']:
                        for coord in point:
                            output_txt.write(f" {coord}")
                    output_txt.write("\n")
