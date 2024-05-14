#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:23:53 2024

@author: hagedorn
"""

import os
import cv2

# Define the folder containing the images
folder_path =  "/Users/hagedorn/Desktop/calibration/synched/synched-LA"

# Define the desired resolution
new_resolution = (1920, 1080)

# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Check if the file is an image
        # Construct the full path to the image
        image_path = os.path.join(folder_path, filename)
        
        # Load the image
        image = cv2.imread(image_path)
        
        # Resize the image
        resized_image = cv2.resize(image, new_resolution)
        
        # Save the resized image
        resized_filename = os.path.splitext(filename)[0] + "_resized.jpg"  # Add "_resized" to the filename
        resized_image_path = os.path.join(folder_path, resized_filename)
        cv2.imwrite(resized_image_path, resized_image)
        

# Iterate through the files in the directory
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Ensure it's an image file
        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath)
        if img is not None:  # Check if the image is read successfully
            # Get the size of the current image
            size = img.shape[:2]  # Extract width and height
            print(f"Image {filename} size/shape: {size}")
        else:
            print(f"Failed to read image {filename}")

