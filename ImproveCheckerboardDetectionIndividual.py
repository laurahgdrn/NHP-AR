#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 08:10:15 2024

@author: hagedorn
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def refine_corners(image, max_corners=100, quality_level=0.01, min_distance=10):
    corners = cv2.goodFeaturesToTrack(image, max_corners, quality_level, min_distance)
    corners = np.int0(corners)
    return corners

def otsu_threshold(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    return thresh

def sharpen(image): 
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]]) 
      
    # Sharpen the image 
    sharpened_image = cv2.filter2D(image, -1, kernel) 
    
    return sharpened_image

def gamma_correction(image):
    gamma = 2
    gamma_corrected = np.uint8(((image / 255.0) ** gamma) * 255)
    
    return gamma_corrected 


def erode(thresh): 
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    
    return eroded

# [ 232.8,232.9,233, 233.1, 233.2, 233.3 ] 
image_path = "/Users/hagedorn/Desktop/calibration/New_LA/LA_415.jpg"
image = cv2.imread(image_path)
filename = os.path.basename(image_path)

# Preprocess image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
gamma_corrected = gamma_correction(gray_image)
sharpened = sharpen(gamma_corrected)
eroded = erode(sharpened)
processed_image = gray_image.copy()

plt.imshow(processed_image, cmap='gray')
plt.show()

corners = refine_corners(processed_image)
# Apply cv2.findChessboardCorners()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
conv_size = (10, 10)
# cv2.imwrite(os.path.join("/Users/hagedorn/Desktop/calibration/labeled_images", "RA_calib_228.0_unlabeled.jpg"), processed_image)
if corners is not None:
    ret, corners = cv2.findChessboardCorners(processed_image, (13, 19), None)

    if ret:
        corners = cv2.cornerSubPix(processed_image, corners, conv_size, (-1, -1), criteria)
        labeled_image = cv2.drawChessboardCorners(processed_image, (13, 19), corners, ret)
        filename_without_extension, extension = os.path.splitext(filename)
        new_filename_labeled = filename_without_extension + "_labeled" + extension
        new_filename_unlabeled = filename_without_extension + "_unlabeled" + extension
        cv2.imwrite(os.path.join("/Users/hagedorn/Desktop/calibration/labeled_images", new_filename_labeled), labeled_image)
        # cv2.imwrite(os.path.join("/Users/hagedorn/Desktop/calibration/LA/new_calib_images", new_filename_unlabeled), processed_image)
        # Convert image to RGB for Matplotlib
        labeled_image_rgb = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
        
        # Plot the image using Matplotlib
        plt.imshow(labeled_image_rgb)
        plt.axis('off')
        plt.show()

print(ret)



