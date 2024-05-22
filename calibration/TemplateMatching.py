#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:23:53 2024

@author: hagedorn
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/Users/hagedorn/Desktop/calibration/LA/LA_calib_121.jpg', cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
img2 = img.copy()
template = cv2.imread('/Users/hagedorn/Desktop/calibration/template.jpg', cv2.IMREAD_GRAYSCALE)

def sharpen(image): 
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
      
    # Sharpen the image 
    sharpened_image = cv2.filter2D(image, -1, kernel) 
    
    return sharpened_image

def gamma_correction(image):
    gamma = 4
    gamma_corrected = np.uint8(((image / 255.0) ** gamma) * 255)
    
    return gamma_corrected 

def binary_thresh(image): 
    _, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    
    return thresh 

def erode(thresh): 
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    
    return eroded
assert template is not None, "file could not be read, check with os.path.exists()"

w, h = template.shape[::-1]

sharpened_image = sharpen(img)
gamma_corrected = gamma_correction(sharpened_image)


res = cv2.matchTemplate(img2,template,2)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img2,top_left, bottom_right, 255, 2)
plt.subplot(121),plt.imshow(res,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img2,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

plt.savefig("HQPlot.png", format="png", dpi=100)
plt.show()







# # All the 6 methods for comparison in a list
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
#             'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
# for meth in methods:
#     img = img2.copy()
#     method = eval(meth)
#     # Apply template Matching
#     res = cv2.matchTemplate(img,template,'cv.TM_CCORR')
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv2.rectangle(img,top_left, bottom_right, 255, 2)
#     plt.subplot(121),plt.imshow(res,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()