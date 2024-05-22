import os
import cv2
import numpy as np

def preprocess_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)

    return enhanced_gray

def sharpen(image): 
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]]) 

    sharpened_image = cv2.filter2D(image, -1, kernel) 
    
    return sharpened_image

def gamma_correction(image):
    gamma = 2
    gamma_corrected = np.uint8(((image / 255.0) ** gamma) * 255)
    
    return gamma_corrected 

def binary_thresh(image): 
    _, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    
    return thresh 

def erode(thresh): 
    kernel = np.ones((1, 1), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    
    return eroded

image_processing = [sharpen, gamma_correction, binary_thresh, erode]

def find_checkerboard(image):
    labeled_image = None
    ret = False
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    conv_size = (10,10)
    
    #processing_functions = [lambda img: img, sharpen, gamma_correction, binary_thresh, erode]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    gamma_corrected = gamma_correction(gray_image)
    sharpened = sharpen(gamma_corrected)
    
    prc = gray_image.copy()
    ret, corners = cv2.findChessboardCorners(prc, (4, 6), None)
        
    if ret:
        corners = cv2.cornerSubPix(prc, corners, conv_size, (-1, -1), criteria)
        labeled_image = image.copy()  # Create a copy of the original image
        labeled_image = cv2.drawChessboardCorners(labeled_image, (4, 6), corners, ret)  # Draw corners on the copy)
            
    return labeled_image, prc, ret

def process_images_in_directory(directory):
    labeled_images_directory = os.path.join(directory, "labeled_images")
    os.makedirs(labeled_images_directory, exist_ok=True)
    
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            original_image = image.copy()
            # Preprocess the image
            # preprocessed_image = preprocess_image(image)

            # Try to find the checkerboard
            labeled_image, processed_image, checkerboard_found = find_checkerboard(image)

            if checkerboard_found:
                labeled_filename = os.path.splitext(filename)[0] + "_labeled.jpg"
                processed_filename = os.path.splitext(filename)[0] + "_processed.jpg"
                cv2.imwrite(os.path.join(labeled_images_directory, labeled_filename), labeled_image)
                cv2.imwrite(os.path.join(labeled_images_directory, processed_filename), processed_image)
            else:
                print(f"Checkerboard not found in {filename}")

# Define the directory containing calibration images
calibration_images_directory = "/Users/hagedorn/Desktop/calibration/synched/synched-LA"

# Process images in the directory
process_images_in_directory(calibration_images_directory)
