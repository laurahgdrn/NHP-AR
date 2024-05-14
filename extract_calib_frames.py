#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:02:56 2024

@author: hagedorn
"""

import cv2

def extract_frames(video_path, timestamps, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Create VideoCapture object for each timestamp
    for timestamp in timestamps:
        # Check if the timestamp is within the duration of the video
        if timestamp > duration:
            print(f"Error: Timestamp {timestamp} exceeds video duration.")
            continue
        
        # Calculate the frame number corresponding to the timestamp
        frame_number = int(timestamp * fps)
        
        # Set the video capture to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to read frame at timestamp {timestamp}.")
            continue
        
        # Write the frame to the output folder
        output_path = f"{output_folder}/RA_calib_{timestamp}.jpg"
        cv2.imwrite(output_path, frame)
        print(f"Frame extracted at timestamp {timestamp}: {output_path}")
    
    # Release the video capture object
    cap.release()

# Example usage
video_path = "/Users/hagedorn/Desktop/calibration/RA_calib_video.mp4"
output_folder = "/Users/hagedorn/Desktop/calibration/extracted_frames"

timestamps_minutes = [1.23, 1.48, 1.57, 1.59, 2.01, 2.19, 2.22, 2.47, 3.13, 3.15, 3.16, 3.19, 3.22, 3.23, 3.38, 3.42, 3.47, 3.49, 3.52, 4.16, 4.20, 4.21, 4.31, 4.26, 4.38, 4.42, 4.44, 4.45, 5.10, 5.31]

timestamps_seconds = [time * 60 for time in timestamps_minutes]

#timestamps_seconds = [83.0, 108.0, 117.0, 119.0, 121.0, 139.0, 142.0, 167.0, 193.0, 195.0, 196.0, 199.0, 202.0, 203.0, 218.0, 222.0, 227.0, 229.0, 232.0, 256.0, 260.0, 261.0, 271.0, 266.0, 278.0, 282.0, 284.0, 285.0, 310.0, 331.0]
timestamps_seconds = [80.0, 87.0, 137.0, 157.0, 160.0, 160.0, 162.0, 165.0, 178.0, 184.0, 208.0, 211.0, 215.0, 215.0, 221.0, 228.0, 231.0, 233.0, 237.0, 243.0, 265.0, 269.0, 271.0, 272.0, 275.0, 276.0, 308.0, 331.0, 333.0]

# Extract frames at specified timestamps
extract_frames(video_path, timestamps_seconds, output_folder)
