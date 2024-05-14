import os
import cv2
from ultralytics import YOLO 

# Define the fixed size crop region (width, height)
crop_width = 500
crop_height = 500

model_path = "/Users/hagedorn/runs/pose/train3/weights/best.pt"
model = YOLO(model_path)

# Function to calculate union of two bounding boxes
def find_union(box1, box2):
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return x1, y1, x2, y2  

# Function to crop a video and save the cropped video
def crop_and_save_video(video_path, output_directory):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create a VideoWriter object to save the cropped video
    output_path = os.path.join(output_directory, f"{filename}_cropped.mp4")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))
    
    # Loop through the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame)

        boxes = results[0].boxes.xyxy.tolist()
        boxes_list = []  # List to store bounding box coordinates

        for box in boxes: 
            boxes_list.append(box)
        
        if len(boxes_list) <= 0: 
            pass 
        else:
            # Calculate union of bounding boxes
            if len(boxes_list) == 1:
                x1, y1, x2, y2 = boxes[0]
                # x2, y2 = x1 + w, y1 + h
            else:
                x1, y1, x2, y2 = find_union(boxes_list[0], boxes_list[1])
            
            # Calculate the crop region around the center of the bounding box union
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            x1_crop = max(0, center_x - crop_width // 2)
            y1_crop = max(0, center_y - crop_height // 2)
            x2_crop = min(frame.shape[1], center_x + crop_width // 2)
            y2_crop = min(frame.shape[0], center_y + crop_height // 2)
            
            # Crop the frame around the fixed area
            cropped_frame = frame[int(y1_crop):int(y2_crop), int(x1_crop):int(x2_crop)]
            
            # Write the cropped frame to the output video
            out.write(cropped_frame)

    # Release video capture and close the output video writer
    cap.release()
    out.release()

# Directory containing the videos
videos_directory = "/Users/hagedorn/Desktop/synched_actions/TSN/Merged/"

# Create a directory to store cropped videos if it doesn't exist
output_directory = "/Users/hagedorn/Desktop/synched_actions/TSN/Cropped"
os.makedirs(output_directory, exist_ok=True)

# Iterate through the videos in the directory
for filename in os.listdir(videos_directory):
    if filename.endswith(".mp4"):
        video_path = os.path.join(videos_directory, filename)
        crop_and_save_video(video_path, output_directory)
        print(f"Video saved to {output_directory}/{filename}_cropped.mp4")
