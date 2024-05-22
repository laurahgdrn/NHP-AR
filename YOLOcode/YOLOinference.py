from ultralytics import YOLO
import os 
import cv2

model = YOLO("/Users/hagedorn/runs/pose/train/weights/best.pt") 

video_path = "/Users/hagedorn/Desktop/YOLO/output_video.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detected_frames = 0  # Counter for frames with detections
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter("output_detected_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    print(f"Processing {os.path.basename(video_path)}")
    results = model.predict(frame)
    num = len(results[0].boxes.xyxy.tolist()) 
    if num >= 2: 
        print("Two macaques detected!")
        out.write(frame)
        detected_frames += 1

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Detected frames: {detected_frames}/{total_frames}")
print("Frames with two macaques detected saved successfully.")
