# read video file 
# perform background subtraction 
# only save those frames where there is 'motion'

# if there is motion, (pixel difference above a certain threshold): run YOLO: are there two macaques? 

# If there are two macaques, run inference 

import cv2
from ultralytics import YOLO
import os 

# Function to decrease video size
def decrease_video_size(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def get_background_frame(background_path): 
# Read the background frame
    background_img = cv2.imread(background_path)
    if background_img is None:
        print("Error: Failed to read the background frame.")
        exit()
    background_frame = cv2.resize(background_img, (1280, 720)) 
    return background_frame

def get_output_motion_path(video_path, folder): 
    video_basename = os.path.basename(video_path)
    extension = ".mp4"
    new_video_basename = video_basename[8:26]
    output_motion = f"{new_video_basename}_extracted_motion{extension}"
    output_motion_path = os.path.join(folder, output_motion)
    return output_motion_path

def background_subtraction(video_path="", background_frame="", threshold=250, output_path=""):
    # Read the video
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Failed to open video.")
        exit()

    # Video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = 1280
    frame_height = 720
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    mf = 0
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Decrease video size
        frame = decrease_video_size(frame, 50)

        # Compute the absolute difference between the frame and the background frame
        diff = cv2.absdiff(background_frame, frame)
        _, diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Check if there is motion
        motion_detected = cv2.countNonZero(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY))

        if motion_detected > 0:
            out.write(frame)
            mf += 1 

    # Release video capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Total frames: {total_frames}. Total frames with motion: {mf}")
    print("Motion extracted video saved successfully.")

def get_output_path_detections(output_motion_path, extracted_detections_path): 

    basename = os.path.basename(output_motion_path)[:-4]
    path_detections = f"{basename}_detected.mp4"  
    output_path_detections = os.path.join(extracted_detections_path, path_detections)
    return output_path_detections

def get_detections(video_path, model, output_path_detections):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detected_frames = 0  # Counter for frames with detections
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path_detections, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

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

# Function to create video fragments
def create_video_fragments(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video properties
    out_width = frame_width
    out_height = frame_height
    out_fps = fps

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fragment_index = 0
    frame_count = 0
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count == 1:
            # Open new output video writer for the fragment
            fragment_file = os.path.join(output_folder, f"{video_path}_{fragment_index}.mp4")
            out = cv2.VideoWriter(fragment_file, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (out_width, out_height))

        # Write frame to the output video
        out.write(frame)

        # Check if 5 seconds have elapsed
        if frame_count == out_fps * 5:
            # Check if the fragment has at least 60 frames
            if frame_count >= 60:
                print(f"Fragment {fragment_index}: {frame_count} frames saved.")
                fragment_index += 1
            else:
                # Delete the fragment file if it doesn't meet the frame count requirement
                os.remove(fragment_file)
                print(f"Fragment {fragment_index}: Discarded due to insufficient frames.")

            # Reset frame count and close the output video writer
            frame_count = 0
            out.release()

    # Release video capture
    cap.release()
    cv2.destroyAllWindows()

# Example usage
# video_path = "/Users/hagedorn/Desktop/YOLO/output_detected_video.mp4"
# output_folder = "/Users/hagedorn/Desktop/YOLO/video_fragments"
# create_video_fragments(video_path, output_folder)

# if __name__ == "__main__": 
#     folder =  "/Users/hagedorn/Desktop/WholeNewVideos"
#     for video in os.listdir(folder): 
#         if video.endswith(".mp4"):
#             print(f"Processing video: {video}")

#             video_path = os.path.join(folder, video)
#             extracted_motion_path = "/Users/hagedorn/Desktop/WholeNewVideos/extracted_motion"
#             output_motion_path = get_output_motion_path(video_path, extracted_motion_path)

#             # print(f"Extracting movement... Output folder: {output_motion_path}")
#             # # # background subtraction
#             # background_path =  "/Users/hagedorn/Desktop/YOLO/backrground.jpg"
#             # background_frame = get_background_frame(background_path)
#             # background_subtraction(video_path, background_frame, 250, output_motion_path)

#             extracted_detection_path = "/Users/hagedorn/Desktop/WholeNewVideos/detections"
#             output_path_detections = get_output_path_detections(output_motion_path, extracted_detection_path)

#             print(f"Extracting macaques... Output folder: {output_path_detections}")

#             # YOLO detection 
#             model = YOLO("/Users/hagedorn/runs/pose/train/weights/best.pt") 
#             get_detections(output_motion_path, model, output_path_detections)
#         # fragments_path = "/Users/hagedorn/Desktop/fragments"
#         # create_video_fragments(output_path_detections, fragments_path)


if __name__ == "__main__": 
    folder =  "/Users/hagedorn/Desktop/WholeNewVideos/extracted_motion"
    for video in os.listdir(folder): 
        if video.endswith(".mp4"):
            print(f"Processing video: {video}")

            video_path = os.path.join(folder, video)
            # extracted_motion_path = "/Users/hagedorn/Desktop/WholeNewVideos/extracted_motion"
            # output_motion_path = get_output_motion_path(video_path, extracted_motion_path)

            # print(f"Extracting movement... Output folder: {output_motion_path}")
            # # # background subtraction
            # background_path =  "/Users/hagedorn/Desktop/YOLO/backrground.jpg"
            # background_frame = get_background_frame(background_path)
            # background_subtraction(video_path, background_frame, 250, output_motion_path)

            extracted_detection_path = "/Users/hagedorn/Desktop/WholeNewVideos/detections"
            output_path_detections = get_output_path_detections(video_path, extracted_detection_path)

            print(f"Extracting macaques... Output folder: {output_path_detections}")

            # YOLO detection 
            model = YOLO("/Users/hagedorn/runs/pose/train/weights/best.pt") 
            get_detections(video_path, model, output_path_detections)
        # fragments_path = "/Users/hagedorn/Desktop/fragments"
        # create_video_fragments(output_path_detections, fragments_path)



        







