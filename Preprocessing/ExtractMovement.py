import cv2
import os

def decrease_frame_size(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def get_background_frame(background_path):
    background_img = cv2.imread(background_path)
    if background_img is None:
        print("Error: Failed to read the background frame.")
        exit()
    return background_img

def background_subtraction(video_path, background_frame, threshold, output_folder):

    video_name = os.path.basename(video_path[:-4])
    print(video_name)
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Failed to open video.")
        exit()

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    segment_index = 0
    consecutive_no_movement_frames = 0
    max_no_movement_frames = 3
    out = None

    width = 640
    height = 360

    background_frame = decrease_frame_size(background_frame, width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Decrease video size if needed
        frame = decrease_frame_size(frame, width, height)

        # Compute the absolute difference between the frame and the background frame
        diff = cv2.absdiff(background_frame, frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_thresh = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)

        # Check if there is significant movement
        motion_detected = cv2.countNonZero(diff_thresh) > 0
        if motion_detected:
            if consecutive_no_movement_frames >= max_no_movement_frames or out is None:
                if out is not None:
                    out.release()
                segment_index += 1
                output_path = f"{output_folder}/{video_name}_segment_{segment_index}.mp4"
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                print(f"Started new segment: {output_path}")

            consecutive_no_movement_frames = 0
            out.write(frame)
        else:
            consecutive_no_movement_frames += 1
            if consecutive_no_movement_frames < max_no_movement_frames and out is not None:
                out.write(frame)
            elif consecutive_no_movement_frames >= max_no_movement_frames and out is not None:
                out.release()
                out = None
                print(f"Ended segment at frame {frame_count}")

    # Release video capture and writer
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print("Motion extraction and segmentation completed.")

# Example usage
video_path = "WholeNewVideos/J1 gk4 la-20240402-110000.mp4"
background_path = "background.png"
output_folder = "C:/Users/hagedorn/mmaction2/WholeNewVideos/extracted_motion"
threshold = 30  # Adjust threshold value as needed

background_frame = get_background_frame(background_path)
background_subtraction(video_path, background_frame, threshold, output_folder)
