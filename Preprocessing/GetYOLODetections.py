import cv2
import os
from ultralytics import YOLO

def save_segment(output_path, fps, width, height):
    """Initialize the video writer for a new segment."""
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

def process_video_with_detections(video_path, model, output_folder, min_detections=2, max_no_detections_frames=5):
    """Process the video and segment based on object detections."""
    video_name = os.path.basename(video_path[:-4])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Failed to open video {video_path}.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    segment_index = 0
    consecutive_no_detections_frames = 0
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count} of {video_name}")

        results = model.predict(frame)
        num_detections = len(results[0].boxes.xyxy.tolist())

        if num_detections >= min_detections:
            if consecutive_no_detections_frames >= max_no_detections_frames or out is None:
                if out is not None:
                    out.release()
                segment_index += 1
                output_path = os.path.join(output_folder, f"{video_name}_segment_{segment_index}.mp4")
                out = save_segment(output_path, fps, frame_width, frame_height)
                print(f"Started new segment: {output_path}")

            consecutive_no_detections_frames = 0
            out.write(frame)
        else:
            consecutive_no_detections_frames += 1
            if consecutive_no_detections_frames < max_no_detections_frames and out is not None:
                out.write(frame)
            elif consecutive_no_detections_frames >= max_no_detections_frames and out is not None:
                out.release()
                out = None
                print(f"Ended segment at frame {frame_count}")

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print("Object detection and segmentation completed.")

if __name__ == "__main__":
    video_path = "WholeNewVideos/extracted_motion/J1 gk4 la-20240402-110000_segment_1.mp4"
    output_folder = "C:/Users/hagedorn/mmaction2/WholeNewVideos/extracted_detections"
    
    # Replace this with your YOLO model loading logic
    model = YOLO("train2-best.pt") # Load your YOLO model here

    try:
        process_video_with_detections(video_path, model, output_folder)
    except Exception as e:
        print(e)
