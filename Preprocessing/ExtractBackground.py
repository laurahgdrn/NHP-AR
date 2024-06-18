import cv2

def extract_frame(video_path, time_str, output_path):
    # Convert time in the format HH:MM:SS or MM:SS to seconds
    time_parts = time_str.split(':')
    if len(time_parts) == 2:
        minutes, seconds = map(int, time_parts)
        total_seconds = minutes * 60 + seconds
    elif len(time_parts) == 3:
        hours, minutes, seconds = map(int, time_parts)
        total_seconds = hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("Time format should be MM:SS or HH:MM:SS")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the frame number to extract
    frame_number = int(total_seconds * fps)

    # Set the video to the frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame at {time_str}")
        return

    # Save the frame as an image
    cv2.imwrite(output_path, frame)
    print(f"Frame at {time_str} extracted and saved to {output_path}")

    # Release the video capture
    cap.release()

# Example usage
video_path = "J1 gk4 la-20240402-110000.mp4"
time_str = "31:04"  # MM:SS format
output_path = "background.png"
extract_frame(video_path, time_str, output_path)
