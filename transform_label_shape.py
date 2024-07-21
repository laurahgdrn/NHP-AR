import os

def transform_label_file(input_file, output_file):
    with open(input_file, 'r') as f_in:
        with open(output_file, 'w') as f_out:
            for line in f_in:
                line = line.strip().split()  # Split the line into components
                class_index = line[0]  # Class index
                bbox = line[1:5]  # Bounding box coordinates
                keypoints = line[5:]  # Key point pairs

                # Write class index and bounding box coordinates to output file
                f_out.write(f"{class_index} {' '.join(bbox)} ")

                # Iterate over key point pairs and add "2" after each pair
                for i in range(0, len(keypoints), 2):
                    x = keypoints[i]
                    y = keypoints[i + 1]
                    f_out.write(f"{x} {y} 2 ")

                f_out.write("\n")  # Add newline after each object

def transform_labels_in_directory(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):  # Process only text files
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)

            # Transform labels and save to output file
            transform_label_file(input_file, output_file)

# Example usage

input_directory = "/Users/hagedorn/Desktop/YOLO/labels"
output_directory = "/Users/hagedorn/Desktop/YOLO/labels_kp3"

transform_labels_in_directory(input_directory, output_directory)
