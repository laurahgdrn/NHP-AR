"""
This code extracts bounding box coordinates from xml files and converts them to YOLO format.  
"""

import os
import xml.etree.ElementTree as ET

def normalize_coordinates(xtl, ytl, xbr, ybr, width, height):
    # Calculate center coordinates
    center_x = (xtl + xbr) / (2 * width)
    center_y = (ytl + ybr) / (2 * height)
    # Calculate width and height
    normalized_w = (xbr - xtl) / width
    normalized_h = (ybr - ytl) / height
    return center_x, center_y, normalized_w, normalized_h

# Parse the XML file
tree = ET.parse('annotations_merged_new.xml')
root = tree.getroot()

output_folder = "detect-yolo-1bbox/"

# Iterate through images
for image in root.findall('./image'):
    # Extract image name
    image_name = image.get('name')
    # Extract image width and height
    image_width = float(image.get('width'))
    image_height = float(image.get('height'))
    # Initialize list to store bounding boxes and labels
    bounding_boxes = []

    # Iterate through boxes in the image
    for box in image.findall('./box'):
        label = box.get('label')
        # Check if the label is "grooming" or "playing"
        if label in ['grooming', 'playing']:
            # Extract coordinates of the box
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            # Normalize coordinates
            center_x, center_y, normalized_w, normalized_h = normalize_coordinates(xtl, ytl, xbr, ybr, image_width, image_height)
            # Convert label to YOLO format (0 for "grooming", 1 for "playing")
            yolo_label = 0 if label == 'grooming' else 1
            # Add bounding box info to the list
            bounding_boxes.append((yolo_label, center_x, center_y, normalized_w, normalized_h))

    # Create a text file for the image only if there are bounding boxes
    if bounding_boxes:
        output_file = image_name.replace('.jpg', '.txt')
        output_path = os.path.join(output_folder, output_file)
        with open(output_path, 'w') as f:
            # Write bounding boxes and labels to the file in YOLO format
            for box_info in bounding_boxes:
                f.write(f"{box_info[0]} {box_info[1]} {box_info[2]} {box_info[3]} {box_info[4]}\n")

        print(f"Bounding boxes for {image_name} written to {output_file}")
