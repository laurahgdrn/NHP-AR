import os
from xml.dom import minidom
def change_filename(filename): 
    substring_to_remove = "_extracted_frames_frame"
    # Find the index of the substring to remove
    index_to_remove = filename.index(substring_to_remove)
    # Remove the substring
    new_filename = filename[:index_to_remove] + filename[index_to_remove + len(substring_to_remove):]
    # Optionally, you can add ".txt" extension to the new filename
    new_filename = new_filename[:-4] 
    return new_filename + ".txt"
# Create directory to store the text files in YOLO format
out_dir = './YOLO/yolo_labels_unnormalized/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Path to annotations.xml file 
file = minidom.parse('/Users/hagedorn/Desktop/YOLO/annotations/annotations_merged_new.xml')

# Get a list of all image objects in the XML
image_objects = file.getElementsByTagName('image')

# Iterate through each image object
for image_obj in image_objects:
    # Get image attributes
    filename = image_obj.getAttribute('name')
    # width = int(image_obj.getAttribute('width'))
    # height = int(image_obj.getAttribute('height'))
    new_filename = change_filename(filename)
    output_file_path = os.path.join(out_dir, new_filename)
    with open(output_file_path, 'w') as output_file:
        
        # Get box and points objects for the current image
        boxes = image_obj.getElementsByTagName('box')
        points_list = image_obj.getElementsByTagName('points')

        # Iterate through each box and points
        for box, points in zip(boxes, points_list):
            # Extract box coordinates
            xtl = int(float(box.getAttribute('xtl')))
            ytl = int(float(box.getAttribute('ytl')))
            xbr = int(float(box.getAttribute('xbr')))
            ybr = int(float(box.getAttribute('ybr')))

            # Extract points coordinates
            points_str = points.attributes['points'].value
            points_list = [list(map(float, p.split(','))) for p in points_str.split(';')]

            # Write label, box coordinates, and points to the output file
            output_file.write(f"0 {xtl} {ytl} {xbr} {ybr}")
            for point in points_list:
                output_file.write(f" {point[0]} {point[1]}")
            output_file.write("\n")
