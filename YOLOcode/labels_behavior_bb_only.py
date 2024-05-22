import os

WIDTH = 2560
HEIGHT = 1440

def get_class_from_filename(file): 
    if os.path.basename(file[3:6]) == "gro": 
        return 0
    else: 
        return 1
def normalize_coordinates(xtl, ytl, xbr, ybr, width, height):
    # Calculate center coordinates
    center_x = (xtl + xbr) / (2 * width)
    center_y = (ytl + ybr) / (2 * height)
    # Calculate width and height
    normalized_w = (xbr - xtl) / width
    normalized_h = (ybr - ytl) / height
    return center_x, center_y, normalized_w, normalized_h

input_path = "/Users/hagedorn/Desktop/YOLO/output_files"
output_path = "/Users/hagedorn/Desktop/YOLO/one_bbox_labels"

for file in os.listdir(input_path):
    cl = get_class_from_filename(file)
    input_file_path = os.path.join(input_path, file)
    output_file_path = os.path.join(output_path, f"{os.path.splitext(file)[0]}_original.txt")
    
    with open(input_file_path, 'r') as f:
        lines = f.readlines()
        box1 = list(map(float, lines[0].strip().split()[1:5]))
        box2 = list(map(float, lines[1].strip().split()[1:5]))
        min_x = min(box1[0], box1[2], box2[0], box2[2])
        min_y = min(box1[1], box1[3], box2[1], box2[3])
        max_x = max(box1[0], box1[2], box2[0], box2[2])
        max_y = max(box1[1], box1[3], box2[1], box2[3])
    with open(output_file_path, 'w') as output_file:
        output_file.write(f"{cl} {min_x/WIDTH} {min_y/HEIGHT} {max_x/WIDTH} {max_y/HEIGHT}\n")
