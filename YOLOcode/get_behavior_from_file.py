import os

def get_class_from_filename(file): 
    if os.path.basename(file[3:6]) == "gro": 
        return 1 
    else: 
        return 2 

path = "/Users/hagedorn/Desktop/YOLO/output_files"

for file in os.listdir(path):
    cl = get_class_from_filename(file)
    file_path = os.path.join(path, file)
    with open(file_path, 'r+') as f:
    # Read lines from the file
        lines = f.readlines()

        # Extract bounding box coordinates from the first two lines
        box1 = list(map(float, lines[0].strip().split()[1:5]))
        box2 = list(map(float, lines[1].strip().split()[1:5]))

        # Compute the combined bounding box
        min_x = min(box1[0], box1[2], box2[0], box2[2])
        min_y = min(box1[1], box1[3], box2[1], box2[3])
        max_x = max(box1[0], box1[2], box2[0], box2[2])
        max_y = max(box1[1], box1[3], box2[1], box2[3])
        print(cl, min_x, min_y, max_x, max_y)
    # # Write the updated line to the file
    # file.seek(0,2)  # Move to the end of the file
    # file.write(f"{label} {min_x} {min_y} {max_x} {max_y}\n")  
    # print(get_behavior_from_filename(file))