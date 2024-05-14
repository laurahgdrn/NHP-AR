import os.path
from xml.dom import minidom
import os
import albumentations as A
import cv2

def rotate(image, bbox, keypoints):
    transform = A.Compose([
        A.Rotate(p=0.9),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']), keypoint_params=A.KeypointParams(format="xy"))
    class_labels=['macaque']
    transformed_rotated = transform(image=image, bboxes=bbox, keypoints=keypoints, class_labels=class_labels)
   
    rotated_boxes = transformed_rotated['bboxes']
    rotated_keypoints = transformed_rotated['keypoints']
    rotated_image = transformed_rotated['image']
    return rotated_boxes, rotated_keypoints, rotated_image

def normalize_points(points, width, height):
    normalized_points = []
    for point in points:
        x, y = point
        normalized_x = x / width
        normalized_y = y / height
        normalized_points.append((normalized_x, normalized_y))
    return normalized_points

def normalize_coordinates(xtl, ytl, xbr, ybr, width, height):
    center_x = (xtl + xbr) / (2 * width)
    center_y = (ytl + ybr) / (2 * height)
    normalized_w = (xbr - xtl) / width
    normalized_h = (ybr - ytl) / height
    return center_x, center_y, normalized_w, normalized_h

# create directory to store the text files in YOLO format 
out_dir = './labels/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# path to annotations.xml file 
file = minidom.parse('/Users/hagedorn/Downloads/annotations.xml')

images = file.getElementsByTagName('image')
input_images_folder = "/Users/hagedorn/Desktop/synched_actions/synched-playing/RA/RA_Playing_Frames_Merged"

for image in images:
    width = int(image.getAttribute('width'))
    height = int(image.getAttribute('height'))
    name = image.getAttribute('name')
    boxes = image.getElementsByTagName('box')
    points_list = image.getElementsByTagName('points')

    label_file = open(os.path.join(out_dir, name[:-4] + '.txt'), 'w')
    image_filename = os.path.splitext(name)[0] + ".jpg"
    
    image_filepath = os.path.join(input_images_folder, image_filename)
    
    image = cv2.imread(image_filepath)
    for i in range(len(boxes)):
        box = boxes[i]
        xtl = int(float(box.getAttribute('xtl')))
        ytl = int(float(box.getAttribute('ytl')))
        xbr = int(float(box.getAttribute('xbr')))
        ybr = int(float(box.getAttribute('ybr')))
        w = xbr - xtl
        h = ybr - ytl

        box_label = box.getAttribute('label')
        class_label = 'macaque'

        center_x, center_y, normalized_w, normalized_h = normalize_coordinates(xtl, ytl, xbr, ybr, width, height)

        label_file.write('{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(class_label, center_x, center_y, normalized_w, normalized_h))

        points = points_list[i].attributes['points']
        points = points.value.split(';')
        points_ = []
        for p in points:
            p = p.split(',')
            p1, p2 = p
            points_.append([int(float(p1)), int(float(p2))])

        for p_ in points_: 
           label_file.write(' {:.6f} {:.6f}'.format(p_[0], p_[1]))

        # Rotate bounding box and keypoints

        bbox = [[center_x, center_y, normalized_w, normalized_h]]
        keypoints = points_
        rotated_boxes, rotated_keypoints, _ = flip(image, bbox, keypoints)

        xtl_r, ytl_r, xbr_r, ybr_r = rotated_boxes[0]
        center_x_r, center_y_r, normalized_w_r, normalized_h_r = normalize_coordinates(xtl_r, ytl_r, xbr_r, ybr_r, width, height)

        # Write normalized rotated bounding box coordinates
        label_file.write(' {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(center_x_r, center_y_r, normalized_w_r, normalized_h_r))

        # Write normalized rotated keypoints
        normalized_rotated_keypoints = normalize_points(rotated_keypoints, width, height)
        for p_ in normalized_rotated_keypoints:
            label_file.write(' {:.6f} {:.6f}'.format(p_[0], p_[1]))

        label_file.write('\n')

    label_file.close()
