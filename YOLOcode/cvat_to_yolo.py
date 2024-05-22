#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CThis code transforms CVAT annotation files to YOLO format.  

"""
import os.path
from xml.dom import minidom
import os

# path to annotations.xml file 
file = minidom.parse('/Users/hagedorn/Desktop/YOLO/annotations_merged_new.xml')

def normalize_points(points, width, height):
    normalized_points = []
    for point in points:
        x, y = point
        normalized_x = x / width
        normalized_y = y / height
        normalized_points.append((normalized_x, normalized_y))
    return normalized_points

def normalize_coordinates(xtl, ytl, xbr, ybr, width, height):
    # Calculate center coordinates
    center_x = (xtl + xbr) / (2 * width)
    center_y = (ytl + ybr) / (2 * height)
    # Calculate width and height
    normalized_w = (xbr - xtl) / width
    normalized_h = (ybr - ytl) / height
    return center_x, center_y, normalized_w, normalized_h

# create directory to store the text files in YOLO format 

out_dir = './YOLO/new_labels/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

images = file.getElementsByTagName('image')

for image in images:
    width = int(image.getAttribute('width'))
    height = int(image.getAttribute('height'))
    name = image.getAttribute('name')
    boxes = image.getElementsByTagName('box')
    points_list = image.getElementsByTagName('points')

    label_file = open(os.path.join(out_dir, name[:-4] + '.txt'), 'w')

    for i in range(len(boxes)):
        box = boxes[i]
        xtl = int(float(box.getAttribute('xtl')))
        ytl = int(float(box.getAttribute('ytl')))
        xbr = int(float(box.getAttribute('xbr')))
        ybr = int(float(box.getAttribute('ybr')))
        w = xbr - xtl
        h = ybr - ytl

        # Get the class label from the "label" attribute of the "box" element
        # class_label = box.getAttribute('label')

        class_label = 0 # ignore position for now 
        
        center_x, center_y, normalized_w, normalized_h = normalize_coordinates(xtl, ytl, xbr, ybr, width, height)

        # Write the class label and normalized coordinates to the file
        label_file.write('{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(class_label, center_x, center_y, normalized_w, normalized_h))
       
        points = points_list[i].attributes['points']
        points = points.value.split(';')
        points_ = []
        for p in points:
            p = p.split(',')
            p1, p2 = p
            points_.append([int(float(p1)), int(float(p2))])

        normalized_points = normalize_points(points_, width, height)

        for p_ in normalized_points:
            label_file.write(' {:.6f} {:.6f}'.format(p_[0], p_[1]))

        label_file.write('\n')

    label_file.close()


