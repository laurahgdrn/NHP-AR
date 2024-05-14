import xml.etree.ElementTree as ET
import os

def extract_data(xml_file, output_dir):
    with open(xml_file, 'rt') as f:
        tree = ET.parse(f)
    
    for image in tree.findall('.//image'):
        image_id = image.get('id')
        image_name = image.get('name')
        output_file = os.path.join(output_dir, f"{image_name.split('.')[0]}.txt")
        
        with open(output_file, 'w') as out_file:

            boxes = image.findall('.//box')
            points = image.findall('.//points')
            for box, point in zip(boxes, points):
                label = box.get('label')
                xtl, ytl, xbr, ybr = box.get('xtl'), box.get('ytl'), box.get('xbr'), box.get('ybr')
                keypoints = point.get('points')
                kps_str1 = keypoints.replace(",", " ")
                kps_str2 = kps_str1.replace(";", " ")
                out_file.write(f"{0} {xtl} {ytl} {xbr} {ybr} {kps_str2}\n")

            out_file.write('\n')


# Example usage
xml_file = '/Users/hagedorn/Desktop/YOLO/annotations_merged_new.xml'  # Replace 'data.xml' with the path to your XML file
output_dir = '/Users/hagedorn/Desktop/YOLO/output_files/'  # Replace 'output.txt' with the desired output file path
extract_data(xml_file, output_dir)
