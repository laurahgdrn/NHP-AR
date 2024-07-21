#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:35:12 2024

@author: hagedorn
"""

import os
import xml.etree.ElementTree as ET

def merge_xml_files(folder_path, output_file):
    merged_tree = None

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            file_path = os.path.join(folder_path, filename)
            
            # Parse XML file
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Merge XML trees
            if merged_tree is None:
                merged_tree = root
            else:
                for child in root:
                    merged_tree.append(child)

    # Write merged tree to a new file
    if merged_tree is not None:
        merged_tree = ET.ElementTree(merged_tree)
        merged_tree.write(output_file, encoding='utf-8', xml_declaration=True)

# Provide folder path containing XML files and output file path
folder_path = '/Users/hagedorn/Desktop/YOLO/new_annotations'
output_file = 'annotations_merged_new.xml'

# Merge XML files
merge_xml_files(folder_path, output_file)
