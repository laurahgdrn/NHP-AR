#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:29:29 2024

@author: hagedorn
"""

import os

# Define the directory containing the files
directory = '/Users/hagedorn/Desktop/YOLO/original_images'

# Iterate over each file in the directory
for filename in os.listdir(directory):
    # Check if the item is a file
    if os.path.isfile(os.path.join(directory, filename)):
        # Extract the file extension
        name, extension = os.path.splitext(filename)
        # Rename the file by adding 'LA_' at the beginning
        new_filename = name[:-9]

        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
        print(f"Renamed {filename} to {new_filename}")
