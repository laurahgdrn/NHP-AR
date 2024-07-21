import os
from PIL import Image

# Path to the directory containing PNG files
directory = '/Users/hagedorn/Desktop/YOLO/private/images'

# Iterate through the directory
for filename in os.listdir(directory):
    if filename.endswith('.png'):
        # Open the PNG file
        image = Image.open(os.path.join(directory, filename))
        
        # Convert and save as JPG
        new_filename = os.path.splitext(filename)[0] + '.jpg'
        image.save(os.path.join(directory, new_filename))
        
        # Optionally, delete or rename the original PNG file
        os.remove(os.path.join(directory, filename))
        # os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
