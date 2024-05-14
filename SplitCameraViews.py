import os
import shutil

# Define the directory containing all the videos
directory = "/Users/hagedorn/Desktop/YOLO/macaques_data/all_videos"

# Define the directories for LA and RA videos
la_folder = "./LA"
ra_folder = "./RA"

# Create the folders if they don't exist
os.makedirs(la_folder, exist_ok=True)
os.makedirs(ra_folder, exist_ok=True)

# Iterate through the files in the directory
for filename in os.listdir(directory):
    print(f"Processing {filename}")
    # Check if the filename starts with "la"
    if filename.lower().startswith("la"):
        # Move the file to the LA folder
        shutil.move(os.path.join(directory, filename), os.path.join(la_folder, filename))
    else:
        # Move the file to the RA folder
        shutil.move(os.path.join(directory, filename), os.path.join(ra_folder, filename))

print("Files have been moved to their respective folders.")
