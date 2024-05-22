import os
import shutil

# Define the folders for LA and RA videos
la_folder = "./LA"
ra_folder = "./RA"

# Define the train and val folders
train_folder = "./train"
val_folder = "./val"

# Function to split the files into train and validation sets
def split_files(folder, train_dest, val_dest, split_ratio=0.8):
    # Sort the files in the folder
    files = sorted(os.listdir(folder))
    total_files = len(files)
    # Calculate the number of files for train and val sets
    train_count = int(total_files * split_ratio)
    val_count = total_files - train_count
    
    # Create train and val folders if they don't exist
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest, exist_ok=True)
    
    # Copy files to train set
    for file in files[:train_count]:
        src = os.path.join(folder, file)
        dst = os.path.join(train_dest, file)
        shutil.copy(src, dst)
    
    # Copy files to val set
    for file in files[train_count:]:
        src = os.path.join(folder, file)
        dst = os.path.join(val_dest, file)
        shutil.copy(src, dst)

# Split files in LA folder
split_files(la_folder, os.path.join(train_folder, "LA"), os.path.join(val_folder, "LA"))

# Split files in RA folder
split_files(ra_folder, os.path.join(train_folder, "RA"), os.path.join(val_folder, "RA"))

print("Files have been split into train and val sets.")
