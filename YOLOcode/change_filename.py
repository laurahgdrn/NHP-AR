import os

def change_filename(filename): 
    substring_to_remove = "_extracted_frames_frame"
    # Find the index of the substring to remove
    index_to_remove = filename.index(substring_to_remove)
    # Remove the substring
    new_filename = filename[:index_to_remove] + filename[index_to_remove + len(substring_to_remove):]
    # Optionally, you can add ".txt" extension to the new filename
    new_filename = new_filename[:-4] 
    return new_filename + ".txt"

import os

def remove_original(filename):
    # Define the substring to remove
    substring_to_remove = "_original"
    # Check if the substring exists in the filename
    if substring_to_remove in filename:
        # Remove the substring
        new_filename = filename.replace(substring_to_remove, "")
        return new_filename
    else:
        return filename

directory = "/Users/hagedorn/Desktop/new_model/lara_merged_labels_kp3"
for filename in os.listdir(directory):
    if filename.endswith(".txt"):  # Make sure you're only renaming text files
        old_filepath = os.path.join(directory, filename)
        new_filename = remove_original(filename)
        new_filepath = os.path.join(directory, new_filename)
        os.rename(old_filepath, new_filepath)

