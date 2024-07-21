import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(images_folder, labels_folder, output_folder, val_size=0.15, test_size=0.15, random_state=42):
    # Create output folders if they don't exist
    for split in ['train', 'val', 'test']:
        split_images_folder = os.path.join(output_folder, 'images', split)
        split_labels_folder = os.path.join(output_folder, 'labels', split)
        os.makedirs(split_images_folder, exist_ok=True)
        os.makedirs(split_labels_folder, exist_ok=True)

    # Get a list of all label files in the labels folder
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]

    # Extract corresponding image filenames from label filenames
    image_files = [os.path.splitext(label)[0] + '.jpg' for label in label_files]

    # Split the data into train, validation, and test sets
    train_images, test_images, _, _ = train_test_split(image_files, image_files, test_size=test_size, random_state=random_state)
    train_images, val_images, _, _ = train_test_split(train_images, train_images, test_size=val_size/(1-test_size), random_state=random_state)

    # Copy images and labels to the corresponding folders
    for split, images in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
        for image in images:
            # Get corresponding label file
            label_file = os.path.splitext(image)[0] + '.txt'
            src_label_path = os.path.join(labels_folder, label_file)
            dest_label_path = os.path.join(output_folder, 'labels', split, label_file)

            # Copy label file
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dest_label_path)

                # Copy corresponding image file
                src_image_path = os.path.join(images_folder, image)
                dest_image_path = os.path.join(output_folder, 'images', split, image)
                shutil.copy(src_image_path, dest_image_path)

if __name__ == "__main__":
    images_folder = "/Users/hagedorn/Desktop/new_model/ALLIMAGES"  # Update this with the path to your images folder
    labels_folder = "/Users/hagedorn/Desktop/new_model/ALLLABELS"  # Update this with the path to your labels folder
    output_folder = "/Users/hagedorn/Desktop/new_model/split_data"  # Update this with the desired output path

    split_data(images_folder, labels_folder, output_folder)
