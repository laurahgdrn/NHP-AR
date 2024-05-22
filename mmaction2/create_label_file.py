import os

root = "/Users/hagedorn/mmaction2/tools/data/macaques/"
subfolder = os.path.join(root, "val60/")
train_file = os.path.join(root, "train_labels.txt")
val_file = os.path.join(root, "val_labels.txt")

with open(val_file, "w") as file:
    for filename in os.listdir(subfolder):
        if filename[3:6] == "gro":
            label = 0
        else:
            label = 1
        file.write(f"{filename} {label}\n")

        