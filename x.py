import os
import random
from shutil import copyfile

# Path to the directory containing your images
dataset_path = "D:\Semester 7\ML\Project\dataset\False"

# Path to the directory where you want to create the train and valid subdirectories
output_path = "D:/Semester 7/ML/Project/dataset"

# Create train and valid directories
train_path = os.path.join(output_path, "1")
valid_path = os.path.join(output_path, "2")

os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)

# List all image files in the dataset directory
all_files = [file for file in os.listdir(dataset_path) if file.endswith(".jpg")]

# Shuffle the list of files to randomize the split
random.shuffle(all_files)

# Determine the index to split the list into train and valid sets (80:20 ratio)
split_index = int(0.8 * len(all_files))

# Assign files to train and valid sets
train_files = all_files[:split_index]
valid_files = all_files[split_index:]

# Copy files to train directory
for file in train_files:
    src_path = os.path.join(dataset_path, file)
    dst_path = os.path.join(train_path, file)
    copyfile(src_path, dst_path)

# Copy files to valid directory
for file in valid_files:
    src_path = os.path.join(dataset_path, file)
    dst_path = os.path.join(valid_path, file)
    copyfile(src_path, dst_path)
