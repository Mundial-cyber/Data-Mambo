""" 
Visualize the dataset in the terminal using the tree command. Make datasets directory and original directory that holds the patient's data.
mkdir datasets, mkdir original.
C:\Users\Admin\PracticePage\PracticePageProjects\Scripts\Breast-Cancer-Classification-Deep\datasets\original\tree

Create a config.py module that will declare paths for the dataset, paths for training, validation, and test
data using basepath. Also declare the training data as 80%(Including validation data as 10%)
""" 


import os

input_dataset = "datasets/original"

base_path = "datasets/idc"

#Path to where training images will be stored.
train_path = os.path.sep.join([base_path, "training"])

#Path to where validation images will be stored.
val_path = os.path.sep.join([base_path, "validation"])

#Path to where testing images will be stored.
test_path = os.path.sep.join([base_path, "testing"])

train_split = 0.8

val_split = 0.1

#Build the dataset in build_dataset.py
