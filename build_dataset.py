"""
Build another python module called build_dataset.py that includes codes that
a) Split data into training, validation, and testing using 80%(including 10% validation), and 20% testing data.
b) Use Keras to use ImageDataGenerator which will extract batches of images. Avoids making space of the dataset in memory at once.
c) Later run the py build_dataset.py in the terminal
"""
#Import libraries
from Cancernet import config
import sys
import imutils
from imutils import paths
import random, shutil, os

#Get all the image paths in the input_dataset from config directory.
originalPaths = list(paths.list_images(config.input_dataset))

#Same as random_state for reproducability
random.seed(7)

#Shuffle the images.
random.shuffle(originalPaths)

#Split the dataset to training and testing sets
index = int(len(originalPaths) * config.train_split)
trainPaths = originalPaths[:index]
testPaths = originalPaths[index:]

#Split the dataset to training and validation sets.
index = int(len(trainPaths) * config.val_split)
valPaths = trainPaths[:index]
trainPaths = trainPaths[index:]

#Organize the files.
datasets = [("training", trainPaths, config.train_path),
		   ("validation", valPaths, config.val_path),
		   ("testing", testPaths, config.test_path)
		  ]
#Iterate over this list to process each dataset type(training, validation or testing-setType)
#originalPaths where images are and basePath where images will be stored.
for (setType, originalPaths, basePath) in datasets:
	print(f"Building {setType}set")
	if not os.path.exists(basePath):
		print (f"Building directory {basePath}")
		os.makedirs(basePath)

	#Iterates over all the image paths.
	for path in originalPaths:
		#Extract the filename by splitting the file's full path at the directory separators(/ or \)
		file = path.split(os.path.sep)[-1]
		#Extract labels. Extracts second-to-last character of the filename (eg image_1.jpg)
		label = file[-5:-4]
		#Path for specific label subdirectory.
		labelPath = os.path.sep.join([basePath, label])
		#Check and create the subdirectory if the path is not present.
		if not os.path.exists(labelPath):
			print(f"Building directory {labelPath}")
			os.makedirs(labelPath)
		#Combines labelPath and file to create the full destination path of the image(Eg dataset\training\1\image_1.jpg)
		newPath = os.path.sep.join([labelPath, file])
		#Copy file from the original path to new path to preserve metadata.
		shutil.copy2(path, newPath)
		

   


