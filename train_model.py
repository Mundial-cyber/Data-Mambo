"""
Deep learning model for training a CancerNet model to classify breast images for cancer using the Keras framework.
It is built on Convolutional Neural Networks. It includes data augmentation, model training, and plotting results.
"""

#Running matplotlib on backend that is non-interactive for servers.
import matplotlib
matplotlib.use("Agg")

#Apply real-time data augmentation during training.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Modify learning rate during training.
from tensorflow.keras.callbacks import LearningRateScheduler
#Optimizer used for training.
from tensorflow.keras.optimizers import Adagrad
#Converts names to number using one hot encoder.
from tensorflow.keras.utils import to_categorical
#Generate classification report
from sklearn.metrics import classification_report
#Visualize using a confusion matrix.
from sklearn.metrics import confusion_matrix
#Import class CancerNet from cancernet.py in folder Cancernet
from Cancernet.cancernet import CancerNet
#import config.py from Cancernet folder
from Cancernet import config
#Import paths
from imutils import paths
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
import os

#Set hyperparameters-Number of iterations.
NUM_EPOCHS=40; INIT_LR=1e-2; BS=32

#Load dataset paths
trainPaths=list(paths.list_images(config.train_path))
lenTrain=len(trainPaths)
lenVal=len(list(paths.list_images(config.val_path)))
lenTest=len(list(paths.list_images(config.test_path)))

#Prepare labels.
trainLabels=[int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels=to_categorical(trainLabels)

#Calculate class weights.
classes = np.unique(trainLabels.argmax(axis=1))
classWeight = compute_class_weight('balanced', classes=classes, y=trainLabels.argmax(axis=1))
classWeight = dict(enumerate(classWeight))


#Data augmentation.
trainAug = ImageDataGenerator(
	rescale=1/255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.05,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")

valAug=ImageDataGenerator(rescale=1 / 255.0)

trainGen = trainAug.flow_from_directory(
	config.train_path,
	class_mode="categorical",
	target_size=(48,48),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)
valGen = valAug.flow_from_directory(
	config.val_path,
	class_mode="categorical",
	target_size=(48,48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)
testGen = valAug.flow_from_directory(
	config.test_path,
	class_mode="categorical",
	target_size=(48,48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

#Model creation.
model=CancerNet.build(width=48,height=48,depth=3,classes=2)

#Model compilation
opt=Adagrad(learning_rate=INIT_LR,decay=INIT_LR/NUM_EPOCHS) # Adaptive learning rate optimizer.
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])#model compilation using binary cross-entropy cross function.

def custom_gen(generator, class_weight):
    for x, y in generator:
        sample_weight = np.array([class_weight.get(label, 1.0) for label in y.argmax(axis=1)])
        yield x, y, sample_weight

trainGen = custom_gen(trainGen, classWeight)
valGen = custom_gen(valGen, classWeight)

#Model training.
M = model.fit(
    trainGen,
    steps_per_epoch=lenTrain // BS,
    validation_data=valGen,
    validation_steps=lenVal // BS,
    epochs=NUM_EPOCHS
)
#Validate outputs
x, y, sample_weight = next(trainGen)

print("Now evaluating the model")
#Model evaluation.
testGen.reset()
pred_indices=model.predict(testGen,steps=(lenTest//BS)+1)

pred_indices=np.argmax(pred_indices,axis=1)

#Classification report and confusion matrix.
print(classification_report(testGen.classes, pred_indices, target_names=testGen.class_indices.keys()))

cm=confusion_matrix(testGen.classes,pred_indices)
total=sum(sum(cm))
#Metrics calculation.
accuracy=(cm[0,0]+cm[1,1])/total
specificity=cm[1,1]/(cm[1,0]+cm[1,1])
sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])
print(cm)
print(f'Accuracy: {accuracy}')
print(f'Specificity: {specificity}')
print(f'Sensitivity: {sensitivity}')

N = NUM_EPOCHS

#Plotting results.
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), M.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), M.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), M.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), M.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy on the IDC Dataset")
plt.xlabel("Epoch No.")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')
