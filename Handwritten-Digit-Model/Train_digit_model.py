"""
Import the libraries and load the dataset. The dataset used here is the MNIST dataset, which has handwritten digit images in 28*28 sizes.
Keras Library will be used to train this deep learning model. A Convolutional Neural Network will be made to train the deep learning model.
In the end, a Graphic User Interface will be built so that a digit can be drawn and recognized immediately.
Keras already has mnist as one of its datasets. The dataset has a shape of (60000, 28, 28). The CNN must have four dimensions so we make it (60000,28,28,1)
epochs - One complete pass of the entire training dataset through the neural network.
I use BatchNormalization() to improve the model's performance and ensure no overfitting.
The number of epochs is set at 40 to ensure the model has time to train and is highly accurate in t.

"""

# Import all the libraries 
import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization

# Load mnist data. The mnist.load_data returns two tuples, not four individual ones.
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape)

#Reshape X and y train to fit the CNN's dimensions (60000,28,28,1)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

#Convert class vectors into binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#Normalize data into a range of 0-1
X_train = X_train.astype("float32")/255.0
X_test = X_test.astype("float32")/255.0
print("X_train.shape:", X_train.shape)
print("Train samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

#Create CNN model.
batch_size = 128
num_classes = 10
epochs = 30


model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), activation="relu", input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

#Compile the model with Adadelta optimizer (Adaptive learning rate optimizer. No need of specifying the learning rate. Works well with image or text classification)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

#Data augmentation to make sure that the data mimics real world data.
datagen = ImageDataGenerator(
	rotation_range = 10,
	width_shift_range = 0.1,
	height_shift_range = 0.1,
	zoom_range = 0.1
)
datagen.fit(X_train)
#Train the model with augmented data
hist = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
		 validation_data=(X_test, y_test),
		 epochs=epochs)
print("The model has trained successfully")

#Evaluate the model (assess how the model generalizes to unseen data)
score = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss", score[0])
print("Test Accuracy", score[1])

#Save the model
model.save("mundial_mnist.h5")
print("Saving the model as mundial_mnist.h5")