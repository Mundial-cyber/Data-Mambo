"""
This is a Convolutional Neural Network that does the following;
a) Use 3*3 CONV filters
b) Stack these filters on top of each other.
c) Perform Max Pooling
d) Use depthwise separable convolution.
Used this project to classify images using Keras with TensorFlow as the backend to create deep learning model.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, SeparableConv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adagrad

"""
Width and height-Dimensions of input image, 
depth-Number of color channels in the image, 
classes-Number of classes for classification.
"""

class CancerNet:
	@staticmethod
	def build(width,height,depth,classes):
		#Model Initialization(Models will be stacked over one another(sequential))
		model=Sequential()
		shape=(height,width,depth)
		#Default channel dimension.
		channelDim=-1

		#Used if channels dimensions comes first(eg, depth, width, height). It changes to 1.

		if K.image_data_format()=="channels_first":
			shape=(depth,height,width)
			channelDim=1
		#First Convolution.
		#Adds separable convolution layer with 32 filters, kernel size (3,3) and same padding(output will have same spatial dimensions)
		model.add(SeparableConv2D(32, (3,3), padding="same",input_shape=shape))
		#Adds ReLu activation function
		model.add(Activation("relu"))
		#Normalize output to help improve training.
		model.add(BatchNormalization(axis=channelDim))
		#Reduce spatial dimension of the output map.
		model.add(MaxPooling2D(pool_size=(2,2)))
		#Prevent overfitting by setting 25% of neurons to 0 during training.
		model.add(Dropout(0.25))

		#Second Convolution.
		#Same as the first convolution but with 64 filters and two convolutions with ReLu activation.
		model.add(SeparableConv2D(64, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(SeparableConv2D(64, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

		#Third Convolution
		#Similar to the second but with 128 filters. Three convolution layers are used with ReLu followed by Max Pooling and drop out. 
		model.add(SeparableConv2D(128, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(SeparableConv2D(128, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(SeparableConv2D(128, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

		#Fully connected layers.
		#Convert 3D feature map into 1D vector.
		model.add(Flatten())
		#Add full dense layer with 256 and activate with ReLu.
		model.add(Dense(256))
		model.add(Activation("relu"))
		#Prevent overfitting of data.
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		#Output layer where outputs are classes and correspond to output categories.
		model.add(Dense(classes))
		#'softmax' used in the output layer to make sure that the outputs sum up to 1.
		model.add(Activation("softmax"))

		return model
