import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K

class SmallVggNet:
	@staticmethod
	def build(width, height, depth, classes):
		#Initialize the model
		#Set the input shape to "channels last" and set the dimensions
		model = Sequential()
		inputShape = (width, height, depth)
		channelsDimension = -1

		#If using channels first update input shape and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			channelsDimension = 1

		#Convolution -> relu -> pooling
		model.add(Conv2D(32, (3,3), padding='same', input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelsDimension))
		model.add(MaxPooling2D(pool_size=(3,3)))
		model.add(Dropout(0.25))

		#(convolution -> relu) *2 -> pool
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelsDimension))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelsDimension))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		#(convolution -> relu) *2 -> pool
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelsDimension))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelsDimension))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		#first and only set of fc -> relu layers
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		#Softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		#Return constructed network architecture
		return model