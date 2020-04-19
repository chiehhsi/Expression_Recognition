#!/usr/bin/env python

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Flatten, Dropout, Dense

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.utils.vis_utils import plot_model


def load_model ():

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape = (48, 48, 1), activation = 'relu'))
	#model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(32,(3,3), activation='relu'))
	#model.add(BatchNormalization(axis=-1))
	model.add(MaxPooling2D((2,2), strides = (2,2)))
	model.add(Dropout(0.1))

	model.add(Conv2D(64, (3, 3), activation = 'relu'))
	#model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(64,(3,3), activation='relu'))
	#model.add(BatchNormalization(axis=-1))
	model.add(MaxPooling2D((2,2), strides = (2,2)))
	model.add(Dropout(0.1))

	model.add(Conv2D(128, (3, 3), activation = 'relu'))
	#model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(128,(3,3), activation='relu'))
	#model.add(BatchNormalization(axis=-1))
	model.add(MaxPooling2D((2,2), strides = (2,2)))
	model.add(Dropout(0.4))

	#Fully connected layer
	model.add(Flatten())
	model.add(Dense(2048))
#input shape (depth, length, width)
	model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
	plot_model(model, to_file = 'customcnn.png')

	return model

def VGG():
	model = VGG16()
	plot_model(model, to_file ='VGG.png')

	return model


#a = VGG()
#a.summary()





