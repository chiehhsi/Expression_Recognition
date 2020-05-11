#!/usr/bin/env python

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers import Input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.utils.vis_utils import plot_model
from keras import Model

num_classes = 7

#SIFT layers
def Sift_layer():
	#sift features as input
	model = Sequential()
	model.add(Dense(4096, input_shape=(2048,), kernel_regularizer=l2(0.01)))
	#model.add(Dense(4096, input_shape=(2048,)))
	model.add(Dropout(0.2))

	return model

#cnn model
def custom_cnn ():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(48, 48 ,1), padding='same', activation = 'relu'))
	model.add(Conv2D(32,(3,3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides = (2,2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))

	model.add(Conv2D(64, (3, 3), activation = 'relu'))
	model.add(Conv2D(64,(3,3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides = (2,2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))

	model.add(Conv2D(128, (3, 3), activation ='relu'))
	model.add(Conv2D(128,(3,3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides = (2,2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	#Fully connected layer
	model.add(Flatten())
	model.add(Dense(2048))
	model.add(Dropout(0.5))
    
	model.add(Dense(num_classes, activation='softmax'))
	print(model.summary())
	print('Custon CNN Model created successfully...')
	#plot_model(model, to_file = 'custom.png')
	return model

#cnn layer
def cnn_layer():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(48, 48 ,1), padding='same', activation = 'relu'))
	model.add(Conv2D(32,(3,3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides = (2,2)))
	model.add(Dropout(0.1))

	model.add(Conv2D(64, (3, 3), activation = 'relu'))
	model.add(Conv2D(64,(3,3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides = (2,2)))
	model.add(Dropout(0.1))

	model.add(Conv2D(128, (3, 3), activation = 'relu'))
	model.add(Conv2D(128,(3,3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides = (2,2)))
	model.add(Dropout(0.4))
	model.add(Flatten())

	print('CNN layer created successfully...')
	return model

#fully-connected layer
def FC_layer(mergemodel):
	fc = Dense(2048, activation='relu')(mergemodel)
	fc = Dropout(0.5)(fc)
	fc = Dense(num_classes, activation = 'softmax')(fc)
	print('Fully-connected Layer created successfully...')
	return fc

'''
def VGG():
	input_tensor = Input(shape=(48,48,3))
	model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor, classes=7)
	x = model.output
	x = Flatten()(x)
	x = Dense(4096, activation = 'relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(4096, activation = 'relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(7, activation = 'softmax')(x)
	model = Model(model.input, x )
	plot_model(model, to_file ='VGG.png')

	return model
'''


#a = custom_cnn()
#a.summary
#print(a.load_weights('cnn_best_model_1.hdf5'))





