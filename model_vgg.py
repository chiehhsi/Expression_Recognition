from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D,Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import cv2, numpy as np
import linecache
import sys
import h5py
from keras.utils.vis_utils import plot_model	


def VGG_16(weights_path=None):
	model = Sequential()
	model.add(Conv2D(64, (3, 3), input_shape=(48, 48 ,1), padding='same', activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(7, activation='softmax'))

	#save model summary to figure
	#plot_model(model, to_file = '16.png')
	print('VGG16 model created successfully...')
    
	return model
	
def VGG_layer ():
	model = Sequential()
	model.add(Conv2D(64, (3, 3), input_shape=(48, 48 ,1), padding='same', activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Flatten())

	print('VGG layer created successfully...')
	return model

def print_weights(model):
    for layer in model.layers:
        print(layer)
        print(layer.get_weights())

#a = VGG_layer()
#a.summary()
