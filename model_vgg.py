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
'''
if __name__ == "__main__":
    img_filename = 'husky.jpg'
    img_label_index = 250
    
    #im = cv2.resize(cv2.imread(img_filename), (224, 224)).astype(np.float32)
    im = cv2.resize(cv2.imread(img_filename), (300, 300)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    X = np.expand_dims(im, axis=0)
    Y = np.zeros(1000)
    Y[img_label_index] = 1
    Y = np.expand_dims(Y, axis=0)
    print(X.shape)
    print(Y.shape)
        
    # load custom vgg16 and train 1 epoch
    model = VGG_16('vgg16_weights.h5')
    #print_weights(model)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')       
    model.fit(X, Y, batch_size=1, nb_epoch=1, validation_data=(X, Y))
'''
