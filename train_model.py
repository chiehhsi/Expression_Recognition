import numpy as np
import time, os, argparse
import sklearn
import model, model_vgg, feature_extraction
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

#argparse
parser = argparse.ArgumentParser()

parser.add_argument('--model', type = str, 
	choices = ['cnn', 'scnn', 'dcnn','vgg','svgg', 'dvgg'],
	help = 'Initial Model Type', default = 'cnn')
parser.add_argument('--epochs', type = int, 
	help = 'Number of epochs to run', default = 100)
parser.add_argument('--batchsize', type = int,
	help = 'Number of images to process in a batch', default = 100)

args = parser.parse_args()

model_type = args.model #cnn
epochs = args.epochs #100
batch_size = args.batchsize #100
fileName = args.model #cnn
num_classes = 7
width, height = 48, 48

print('Start Training Model on '+ model_type+'...')

#load data
x = np.load('./dataX.npy')
y = np.load('./dataY.npy')

print('DataSize', x.shape, y.shape) #(35887,48,48), (35887,)


xc = np.expand_dims(x,-1) #(48,48,1)
n_values = np.max(y) + 1
yc = np.eye(n_values)[y] #(_,1)

X_train, X_test, y_train, y_test = train_test_split(xc, yc, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state= 40)
# X_train, y_train : (22967, 48, 48, 1), (22967, 7)
# X_val, y_val : (5742, 48, 48, 1), (5742, 7)
# X_test, y_test : (7178, 48, 48, 1), (7178, 7)

#construct the training image generator for data augmentation
data_generator = ImageDataGenerator(featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

#Save model results
def save_model(model, history):
	np.save('Result/'+fileName+'_histo', history)
	print(fileName+'_histo.npy has been saved!')
	#serialize model to json
	model_json = model.to_json()
	with open('Result/'+fileName+'.json', 'w') as file:
		file.write(model_json)
	print(fileName+'.json has been saved!')
	#model.save_weights(fileName+'.h5')
	print('Mode saved!')

def SIFT_data_split(X_sift):
	X_sift_train, X_sift_test = train_test_split(X_sift, test_size=0.2, random_state=42)
	X_sift_train, X_sift_val = train_test_split(X_sift_train, test_size = 0.2, random_state= 40)
	#(22976, 2048), (5472, 2048), (7178, 2048)
	return X_sift_train, X_sift_val, X_sift_test


def load_sift():
	if not os.path.exists('Result/sift_histogram.npy'):
		X_sift = feature_extraction.Sift_extraction(x)
	else:
		print('sift histogram exist!')
		X_sift = np.load('Result/sift_histogram.npy')
	return X_sift

def load_dsift():
	if not os.path.exists('Result/d_sift.npy'):
		X_dsift = feature_extraction.DSift_extraction(x)
	else:
		print('dsift descriptors exist!')
		X_dsift = np.load('Result/d_sift.npy')
	return X_dsift


#train the model--------------------------------------

#CNN
if (model_type == 'cnn'):
	model = model.custom_cnn()
	data_input = X_train
	val_data = (X_val, y_val)

#SIFT+CNN
elif(model_type == 'scnn'):
	X_sift = load_sift()
	X_sift_train, X_sift_val, X_sift_test = SIFT_data_split(X_sift)

	sift = model.Sift_layer()
	CNN = model.cnn_layer()

	MergeModel = concatenate([sift.output, CNN.output])
	fc = model.FC_layer(MergeModel)
	model = Model(inputs=[CNN.input, sift.input], outputs = fc)

	data_input = [X_train,X_sift_train]
	val_data = ([X_val,X_sift_val],y_val)


#DSIFT+CNN
elif(model_type == 'dcnn'):
	X_dsift = load_dsift()
	X_dsift_train, X_dsift_val, X_dsift_test = SIFT_data_split(X_dsift)

	dsift = model.Sift_layer()
	CNN = model.cnn_layer()

	MergeModel = concatenate([dsift.output, CNN.output])
	fc = model.FC_layer(MergeModel)
	model = Model(inputs=[CNN.input, dsift.input], outputs = fc)

	data_input = [X_train,X_dsift_train]
	val_data = ([X_val,X_dsift_val],y_val)

#VGG16
elif(model_type =='vgg'):
	model = model_vgg.VGG_16()
	data_input = X_train
	val_data = (X_val, y_val)

#SIFT+VGG16
elif(model_type == 'svgg'):

	X_sift = load_sift()
	X_sift_train, X_sift_val, X_sift_test = SIFT_data_split(X_sift)

	sift = model.Sift_layer()
	VGG = model_vgg.VGG_layer()

	MergeModel = concatenate([sift.output, VGG.output])
	fc = model.FC_layer(MergeModel)
	model = Model(inputs=[VGG.input, sift.input], outputs = fc)	

	data_input = [X_train,X_sift_train]
	val_data = ([X_val,X_sift_val],y_val)

elif(model_type == 'dvgg'):
	X_dsift = load_dsift()
	X_dsift_train, X_dsift_val, X_dsift_test = SIFT_data_split(X_dsift)

	dsift = model.Sift_layer()
	VGG = model_vgg.VGG_layer()

	MergeModel = concatenate([dsift.output, VGG.output])
	fc = model.FC_layer(MergeModel)
	model = Model(inputs=[VGG.input, dsift.input], outputs = fc)

	data_input = [X_train,X_dsift_train]
	val_data = ([X_val,X_dsift_val],y_val)



#plot_model(model, to_file = fileName+'.png')
checkpoint = ModelCheckpoint(fileName+"_model.hdf5", monitor='val_accuracy', 
	verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=100,mode='max')
callbacks_list = [checkpoint, early_stop]

model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])

start = time.time()      
        
#dsift lr = 0.0001 with epochs = 150
training_history = model.fit_generator(data_generator.flow(data_input,y_train,
                batch_size=batch_size),
                steps_per_epoch= len(y_train)/ batch_size,
                epochs = epochs,
                verbose = 1,
                callbacks = callbacks_list,
                validation_data = val_data,
                shuffle = True
    )
print('training time:', time.time()-start)
save_model(model, training_history)



#cnn_json = cnn.to_json()
#with open("cnn_model_3.json", 'w') as file:
#	file.write(cnn_json)
#np.save('cnn_history_3', training_history)

'''
vgg16 = vgg.VGG_16()
vgg16.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint("cnn_best_model.hdf5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)
cnn.fit_generator(data_generator.flow(X_train, y_train, batch_size= batch_size), \
#					steps_per_epoch= len(y_train)/batch_size, epochs = epochs, shuffle = True)
training_history = vgg16.fit(X_train,y_train, batch_size= batch_size, validation_data=(X_val, y_val),
		 verbose=1, epochs = epochs, callbacks=[checkpoint])
vgg_json = vgg.to_json()
with open("vgg_model.json", 'w') as file:
	file.write(vgg_json)
np.save('vgg16_history', training_history)



# summarize history for accuracy
plt.plot(training_history.history['accuracy'])
plt.plot(training_history.history['val_accuracy'])
plt.title('vgg16 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("vgg_training_acc.png")
plt.show()

# summarize history for loss
plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])
plt.title('vgg16 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig("vgg_training_loss.png")
plt.legend(['train', 'val'], loc='upper left')
plt.show()
'''


