from keras.models import model_from_json
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

print('Loading Data....')
x = np.load('./dataX.npy')
y = np.load('./dataY.npy')

print(x.shape, y.shape)
xc = np.expand_dims(x,-1)
n_values = np.max(y) + 1
yc = np.eye(n_values)[y]

X_train, X_test, y_train, y_test = train_test_split(xc, yc, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state= 40)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
X_dsift = np.load('./d_sift.npy')
X_dsift_train, X_dsift_test = train_test_split(X_dsift, test_size=0.2, random_state=42)
X_dsift_train, X_dsift_val = train_test_split(X_dsift_train, test_size = 0.2, random_state= 40)

#modify parameters
batch_size = 100
epochs = 50
input_data = [X_train,X_dsift_train]
val_data = ([X_val,X_dsift_val],y_val)

data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

#loading model
json_model = open("Result/cnn.json", 'r')
loaded_json_model = json_model.read()
json_model.close()
model = model_from_json(loaded_json_model)
model.load_weights("Result/cnn_model.hdf5")

print('Model Loaded!')
print('Resume Training...')
checkpoint = ModelCheckpoint("Result/cnn_model.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=100,mode='max')
callbacks_list = [checkpoint, early_stop]
model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])

dsift_histo = model.fit_generator(data_generator.flow(input_data,y_train,
                batch_size=batch_size),
                steps_per_epoch= len(y_train)/ batch_size,
                epochs = epochs,
                verbose = 1,
                callbacks = callbacks_list,
                validation_data = val_data,
                shuffle = True
)

model_json = model.to_json()
with open("Result/cnn.json", 'w') as file:
    file.write(model_json)
np.save('Result/cnn_histo.npy', dsift_histo)
print('cnn_histo.npy has been saved!')
print('Model saved!')