import numpy as np
from keras.models import model_from_json
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load data
x = np.load('./dataX.npy')
y = np.load('./dataY.npy')

xc = np.expand_dims(x,-1) #(48,48,1)
n_values = np.max(y) + 1
yc = np.eye(n_values)[y] #(_,1)

X_train, X_test, y_train, y_test = train_test_split(xc, yc, test_size=0.2, random_state=42)
X_sift = np.load('Result/sift_histogram.npy')
_, X_sift_test = train_test_split(X_sift, test_size=0.2, random_state=42)

X_dsift = np.load('Result/d_sift.npy')
_, X_dsift_test = train_test_split(X_dsift, test_size=0.2, random_state=42)

print(X_sift_test.shape, X_dsift_test.shape)

print('Loading models...')


json_model = open("Result/cnn.json", 'r')
loaded_json_model = json_model.read()
json_model.close()
model_CNN = model_from_json(loaded_json_model)
model_CNN.load_weights("Result/cnn_model.hdf5")

json_model = open("Result/scnn.json", 'r')
loaded_json_model = json_model.read()
json_model.close()
model_sCNN = model_from_json(loaded_json_model)
model_sCNN.load_weights("Result/scnn_model.hdf5")

json_model = open("Result/dcnn.json", 'r')
loaded_json_model = json_model.read()
json_model.close()
model_dCNN = model_from_json(loaded_json_model)
model_dCNN.load_weights("Result/dcnn_model.hdf5")

json_model = open("Result/vgg.json", 'r')
loaded_json_model = json_model.read()
json_model.close()
model_VGG = model_from_json(loaded_json_model)
model_VGG.load_weights("Result/vgg_model.hdf5")

json_model = open("Result/svgg.json", 'r')
loaded_json_model = json_model.read()
json_model.close()
model_sVGG = model_from_json(loaded_json_model)
model_sVGG.load_weights("Result/svgg_model.hdf5")

json_model = open("Result/dvgg.json", 'r')
loaded_json_model = json_model.read()
json_model.close()
model_dVGG = model_from_json(loaded_json_model)
model_dVGG.load_weights("Result/dvgg_model.hdf5")

#predict_cnn = model_CNN.predict(X_test)
#predict_scnn = model_sCNN.predict([X_test, X_sift_test])
#predict_dcnn = model_dCNN.predict([X_test, X_dsift_test])
predict_vgg = model_VGG.predict(X_test)
#predict_svgg = model_sVGG.predict([X_test, X_sift_test])
#predict_dvgg= model_dVGG.predict([X_test, X_dsift_test])
'''
predict_c_s = (predict_cnn+predict_scnn)/2.0
predict_c_d = (predict_cnn+predict_dcnn)/2.0
predict_v_s = (predict_vgg+predict_svgg)/2.0
predict_v_d = (predict_vgg+predict_dvgg)/2.0
predict_c_all = (predict_cnn+predict_scnn+predict_dcnn)/3.0
predict_v_all = (predict_vgg+predict_svgg+predict_dvgg)/3.0
'''

def accuracy(p):
	true_y =[]
	predict_y = []
	predicted_list = p.tolist()
	true_y_list = y_test.tolist()
	for i in range(len(y_test)):
		proba_max = max(p[i])
		current_class = max(true_y_list[i])
		class_of_Predict_Y = predicted_list[i].index(proba_max)
		class_of_True_Y = true_y_list[i].index(current_class)

		true_y.append(class_of_True_Y)
		predict_y.append(class_of_Predict_Y)
	np.save("Fer2013_True_y", true_y)
	np.save("Fer2013_Predict_y",predict_y)
	print("Accuracy on test set :" + str(accuracy_score(true_y,predict_y)*100) + "%")

def ConfusionMatrix():
	y_true = np.load('Fer2013_True_y.npy')
	y_pred = np.load('Fer2013_Predict_y.npy')
	print(len(y_true))
	print(len(y_pred))
	print(accuracy_score(y_true, y_pred))
	cm = confusion_matrix(y_true,y_pred)
	labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
	title='Confusion matrix'
	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(labels))
	plt.xticks(tick_marks, labels, rotation=45)
	plt.yticks(tick_marks, labels)
	fmt = 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j],fmt),
            horizontalalignment="center",
			color="white" if cm[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
	plt.show()

print('Predict CNN')
accuracy(predict_cnn)
ConfusionMatrix()

print('Predict SIFT_CNN')
accuracy(predict_scnn)
ConfusionMatrix()

print('Predict DSIFT_CNN')
accuracy(predict_dcnn)
ConfusionMatrix()

print('Predict CNN + SIFT_CNN')
accuracy(predict_c_s)
ConfusionMatrix()

print('Predict CNN + DSIFT_CNN')
accuracy(predict_c_d)
ConfusionMatrix()

print('Predict Aggregated CNN')
accuracy(predict_c_all)
ConfusionMatrix()

print('Predict VGG')
accuracy(predict_vgg)
ConfusionMatrix()

print('Predict SIFT_VGG')
accuracy(predict_svgg)
ConfusionMatrix()

print('Predict DSIFT_VGG')
accuracy(predict_dvgg)
ConfusionMatrix()

print('Predict VGG + SIFT_VGG')
accuracy(predict_v_s)
ConfusionMatrix()

print('Predict VGG + DSIFT_VGG')
accuracy(predict_v_d)
ConfusionMatrix()

print('Predict Aggregated VGG')
accuracy(predict_v_all)
ConfusionMatrix()
