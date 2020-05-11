import numpy as np
from keras.models import model_from_json
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

predict_cnn = model_CNN.predict(X_test)
predict_scnn = model_sCNN.predict([X_test, X_sift_test])
predict_dcnn = model_dCNN.predict([X_test, X_dsift_test])
predict_vgg = model_VGG.predict(X_test)
predict_svgg = model_sVGG.predict([X_test, X_sift_test])
predict_dvgg= model_dVGG.predict([X_test, X_dsift_test])

predict_c_s = (predict_cnn+predict_scnn)/2.0
predict_c_d = (predict_cnn+predict_dcnn)/2.0
predict_v_s = (predict_vgg+predict_svgg)/2.0
predict_v_d = (predict_vgg+predict_dvgg)/2.0
predict_c_all = (predict_cnn+predict_scnn+predict_dcnn)/3.0
predict_v_all = (predict_vgg+predict_svgg+predict_dvgg)/3.0


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
    print("Accuracy on test set :" + str(accuracy_score(true_y,predict_y)*100) + "%")

accuracy(predict_cnn)
accuracy(predict_scnn)
accuracy(predict_dcnn)
accuracy(predict_vgg)
accuracy(predict_svgg)
accuracy(predict_dvgg)
accuracy(predict_c_s)
accuracy(predict_c_d)
accuracy(predict_v_s)
accuracy(predict_v_d)
accuracy(predict_c_all)
accuracy(predict_v_all)
