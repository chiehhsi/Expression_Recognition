import numpy as np
import time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import svm
import customcnn
from sklearn.metrics import accuracy_score


x = np.load('./dataX.npy')
y = np.load('./dataY.npy')

print(x.shape, y.shape)

#x = np.expand_dims(x,-1)
batch, lenth, width = x.shape
x = x.reshape(batch, lenth*width)

#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.9, random_state=42)
X_train = x[:300]
y_train = y[:300]

X_test =x[300:400]
y_test =y[300:400]
print('X_train shape', X_train.shape)
print('y_train shape', y_train.shape)
#train the model
#model = customcnn.load_model()

start = time.time()

'''
#SVM Classifier
clf = svm.SVC(kernel = 'linear')
predicted = cross_val_predict(clf, X_train, y_train, cv=10) 
ecv = accuracy_score(predicted, y_train)
print(ecv)
'''


# get the accuracy
#print (accuracy_score(y_test, prediction))

#accuracy = clf.score(y_train, prediction)
#print(accuracy)
print('training time:', time.time()-start)


#a = customcnn.load_model()
#a.summary()