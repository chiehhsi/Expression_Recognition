import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
#%matplotlib inline


print('Start data preprocessing...')
data = pd.read_csv("fer2013.csv", delimiter=',')

#getting features and labels 
d_X = data['pixels'].values
dataY = data['emotion'].values

lenth, width = 48, 48

dataX = []
for seq in d_X:
    x = [float(i) for i in seq.split(' ')]
    x = np.array(x).reshape(lenth, width)
    dataX.append(x)

dataX = np.array(dataX)

'''
#print 8 samples figures
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
for i in range(0, 8):
    plt.subplot(2, 4, i+1)
    plt.imshow(dataX[i], interpolation='none', cmap='gray')
    plt.title(emotion_dict[dataY[i]])
plt.savefig('samples.png')
'''

print('saving data to dataX.npy, dataY.npy...')
if(len(dataX)==0 or len(dataY)==0):
    raise AssertionError('There is no images inside!')
np.save('dataX', dataX)
np.save('dataY', dataY)

print('data features shape:', dataX.shape)
print('data labels shape:', dataY.shape)