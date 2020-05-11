import numpy as np
import matplotlib.pyplot as plt

#histo_list = ['']

training_history=np.load('./svgg_histo.npy',allow_pickle='TRUE').item()
#print(history.history['accuracy'])
#training_history = np.load('./cnn_history.npy', allow_pickle=True)
#for i in training_history:
#	print(i)

# summarize history for accuracy
plt.plot(training_history.history['accuracy'], label='acc')
plt.plot(training_history.history['val_accuracy'], label = 'val_acc')
plt.title('vgg accuracy')
#plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.legend(['train', 'val'], loc='upper left')
#plt.savefig("vgg_training.png")
plt.show()

# summarize history for loss
plt.plot(training_history.history['loss'], label = 'loss')
#plt.plot(val_loss, label ='val_loss')
plt.plot(training_history.history['val_loss'], label ='val_loss')
plt.title('vgg loss')
#plt.ylabel('loss')
plt.xlabel('epoch')
#plt.savefig("vgg_val.png")
#plt.legend(['train', 'val'], loc='upper left')
plt.show()
#print(history)
