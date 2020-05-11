import cv2
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import os.path, os
import numpy as np
import pickle

def Sift_extraction(images):
	print(images.shape)
	#extract sift descriptos of image
	desc = []
	flag = 'Image file found!'
	if not os.path.exists('pics'):
		flag = 'Writing images...'
		os.makedirs('pics')
	print(flag)

	for idx in range(len(images)):
		filepath = 'pics/'+str(idx)+'.jpg'
		if (flag =='Writing images...'):
			cv2.imwrite(filepath, images[idx])
		img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
		sift = cv2.xfeatures2d.SIFT_create()
		kp, des = sift.detectAndCompute(img, None)

		if des is not None:
			for d in des:
 				desc.append(d)
	print('SIFT descriptors ready to cluster...')

    #KMeans clusting
	k = 2048
	print('clustering..')
	#batch_size = np.size(os.listdir(img_path)) * 3
	kmeans = MiniBatchKMeans(n_clusters=k, max_iter = 200, batch_size=k*2,
		max_no_improvement=30, verbose=1).fit(desc)
	pickle.dump(kmeans, open('Result/kmeans_model.sav', 'wb'))
	print("kMeans Model Saved!")
	load_Kmeans()

#Load kMeans clustering
def load_Kmeans():
	kmeans =pickle.load(open('Result/kmeans_model.sav', 'rb'))
	print('kMeans Model loaded...')
	kmeans.verbose=False
	histo_list = []
	for idx in range(len(images)):
		print(idx)
		filepath = 'pics/'+str(idx)+'.jpg'
		img = cv2.imread(filepath)
		sift = cv2.xfeatures2d.SIFT_create()
		kp, des = sift.detectAndCompute(img, None)
		k=2048
		histo = np.zeros(k)
		nkp = np.size(kp)

		if des is not None:
			for d in des:
				idx = kmeans.predict([d])
				histo[idx] +=1/nkp
	
		histo_list.append(histo)
	print(np.array(histo_list).shape)
	np.save('Result/sift_histogram', histo_list)
	print('sift_histogram saved!')


def DSift_extraction(images):
	print(images.shape) #(35887, 48, 48)
	descriptors = []
	flag = 'Image file found!'
	if not os.path.exists('pics'):
		flag = 'Writing images...'
		os.makedirs('pics')
	print(flag)
	for idx in range(len(images)):
		#print(idx)
		filepath = 'pics/'+str(idx)+'.jpg'
		#print(images[idx])
		#print(filepath)
		if(flag == 'Writing images...'):
			if not cv2.imwrite(filepath, images[idx]):
				raise Exception('cannot write image')

		img = cv2.imread(filepath)
		gray= cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
		sift = cv2.xfeatures2d.SIFT_create()

		step_size = 12
		kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size) 
	                                    	for x in range(0, gray.shape[1], step_size)]
		#img = cv2.drawKeypoints(gray, kp, img)			
		#plt.figure(figsize=(2,2))
		#plt.imshow(img)
		#plt.show()

		dense_feat, desc = sift.compute(gray, kp)

		if desc is not None:
			desc = desc.flatten()
			descriptors.append(desc)

	np.save('Result/d_sift', descriptors) #(35887,2048)
	print('Result/d_sift saved!')

	return descriptors

initXyStep = 12
initFeatureScale = 12
initImgBound = 6

def detect(img):
	keypoints = []
	rows, cols = img.shape[:2]
	#print(rows, cols)
	for x in range(initImgBound, rows, initFeatureScale):
		for y in range(initImgBound, cols, initFeatureScale):
			keypoints.append(cv2.KeyPoint(float(x), float(y), initXyStep))
	return keypoints 

#Dense SIFT feature extraction	
def DSift(images):  #(35887, 48, 48)
	desp = []
	flag = 'Image file found!'
	if not os.path.exists('pics'):
		flag = 'Writing images...'
		os.makedirs('pics')
	print(flag)
	for idx in range(len(images)):
		filepath = 'pics/'+str(idx)+'.jpg'
		if(flag == 'Writing images...'):
			if not cv2.imwrite(filepath, images[idx]):
				raise Exception('cannot write image')

		input_image = cv2.imread(filepath)
		keypoints = detect(input_image)

		sift = cv2.xfeatures2d.SIFT_create()
		dense_feat, desc = sift.compute(input_image, keypoints)
		#print(dense_feat)
		#print(desc.shape)
		temp = []
		for d in desc:
			for i in d:
				temp.append(i)
		desp.append(temp)
	print(np.array(desp).shape)
	np.save('Result/d_sift', desp) #(35887,2048) 
	print('Result/d_sift saved!')

	return desp

	#input_image_dense = cv2.drawKeypoints(input_image_dense, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
	# Display the output image 
	#plt.figure(figsize=(2,2))
	#plt.imshow(input_image_dense)


def checkfile():
	filepath = './pics'
	if os.path.exists('pics'):
		print('file exists!')
	else:
		print('no file')
	print(os.path.getsize('pics'))
	print(os.stat('./pics').st_size)

#DSift_extraction()
#images = np.load('./dataX.npy')
#checkfile()
#DSift_extraction(images)
#Sift_extraction(images)
#load()
#DSift(images)
