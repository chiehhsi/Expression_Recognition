# Facial Expression Recognition
### CNN model aggregated with SIFT/DSIFT descriptors

## Introduction

This project proposes several aggregated model doing facial expression recognition using Facial Expression Recognition 2013(FER2013) dataset. These models based on custom CNN model and VGG16 with SIFT and Dense SIFT feature descriptors and finally aggregated differents models to see how aggregated model performance. 

## Dataset
All the training and evaluations are done on Kaggle dataset - Facial Expression Recognition 2013 (FER2013)

Input are various 48x48 resolution grayscale images (one channel), along with label corresponding to one of seven emotions, 0 = Angry, 1 = Disgust, 2 = Fear, 3 = Happy, 4 = Sad, 5 = Surprise, 6 = Neutral. [Dataset Link](https://www.kaggle.com/deadskull7/fer2013)

## Classification Results

- Training result:

  The table shows the validation accuracy and testing accuracy

  Methods | Accuracy_val(%) | Accuracy_test(%)
  ------------ | -------------|  -----------
  CNN | 63.57 | 62.55
  SIFT_CNN | 62.6 | 61.24
  DIFT_CNN | 60.05 | 58.99
  VGG | 68.16 | 66.75
  SIFT_VGG | 67.19 | 66.66
  DIFT_VGG | 68.1 | 67.69

- Aggregated Models:

  "Aggregated" means combining original model and both models using SIFT and Dense SIFT

  Methods | Accuracy(%) 
  ------------ | -------------
  CNN + SIFT-CNN  | 64.89
  CNN + DSIFT-CNN | 63.54
  Aggregated CNN  | 65.07
  VGG + SIFT-VGG  | 69.20
  VGG + DSIFT-VGG  | 69.45
  Aggregated VGG | 70.37

- Confusion Matrix results can be found under `ConfusionMatrix` file. 

  e.g. Results of CNN model

  <img src="ConfusionMatrix/predict_cnn.png" width="500">


## Prerequisites

Make sure installed these prerequisites before running the code. The installation can be done using `pip`
```bash
- Python 3.7.6
- matplotlib 3.1.2 
- numpy 1.18.1
- Keras 2.3.1
- scikit-learn 0.22.1
- opencv-python 3.4.2.16
```

## Usages

Clone this file and run the following program using - 
```bash
$ git clone https://github.com/chiehhsi/Expression_Recognition.git
$ cd path/to/this/file
```

### Download and Preprocess Data

Download the dataset file `fer2013.csv` from [here](https://www.kaggle.com/deadskull7/fer2013) and put in the root folder of this package.

Preprocess the data and create `dataX.npy` and `dataY.npy` inside root folder
```bash
$ python3 preprocessing.py
```

### Feature Descriptor
The SIFT/ Dense SIFT descriptors can be found by either from exisiting files or building from scratch
1. Get the descriptors by scratch
  

2. Existing file

Make sure there are `sift_histogram.npy` and `d_sift.npy`

### Train Model

Launch training: 
```
$ python3 training_model.py
```
There are also optional arguments according to the needs:
- `--model` (str) : Initial Model type {cnn, scnn, dcc, vgg, svgg', dvgg}, **default = cnn**
- `--epochs` (int) : Number of epochs to run,  **default = 100**
- `--batchsize` (int) : Number of images to process in a batch,  **default = 100**

e.g `python3 training_model.py --model svgg --epochs 200`

The `model.py` defined the structure of CNN model and layers; `model_vgg.py` is for VGG16.

The program can be run in two method:

1. Using built model

If
```bash

```

2. Build from scratch
If you don't want to train the classifier from scratch, you can make the use of fertestcustom.py directly as the the repository already has fer.json (trained model) and fer.h5 (parameters) which can be used to predict emotion on any test image present in the folder. You can modify fertestcustom.py according to your requirements and use it to predict fatial emotion in any use case.

pics files about 57.4MB

### Test Models

To get the test accuracy and the confusion matrix
```bash
python3 predict.py
```

## References

Mundher Al-Shabi, ooi Ping Cheah, Tee Connie, {\it Facial Expression Recognition Using a Hybrid {CNN-SIFT}} (arXiv 1608.02833, 2016)
