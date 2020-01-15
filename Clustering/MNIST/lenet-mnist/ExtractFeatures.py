# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 19:53:24 2018

@author: Ambi
"""

from pyimagesearch.cnn.networks import LeNet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.utils import np_utils
from keras.optimizers import SGD
import numpy as np
from keras import backend as K
import pandas as pd
import time
outputDir = 'output/kmeans/'
def extractFeatures():
    # grab the MNIST dataset (if this is your first time running this
    # script, the download may take a minute -- the 55MB MNIST dataset
    # will be downloaded)
    print("[INFO] downloading MNIST...")
    dataset = datasets.fetch_mldata("MNIST Original")
    
    # reshape the MNIST dataset from a flat list of 784-dim vectors, to
    # 28 x 28 pixel images, then scale the data to the range [0, 1.0]
    # and construct the training and testing splits
    data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
    data = data[:, np.newaxis, :, :]
    (trainData, testData, trainLabels, testLabels) = train_test_split(
    	data / 255.0, dataset.target.astype("int"), test_size=0.33)
    
    # transform the training and testing labels into vectors in the
    # range [0, classes] -- this generates a vector for each label,
    # where the index of the label is set to `1` and all other entries
    # to `0`; in the case of MNIST, there are 10 class labels
    trainLabels = np_utils.to_categorical(trainLabels, 10)
    testLabels = np_utils.to_categorical(testLabels, 10)
    
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.01)
    model = LeNet.build(width=28, height=28, depth=1, classes=10,
    	weightsPath=None)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
    	metrics=["accuracy"])
    
    print("[INFO] training...")
    model.fit(trainData, trainLabels, batch_size=128, nb_epoch=20, verbose=1)

	# show the accuracy on the testing set
    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(testData, testLabels,
		batch_size=128, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
    print(len(model.layers))
    lastLayerOp = K.function([model.layers[0].input],
                                  [model.layers[9].output])

    return lastLayerOp, trainData, testData, trainLabels, testLabels
            
def dumpFeatures(layeredOutput, data, dataLabels, isTrain):
    chunkSize = 5000
    strName = "Train" if isTrain else "Test"
    fileName = outputDir + 'MNIST_' + strName + '_FV_' + str(0) + ".csv"
    titleRow = ["Index", "Label", "Data", "Feature Vector"]
    startTime = time.time()
    totalTime = startTime
    featureDataList = list()
    for idx in range(len(data)):
        '''if idx == 1:
            break'''
        x = [data[idx]]
        featureVector =  layeredOutput([x])[0]
        dataList = [idx, dataLabels[idx].tolist(), (data[idx] * 255.0).tolist(), featureVector.tolist()[0]]
        #print(dataList)
        featureDataList.append(dataList)
        if idx != 0 and (idx  + 1) % chunkSize == 0:
            print("Extracted %d %s data features in [%.3f seconds]" % (idx, strName, time.time() - startTime))
            startTime = time.time()
            fileName = outputDir + 'MNIST_' + strName + '_FV_' + str(idx + 1) + ".csv"
            df = pd.DataFrame(featureDataList, columns = titleRow)
            df.to_csv(fileName)
            featureDataList.clear()
    fileName = outputDir + 'MNIST_' + strName + '_FV_' + str(idx + 1) + ".csv"
    df = pd.DataFrame(featureDataList, columns = titleRow)
    df.to_csv(fileName)
    print("Extracted total of %d %s data features in [%.3f seconds]" % (len(data), strName, time.time() - totalTime))
    
if __name__ == "__main__":
    layeredOutput, trainData, testData, trainLabels, testLabels = extractFeatures()
    dumpFeatures(layeredOutput, trainData, trainLabels, True)
    dumpFeatures(layeredOutput, testData, testLabels, False)
    