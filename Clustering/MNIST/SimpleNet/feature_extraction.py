# -*- coding: utf-8 -*-
"""
@author: Ambi
"""

from model import SimpleNet

import keras
from keras.datasets import mnist
from keras import backend as K
import os, time
import pandas as pd

class FeatureExtraction:
    isLogEnabled = True
    # input image dimensions
    img_rows, img_cols = 28, 28
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def train(self, batch_size, epochs):
        num_classes = 10
        if self.isLogEnabled:
            print('Trainning MNIST!')
        if K.image_data_format() == 'channels_first':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.img_rows, self.img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, self.img_rows, self.img_cols)
            input_shape = (1, self.img_rows, self.img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, 1)
            input_shape = (self.img_rows, self.img_cols, 1)
            
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255
        if self.isLogEnabled:
            print('x_train shape:', self.x_train.shape)
            print(self.x_train.shape[0], 'train samples')
            print(self.x_test.shape[0], 'test samples')
        
        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, num_classes)
        
        self.model = SimpleNet.build(num_classes, input_shape)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(self.x_test, self.y_test))
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        if self.isLogEnabled:
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            
    def extract_dump_features(self):
        
        # As model is of 8 layers, then we need to extract neuron values from
        # 7th layer
        layeredOutput = K.function([self.model.layers[0].input],
                                      [self.model.layers[6].output])
        chunkSize = 5000
        outputDir = './output/cnn_features/'
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        titleRow = ["Index", "Label", "Data", "Feature Vector"]
        for i in range(2):
            strName = "Train" if (i == 0) else "Test"
            data = self.x_train if (i == 0) else self.x_test
            datalabels = self.y_train if (i ==0) else self.y_test
            fileName = outputDir + 'MNIST_' + strName + '_FV_' + str(0) + ".csv"
            startTime = time.time()
            totalTime = startTime
            featureDataList = list()
            isWriteComplete = False
            for idx in range(len(data)):
                x = [data[idx]]
                featureVector =  layeredOutput([x])[0]
                dataList = [idx, datalabels[idx].tolist(), (data[idx] * 255.0).tolist(), featureVector.tolist()[0]]
                #print(dataList)
                featureDataList.append(dataList)
                isWriteComplete = False
                if idx != 0 and (idx  + 1) % chunkSize == 0:
                    print('Features list len: ' + str(len(featureDataList)))
                    print("Extracted %d %s data features in [%.3f seconds]" % (idx + 1, strName, time.time() - startTime))
                    startTime = time.time()
                    fileName = outputDir + 'MNIST_' + strName + '_FV_' + str(idx + 1) + ".csv"
                    df = pd.DataFrame(featureDataList, columns = titleRow)
                    df.to_csv(fileName)
                    featureDataList.clear()
                    isWriteComplete = True

            if isWriteComplete == False:
                fileName = outputDir + 'MNIST_' + strName + '_FV_' + str(idx + 1) + ".csv"
                df = pd.DataFrame(featureDataList, columns = titleRow)
                df.to_csv(fileName)
            print("Extracted total of %d %s data features in [%.3f seconds]" % (len(data), strName, time.time() - totalTime))
            
if __name__ == "__main__":
    batch_size = 1280
    epochs = 15
    
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    featureExtraction = FeatureExtraction(x_train, y_train, x_test, y_test)
    
    # Train CNN model to understand MNIST dataset features for classifications.
    featureExtraction.train(batch_size, epochs)
    
    # Extract features from CNN model for clustering purpose
    featureExtraction.extract_dump_features()