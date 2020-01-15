# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 17:09:02 2018

@author: Ambi
"""
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from KMeansClustering import KMeansClustering
from DBScanClustering import DBScanClustering
from AgglomerativeClustering import AgglomerativeClusterings
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA




def getFeatureVectors(dir):
    featureFiles = [join(dir, file) for file in listdir(dir) if isfile(join(dir, file))]
    featureFiles.sort()
    fileCount = 0
    titleRow = ["Index", "Label", "Data", "Feature Vector"]
    labelList = list()
    characterDataList = list()
    featureVectorList = list()
    for file in featureFiles:
        '''if fileCount == 1:
            break'''
        fileCount += 1
        print("Parsing %s" % (file))
        data = pd.read_csv(file)
        dataFrame = pd.DataFrame(data, columns = titleRow)
        for i in range(len(dataFrame)):
            label = dataFrame['Label'][i].replace('[', '').replace(']', '').split(',')
            characterData =  dataFrame['Data'][i].replace('[', '').replace(']', '').split(',')
            featureVector =  dataFrame['Feature Vector'][i].replace('[', '').replace(']', '').split(',')
            
            labelList.append(label)
            characterDataList.append(characterData)
            featureVectorList.append(featureVector)
            
    for i in range(len(characterDataList)):
        characterDataList[i] = [int(float(value)) for value in characterDataList[i]]
        characterDataList[i] = np.array(characterDataList[i]).reshape(28, 28)
        featureVectorList[i] = [float(value) for value in featureVectorList[i]]
        labelList[i] = [float(value) for value in labelList[i]]
    return labelList, characterDataList, featureVectorList

def visualize(labelList, featureVectorList, outputDir):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(featureVectorList)
    
    labels = []
    for l in range(len(labelList)):
      labels.append(labelList[l].index(1.0))
    
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], alpha=0.7, c=labels, cmap='viridis_r')
    plt.xlabel('MNIST images x dim')
    plt.ylabel('MNIST images y dim'); 
    plt.savefig(outputDir + 'MNIST_clusters_pca.png')

    
def evaluateKMeans(clusterData):
    print('')
    totalAccuracy = 0
    for j in range(10):
        labelsClusteredList = []
        for i in range(10):
            labelsClusteredList.append(0)
        for i in range(len(clusterData[j])):
            labelsClusteredList[clusterData[j][i][1].index(1.0)] += 1
        clusterAccuracy = max(labelsClusteredList) / sum(labelsClusteredList) * 100.0
        totalAccuracy += clusterAccuracy
    print("KMeans clustering Accuracy " + str(totalAccuracy / 10))



startTime = time.time()
labelList, characterDataList, featureVectorList = getFeatureVectors('output/cnn_features/')
print("Extracted details in [%.3f seconds]" % (time.time() - startTime))

kMeansObj = KMeansClustering(featureVectorList, labelList, characterDataList)
kMeansObj.findOptimalK()
kMeansObj.fitAndPredict()


startTime = time.time()
clusterData = kMeansObj.dumpResults()
print("Dumped results in [%.3f seconds]" % (time.time() - startTime))
evaluateKMeans(clusterData)
visualize(labelList, featureVectorList, kMeansObj.kMeansOPDir)

#dbScanClustering = DBScanClustering(featureVectorList, labelList)
#dbScanClustering.fitAndPredict()

#agglomerativeClustering = AgglomerativeClusterings(featureVectorList, labelList)
#agglomerativeClustering.fitAndPredict()


    


