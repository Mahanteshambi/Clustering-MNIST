# -*- coding: utf-8 -*-
"""
@author: Ambi
"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time, os, random


class KMeansClustering:
    
    kMeansOPDir = 'output/kmeans/'    
    
    def __init__(self, featureVectorList, labels, characterDataList):
        self.featureVectorList = featureVectorList
        self.labels = labels
        self.characterDataList = characterDataList
        
    def fitAndPredict(self):
        startTime = time.time()
        k = 10
        kmeans = KMeans(n_clusters=k)
        kmeans = kmeans.fit(self.featureVectorList)
        self.kMeanslabels = kmeans.predict(self.featureVectorList)
        print("Clustered using KMeans in [%.3f seconds]" % (time.time() - startTime))
        self.kMeanscentroids = kmeans.cluster_centers_
      
    def findOptimalK(self):
        wcss = list()
        for k in range(1, 15):
            kmeans = KMeans(n_clusters=k)
            kmeans = kmeans.fit(self.featureVectorList)
            wcss.append(kmeans.inertia_)
            
        plt.figure(figsize=(15, 6))
        plt.plot(range(1, 15), wcss, marker = "o")
        
    def dumpResults(self):
        clusterData = []
        for i in range(10):
            clusterData.append([])
        for i in range(len(self.kMeanslabels)):
            clusterData[self.kMeanslabels[i]].append([self.characterDataList[i], self.labels[i]])
            
        if not os.path.exists(self.kMeansOPDir):
            os.makedirs(self.kMeansOPDir)
            
        for l in range(10):
            print("Plotting for label %d" % (l))
            fig, axes = plt.subplots(nrows=10, ncols=10, sharex=True)
            fig.set_figheight(15)
            fig.set_figwidth(15)
            count = 0
            randomNoList = random.sample(range(0, len(clusterData[l])), 100)
            for i in range(10):
                for j in range(10):
                    axes[i][j].imshow(clusterData[l][randomNoList[count]][0])
                    count += 1
            fig.savefig(self.kMeansOPDir + 'kmeans_cluster' + str(l) + '.png')
        return clusterData