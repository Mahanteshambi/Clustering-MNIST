# -*- coding: utf-8 -*-
"""
@author: Ambi
"""
import time
from sklearn.cluster import AgglomerativeClustering
class AgglomerativeClusterings():
    
    def __init__(self, featureVectorList, labels):
        self.featureVectorList = featureVectorList
        self.labels = labels
        
    def fitAndPredict(self):
        startTime = time.time()
        k = 10
        agglomerative = AgglomerativeClustering(n_clusters=k)
        agglomerative = agglomerative.fit(self.featureVectorList)
        self.kMeanslabels = agglomerative.predict(self.featureVectorList)
        print("Clustered using KMeans in [%.3f seconds]" % (time.time() - startTime))
        self.kMeanscentroids = agglomerative.cluster_centers_
        labels = agglomerative.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)