# -*- coding: utf-8 -*-
"""
@author: Ambi
"""
from sklearn.cluster import DBSCAN
import time

class DBScanClustering:
    
    def __init__(self, featureVectorList, labels):
        self.featureVectorList = featureVectorList
        self.labels = labels
        
    def fitAndPredict(self):
        startTime = time.time()
        k = 10
        dbscan = DBSCAN(eps=5, min_samples=10)
        dbscan.fit(self.featureVectorList)
        labels = dbscan.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)