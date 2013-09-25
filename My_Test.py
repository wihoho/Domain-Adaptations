__author__ = 'GongLi'

import pickle
from scipy.io import loadmat
import TestAll

file = open("LevelZero/all_DistanceMatrix_Level0.pkl", "rb")
distanceOne = pickle.load(file)


distances = []
distances.append(distanceOne)

templabels = loadmat("labels.mat")['labels']

# SIFT Level 0
tempList = []
tempList.append(distanceOne)

TestAll.evaluate(tempList, "MyDistance_SIFT_L0_Result2.xls", templabels)
print "SIFT L0"
