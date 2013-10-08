__author__ = 'GongLi'


import TestAll
from scipy.io import loadmat
import Utility as util


if __name__ == "__main__":

    distanceOne = loadmat("dist_SIFT_L0.mat")['distMat']
    distanceTwo = loadmat("dist_SIFT_L1.mat")['distMat']

    distances = []
    distances.append(distanceOne)
    distances.append(distanceTwo)

    tempLabels = loadmat("labels.mat")['labels']

    # SIFT Level 0
    tempList = []
    tempList.append(distanceOne)

    TestAll.evaluate(tempList, "/Users/GongLi/PycharmProjects/DomainAdaption/DuanSetResult/2013.10.8/SIFT_L0_Result.xls", tempLabels)
    print "SIFT L0"

    # SIFT Level 1
    tempList = []
    tempList.append(distanceTwo)

    TestAll.evaluate(tempList, "/Users/GongLi/PycharmProjects/DomainAdaption/DuanSetResult/2013.10.8/SIFT_L1_Result.xls", tempLabels)
    print "SIFT L1"

    # # SIFT Level 0 & 1
    # tempList = []
    # tempList.append(distanceTwo)
    # tempList.append(distanceOne)
    # TestAll.evaluate(tempList, "/Users/GongLi/PycharmProjects/DomainAdaption/DuanSetResult/2013.10.8/SIFT_L0_L1_Result.xls", tempLabels)
    # print "SIFT L0 & 1"
