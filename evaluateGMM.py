__author__ = 'GongLi'
import SVM_T
import Utility as util
import numpy as np
import math
from sklearn.svm import SVC
from scipy.io import loadmat

def predefinedIndices(distance):

    distances = []
    distances.append(distance)
    labels = loadmat("labels.mat")['labels']

    all_aps = []
    for i in range(1,6,1):
        trainingIndices = loadmat(str(i)+".mat")['training_ind']
        trainingIndiceList = []
        testingIndices = loadmat(str(i)+".mat")['test_ind']
        testingIndiceList = []

        # Construct indices
        for i in range(trainingIndices.shape[0]):
            trainingIndiceList.append(trainingIndices[i][0] - 1)

        for i in range(testingIndices.shape[1]):
            testingIndiceList.append(testingIndices[0][i] - 1)

        targetTrainingIndices = []
        auxiliaryTrainingIndices = []
        for i in trainingIndiceList:
            if i <= 194:
                targetTrainingIndices.append(i)
            else:
                auxiliaryTrainingIndices.append(i)

        aps = SVM_T.runSVM_T(distances, labels, targetTrainingIndices, testingIndiceList)
        all_aps.append(aps)

    all_aps = np.array(all_aps)
    meanAP = np.mean(all_aps)

    rowMean = np.mean(all_aps, axis=1)
    sd = np.std(rowMean)

    return meanAP, sd

if __name__ == "__main__":

    FullPCA36_50 = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/GMM/Full Covariance/PCA36_GMM_n_iteration50_KodakDistances.pkl")
    FullPCA64_50 = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/GMM/Full Covariance/PCA64_FUll_GMM_n_iteration50_KodakDistances.pkl")

    SphericalPCA128_ALL_50 = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/GMM/Spherical Covariance/All 128/All_GMM_distances.pkl")
    SphericalPCA128_Kodak_50 = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/GMM/Spherical Covariance/128 GMM_KodakDistances_n_iteration50.pkl")
    SphericalPCA128_Kodak_20 = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/GMM/Spherical Covariance/128 GMM_KodakDistances_n_iteration20.pkl")
    SphericalPCA128_Kodak_500 = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/GMM/Spherical Covariance/128 GMM_n_iteration500_KodakDistance.pkl")
    SphericalPCA64_Kodak_50 = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/GMM/Spherical Covariance/PCA64_Spherical_GMM_n_iteration50_KodakDistances.pkl")

    EMD_level0 = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/All/Normal/all_DistanceMatrix_Level0.pkl")


    meanAP, std = predefinedIndices(FullPCA36_50)
    print "FullPCA36_50\t" +str(meanAP) +u" \u00B1 " +str(std)

    meanAP, std = predefinedIndices(FullPCA64_50)
    print "FullPCA64_50\t" +   str(meanAP) +u" \u00B1 "+str(std)

    meanAP, std = predefinedIndices(SphericalPCA128_ALL_50)
    print "SphericalPCA128_ALL_50\t" +   str(meanAP) +u" \u00B1 "+str(std)


    meanAP, std = predefinedIndices(SphericalPCA64_Kodak_50)
    print "SphericalPCA64_Kodak_50\t" +   str(meanAP) +u" \u00B1 "+str(std)

    meanAP, std = predefinedIndices(EMD_level0)
    print "EMD level 0\t" + str(meanAP) +u" \u00B1 "+str(std)

