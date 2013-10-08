__author__ = 'GongLi'

import SVM_T
import Utility as util
import numpy as np
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

    levelZero = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc1000/Kodak_LevelZero_Distance.pkl")
    levelOneUnaligned = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc1000/Kodak_LevelOne_Unaligned_Distance.pkl")
    levelOneAligned = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc1000/Kodak_LevelOne_Aligned_Distance.pkl")
    GmmAssign = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc1000/Kodak_GMM_Assignment_Distance.pkl")

    meanAP, std = predefinedIndices(levelZero)
    print "L0\t" +   str(meanAP) +u" \u00B1 "+str(std)

    meanAP, std = predefinedIndices(levelOneUnaligned)
    print "Level 1 unaligned\t" +   str(meanAP) +u" \u00B1 "+str(std)

    meanAP, std = predefinedIndices(levelOneAligned)
    print "Level 1 aligned\t" +   str(meanAP) +u" \u00B1 "+str(std)

    meanAP, std = predefinedIndices(GmmAssign)
    print "GMM assignment\t" +   str(meanAP)+u" \u00B1 "+str(std)
