__author__ = 'GongLi'

from scipy.io import loadmat
import SVM_T
import numpy as np

def evaluate(distance):

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

    print "meanAP: "+str(meanAP)
    print "standard deviation: "+str(sd)




if __name__ == "__main__":
    import Utility as util

    originalKodakDistances = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/Kodak_distanceMatrix_version2.pkl")
    originalLabels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/Kodak_labels_version2.pkl")

    softWeightKodakDistance = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/NewMethods/KodakSoftWeight_DistanceMatrix_Level0.pkl")
    softWeightLabels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/NewMethods/KodakSoftWeight_labels_Level0.pkl")

    evaluate(originalKodakDistances)
    evaluate(softWeightKodakDistance)







