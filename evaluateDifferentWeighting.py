__author__ = 'GongLi'

import SVM_T
import Utility as util
import numpy as np
import math
from sklearn.svm import SVC
from scipy.io import loadmat

def multiClassSVM(distances, trainingIndice, testingIndice, semanticLabels):

    trainDistance = distances[np.ix_(trainingIndice, trainingIndice)]
    testDistance = distances[np.ix_(testingIndice,trainingIndice)]

    meanTrainValue = np.mean(trainDistance)

    trainGramMatrix = math.e **(0 - trainDistance / meanTrainValue)
    testGramMatrix = math.e ** (0 - testDistance / meanTrainValue)
    trainLabels = [semanticLabels[i] for i in trainingIndice]
    testLabels = [semanticLabels[i] for i in testingIndice]

    clf = SVC(kernel = "precomputed")
    clf.fit(trainGramMatrix, trainLabels)
    SVMResults = clf.predict(testGramMatrix)

    correct = sum(1.0 * (SVMResults == testLabels))
    accuracy = correct / len(testLabels)
    # print "accuracy: " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

    return accuracy


def processMatrix(name, arrayForProcess):

    all_aps = np.array(arrayForProcess)
    meanAP = np.mean(all_aps)

    rowMean = np.mean(all_aps, axis=1)
    sd = np.std(rowMean)

    print name + "_meanAP: "+str(meanAP)
    print name + "_standard deviation: "+str(sd)


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

    print "meanAP: "+str(meanAP)
    print "standard deviation: "+str(sd)
    print ""


if __name__ == "__main__":

    origianalDistance = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/Kodak_distanceMatrix_version2.pkl")

    bxxDistance = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/DifferentWeight/bxxDistance.pkl")
    txxDistance = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/DifferentWeight/txxDistance.pkl")
    txcDistance = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/DifferentWeight/txcDistance.pkl")
    tfxDistance = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/DifferentWeight/tfxDistance.pkl")
    tfcDistance = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/DifferentWeight/tfcDistance.pkl")
    softDistance = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/DifferentWeight/softDistance.pkl")

    labels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/DifferentWeight/softLabels.pkl")
    binaryLabels = util.generateBinaryLabels(labels)

    bxxDisatnces = []
    txxDistances = []
    txcDistances = []
    tfxDistances = []
    tfcDistances = []
    softDistances = []
    origianalDistances = []

    bxxDisatnces.append(bxxDistance)
    txxDistances.append(txxDistance)
    txcDistances.append(txcDistance)
    tfxDistances.append(tfxDistance)
    tfcDistances.append(tfcDistance)
    softDistances.append(softDistance)
    origianalDistances.append(origianalDistance)

    all_bxx = []
    all_txx = []
    all_txc = []
    all_tfx = []
    all_tfc = []
    all_soft = []
    all_orighinal = []

    predefinedIndices(bxxDistance)
    predefinedIndices(txxDistance)
    predefinedIndices(txcDistance)
    predefinedIndices(tfxDistance)
    predefinedIndices(tfcDistance)
    predefinedIndices(softDistance)
    predefinedIndices(origianalDistance)

