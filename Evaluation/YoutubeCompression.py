__author__ = 'GongLi'

import Utility as util
import numpy as np
import random
from TransferLearning import SVM_T

def randomEvaluate(distance, binaryLabel, semanticLabels):

    # construct training indices and testing indices
    labelIndices = []
    setlabels = ["birthday", "picnic", "parade", "show", "sports", "wedding"]

    for label in setlabels:
        labelIndices.append([i for i in range(len(semanticLabels)) if semanticLabels[i] == label])

    trainingIndice = []
    for labelIndice in labelIndices:
        # Select 30 indices from labelIndice
        trainingIndice += random.sample(labelIndice, 40)

    testingIndice = [i for i in range(len(semanticLabels))]
    for indice in trainingIndice:
        testingIndice.remove(indice)

    # SVM_T to test the above cases
    aps = SVM_T.runSVM_T(distance, binaryLabel, trainingIndice,testingIndice)

    return aps


if __name__ == "__main__":

    originalDistance = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/All/Normal/all_DistanceMatrix_Level0.pkl")
    originalLabels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/All/Normal/all_labels_Level0.pkl")

    # cut and retain only Youtube videos
    originalDistance = util.sliceArray(originalDistance, [i for i in range(195, 1101, 1)])
    originalLabels = originalLabels[195:]

    compressedDistance = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/CompressedYoutube/YoutubeCompressed_DistanceMatrix_Level0.pkl")
    compressedLabels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/CompressedYoutube/YoutubeCompressed_labels_Level0.pkl")

    # construct binary labels
    setlabels = ["birthday", "picnic", "parade", "show", "sports", "wedding"]
    binaryLabels = np.zeros((len(compressedLabels))).reshape((len(compressedLabels), 1))

    for label in setlabels:

        tempLabel = np.zeros((len(compressedLabels))).reshape((len(compressedLabels), 1))
        for i in range(len(compressedLabels)):
            if compressedLabels[i] == label:
                tempLabel[i][0] = 1

        binaryLabels = np.concatenate((binaryLabels, tempLabel), axis=1)

    binaryLabels = binaryLabels[::, 1::]

    # construct training indices and testing indices

    #----------------------------------- original distances
    originalDistances = []
    originalDistances.append(originalDistance)

    all_aps = []

    for i in range(5):
        aps = randomEvaluate(originalDistances, binaryLabels, originalLabels)
        all_aps.append(aps)

    all_aps = np.array(all_aps)
    meanAP = np.mean(all_aps)

    rowMean = np.mean(all_aps, axis=1)
    sd = np.std(rowMean)

    print "meanAP: "+str(meanAP)
    print "standard deviation: "+str(sd)

    #----------------------------------- compressed distances
    compressedDistances = []
    compressedDistances.append(compressedDistance)

    all_aps = []

    for i in range(5):
        aps = randomEvaluate(compressedDistances, binaryLabels, originalLabels)
        all_aps.append(aps)

    all_aps = np.array(all_aps)
    meanAP = np.mean(all_aps)

    rowMean = np.mean(all_aps, axis=1)
    sd = np.std(rowMean)

    print "meanAP: "+str(meanAP)
    print "standard deviation: "+str(sd)


    print "Yes"






