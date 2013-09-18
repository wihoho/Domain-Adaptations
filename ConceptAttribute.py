__author__ = 'GongLi'

from scipy.io import loadmat
import math
import numpy as np
from sklearn.svm import SVC
import Utility as util
import random

def runConceptAttribute(distances, labels, acturalSemanticLabels,auxiliaryTrainingIndices, targetTrainingIndices, targetTestingIndices):

    all_trainingIndices = auxiliaryTrainingIndices + targetTrainingIndices

    baseKernels = []
    for i in range(len(distances)):
        distance = distances[i]
        distance = distance ** 2
        trainingDistances = util.sliceArray(distance, all_trainingIndices)

        # Define kernel parameters
        gramma0 = 1.0 / np.mean(trainingDistances)
        kernel_params = [gramma0 *(2 ** index) for index in range(-3, 2, 1)]

        # Construct base kernels & pre-learned classifier
        baseKernel = util.constructBaseKernels(["rbf", "lap", "isd","id"], kernel_params, distance)
        baseKernels += baseKernel

    # Train classifiers based on Youtube videos & Assign concept scores to target domain
    targetTrainingConceptScores = np.zeros((len(targetTrainingIndices), labels.shape[1]))
    targetTestingConceptScores = np.zeros((len(targetTestingIndices), labels.shape[1]))

    for classNum in range(labels.shape[1]):
        thisClassLabels = labels[::, classNum]
        auTrainingLabels = [thisClassLabels[index] for index in auxiliaryTrainingIndices]

        targetTrainDvs = []
        targetTestDvs = []
        for m in range(len(baseKernels)):
            baseKernel = baseKernels[m]
            auKtrain = util.sliceArray(baseKernel, auxiliaryTrainingIndices)
            Ktest = baseKernel[::, auxiliaryTrainingIndices]

            clf = SVC(kernel="precomputed")
            clf.fit(auKtrain, auTrainingLabels)
            dv = clf.decision_function(Ktest)

            targetTrainDv = [dv[index][0] for index in targetTrainingIndices]
            targetTrainDvs.append(targetTrainDv)

            targetTestDv = [dv[index][0] for index in targetTestingIndices]
            targetTestDvs.append(targetTestDv)

        targetTrainDvs = np.array(targetTrainDvs)
        targetTestDvs = np.array(targetTestDvs)

        # Fuse decision values from different kernels
        tempScores = 1.0 / (1 + math.e **(-targetTrainDvs))
        targetTrainDvs = np.mean(tempScores, axis = 0)

        tempScores = 1.0 / (1 + math.e ** (-targetTestDvs))
        targetTestDvs = np.mean(tempScores, axis = 0)

        for trainIndex in range(len(targetTrainingIndices)):
            targetTrainingConceptScores[trainIndex][classNum] = targetTrainDvs[trainIndex]
        for testIndex in range(len(targetTestingIndices)):
            targetTestingConceptScores[testIndex][classNum] = targetTestDvs[testIndex]

    # Use new representations to classify
    actualTrainLabels = [acturalSemanticLabels[i] for i in targetTrainingIndices]
    actualTestLabels = [acturalSemanticLabels[i] for i in targetTestingIndices]


    SVMmodel = SVC(kernel = "rbf")
    SVMmodel.fit(targetTrainingConceptScores, actualTrainLabels)
    ap = SVMmodel.score(targetTestingConceptScores, actualTestLabels)

    print "Rbf: " +str(ap)

    return ap

def baseLineSVMT(distances, semanticLabels, targetTrainingIndice, targetTestingIndice):

    baseKernels = []
    for i in range(len(distances)):
        distance = distances[i]
        distance = distance ** 2
        trainingDistances = util.sliceArray(distance, targetTrainingIndice)

        # Define kernel parameters
        gramma0 = 1.0 / np.mean(trainingDistances)
        kernel_params = [gramma0 *(2 ** index) for index in range(-3, 2, 1)]

        # Construct base kernels & pre-learned classifier
        baseKernel = util.constructBaseKernels(["rbf", "lap", "isd","id"], kernel_params, distance)
        baseKernels += baseKernel

    trainLabels = [semanticLabels[i] for i in targetTrainingIndice]
    testLabels = [semanticLabels[i] for i in targetTestingIndice]

    coef = 1.0 / (len(baseKernels))
    finalKernel = coef * baseKernels[0]
    for baseKernel in baseKernels[1:]:
        finalKernel += coef * baseKernel

    trainKernels = util.sliceArray(finalKernel, targetTrainingIndice)
    testKernel = finalKernel[np.ix_(targetTestingIndice, targetTrainingIndice)]

    clf = SVC(kernel="precomputed")
    clf.fit(trainKernels, trainLabels)
    ap = clf.score(testKernel, testLabels)

    print "BaseLine: "+str(ap)

    return ap

def randomEvaluate(distance, binaryLabel, semanticLabels):

    trainingIndice, testingIndice = util.generateRandomIndices(semanticLabels, 3)

    base = baseLineSVMT(distance, semanticLabels, trainingIndice, testingIndice)

    auxiliaryTraining = [i for i in range(195, 1101, 1)]
    cs = runConceptAttribute(distance, binaryLabel, semanticLabels ,auxiliaryTraining, trainingIndice, testingIndice)

    return base, cs


if __name__ == "__main__":

    distanceOne = util.loadObject("LevelZero/all_DistanceMatrix_Level0.pkl")
    labels = loadmat("labels.mat")['labels']

    semanticLabels = util.loadObject("LevelZero/all_labels_Level0.pkl")

    distances = []
    distances.append(distanceOne)

    binaryLabel = util.generateBinaryLabels(semanticLabels)

    semanticLabels = semanticLabels[:195]


    # Write result into an excel file
    import xlwt
    wb = xlwt.Workbook()
    ws = wb.add_sheet("Concept Attribute")

    for i in range(20):
        base, cs = randomEvaluate(distances, binaryLabel, semanticLabels)
        ws.write(0, i, base)
        ws.write(1, i, cs)

    wb.save("ConceptAttribute.xls")

    # all_aps = []
    # for i in range(1,6,1):
    #     print "###################### "+str(i)
    #
    #     trainingIndices = loadmat(str(i)+".mat")['training_ind']
    #     trainingIndiceList = []
    #     testingIndices = loadmat(str(i)+".mat")['test_ind']
    #     testingIndiceList = []
    #
    #     # Construct indices
    #     for i in range(trainingIndices.shape[0]):
    #         trainingIndiceList.append(trainingIndices[i][0] - 1)
    #
    #     for i in range(testingIndices.shape[1]):
    #         testingIndiceList.append(testingIndices[0][i] - 1)
    #
    #     targetTrainingIndices = []
    #     auxiliaryTrainingIndices = []
    #     for i in trainingIndiceList:
    #         if i <= 194:
    #             targetTrainingIndices.append(i)
    #         else:
    #             auxiliaryTrainingIndices.append(i)
    #
    #
    #     baseLineSVMT(distances, semanticLabels, targetTrainingIndices, testingIndiceList)
    #     runConceptAttribute(distances, labels, semanticLabels,auxiliaryTrainingIndices, targetTrainingIndices, testingIndiceList)