__author__ = 'GongLi'
from scipy.io import loadmat
import math
import numpy as np
from sklearn.svm import SVC
import Utility as util

def constructBaseKernels(kernel_type, kernel_params, D2):

    baseKernels = []

    for i in range(len(kernel_type)):
        for j in range(len(kernel_params)):

            type = kernel_type[i]
            param = kernel_params[j]

            if type == "rbf":
                baseKernels.append(math.e **(- param * D2))
            elif type == "lap":
                baseKernels.append(math.e **(- (param * D2) ** (0.5)))
            elif type == "id":
                baseKernels.append(1.0 / ((param * D2) ** (0.5) + 1))
            elif type == "isd":
                baseKernels.append(1.0 / (param * D2 + 1))

    return baseKernels

def sliceArray(testArray, indices):

    return testArray[np.ix_(indices, indices)]

def returnKernel(coefficients, baseKernels):

    resultKernel = coefficients[0] * baseKernels[0]

    for i in range(1, len(coefficients), 1):
        resultKernel += coefficients[i] * baseKernels[i]

    return resultKernel

def SVM_One_vs_All(distances, labels, trainingIndiceList, testingIndiceList):
    # Construct base kernels
    baseKernels = []
    for i in range(len(distances)):
        distance = distances[i]
        distance = distance ** 2
        trainingDistances = sliceArray(distance, trainingIndiceList)

        # Define kernel parameters
        gramma0 = 1.0 / np.mean(trainingDistances)
        kernel_params = [gramma0 *(2 ** index) for index in range(-3, 2, 1)]

        # Construct base kernels & pre-learned classifier
        baseKernel = constructBaseKernels(["rbf", "lap", "isd","id"], kernel_params, distance)
        baseKernels += baseKernel

    scores = []
    for classNum in range(labels.shape[1]):
        thisClassLabels = labels[::, classNum]
        trainingLabels = [thisClassLabels[index] for index in trainingIndiceList]

        # MKL
        trainingBaseKernls = []
        testingBaseKernels = []

        for baseKernel in baseKernels:
            tempKernel = sliceArray(baseKernel, trainingIndiceList)
            trainingBaseKernls.append(tempKernel)

            tempKernel = baseKernel[np.ix_(testingIndiceList, trainingIndiceList)]
            testingBaseKernels.append(tempKernel)

        coefficients = [1.0 / len(trainingBaseKernls)] * len(trainingBaseKernls)
        trainingKernels = returnKernel(coefficients, trainingBaseKernls)
        SVMModel = SVC(kernel="precomputed")
        SVMModel.fit(trainingKernels, trainingLabels)

        coefficients = [1.0 / len(testingBaseKernels)] * len(testingBaseKernels)
        testKernels = returnKernel(coefficients, testingBaseKernels)

        testScores = SVMModel.decision_function(testKernels)
        scores.append(testScores.flatten())

    # Find the label with the largest score
    scores = np.vstack(scores)
    ranks = np.argmax(scores, axis=0)

    labelSet = ["birthday", "parade", "picnic", "show", "sports", "wedding"]
    predictLabels = [labelSet[i] for i in ranks]

    return predictLabels


def SVM_One_vs_One(distances, semanticLabels, trainingIndiceList, testingIndiceList):

    # Construct base kernels
    baseKernels = []
    for i in range(len(distances)):
        distance = distances[i]
        distance = distance ** 2
        trainingDistances = sliceArray(distance, trainingIndiceList)

        # Define kernel parameters
        gramma0 = 1.0 / np.mean(trainingDistances)
        kernel_params = [gramma0 *(2 ** index) for index in range(-3, 2, 1)]

        # Construct base kernels & pre-learned classifier
        baseKernel = constructBaseKernels(["rbf", "lap", "isd","id"], kernel_params, distance)
        baseKernels += baseKernel

    # Build a classifier for each pair
    labelSet = ["birthday", "parade", "picnic", "show", "sports", "wedding"]
    pairs = util.paris(labelSet)

    stackOfPredictions = []
    for pair in pairs:

        positiveClass = pair[0]
        negativeClass = pair[1]

        thisTargetTrainIndices = []
        thisTargetBinaryLabels = []

        for i in trainingIndiceList:
            if semanticLabels[i] in pair:
                thisTargetTrainIndices.append(i)

                if semanticLabels[i] == positiveClass:
                    thisTargetBinaryLabels.append(1)
                elif semanticLabels[i] == negativeClass:
                    thisTargetBinaryLabels.append(-1)

        # MKL
        trainingBaseKernls = []
        testingBaseKernels = []

        for baseKernel in baseKernels:
            tempKernel = sliceArray(baseKernel, thisTargetTrainIndices)
            trainingBaseKernls.append(tempKernel)

            tempKernel = baseKernel[np.ix_(testingIndiceList, thisTargetTrainIndices)]
            testingBaseKernels.append(tempKernel)

        coefficients = [1.0 / len(trainingBaseKernls)] * len(trainingBaseKernls)
        trainingKernels = returnKernel(coefficients, trainingBaseKernls)
        SVMModel = SVC(kernel="precomputed")
        SVMModel.fit(trainingKernels, thisTargetBinaryLabels)

        coefficients = [1.0 / len(testingBaseKernels)] * len(testingBaseKernels)
        testKernels = returnKernel(coefficients, testingBaseKernels)

        testScores = SVMModel.decision_function(testKernels)

        tempFinalScores = 1.0 / (1 + math.e ** (-testScores) )

        thisPredictLabels = []
        for score in tempFinalScores:
            if score < 0.5:
                thisPredictLabels.append(negativeClass)
            else:
                thisPredictLabels.append(positiveClass)
        stackOfPredictions.append(thisPredictLabels)

    stackOfPredictions = np .array(stackOfPredictions)

    finalLabels = []
    shape = stackOfPredictions.shape
    for i in range(shape[1]):
        temp = []
        for j in range(shape[0]):
            temp.append(stackOfPredictions[j][i])
        dict = {}
        for item in temp:
            if item in dict.keys():
                dict[item] = dict[item] + 1
            else:
                dict[item] = 1


        keys = dict.keys()
        curLabel = keys[0]
        curVal = dict[curLabel]

        for l in keys[1:]:
            if curVal < dict[l]:
                curLabel = l
                curVal = dict[curLabel]
        finalLabels.append(curLabel)

    return finalLabels



if __name__ == "__main__":

    distanceOne = loadmat("/Users/GongLi/PycharmProjects/DomainAdaption/dist_SIFT_L0.mat")['distMat']
    distanceTwo = loadmat("/Users/GongLi/PycharmProjects/DomainAdaption/dist_SIFT_L1.mat")['distMat']
    distanceThree = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc1000/All/GMM_ALL_Distance.pkl")

    distances = []
    distances.append(distanceOne)
    # distances.append(distanceTwo)
    # distances.append(distanceThree)
    labels = loadmat("/Users/GongLi/PycharmProjects/DomainAdaption/labels.mat")['labels']

    trueLabels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/All/Normal/all_labels_Level0.pkl")

    for i in range(1,6,1):
        trainingIndices = loadmat("/Users/GongLi/PycharmProjects/DomainAdaption/"+str(i)+".mat")['training_ind']
        trainingIndiceList = []
        testingIndices = loadmat("/Users/GongLi/PycharmProjects/DomainAdaption/"+str(i)+".mat")['test_ind']
        testingIndiceList = []

        # Construct training distances
        for i in range(trainingIndices.shape[0]):
            trainingIndiceList.append(trainingIndices[i][0] - 1)

        for i in range(testingIndices.shape[1]):
            testingIndiceList.append(testingIndices[0][i] - 1)

        trueLabels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/All/Normal/all_labels_Level0.pkl")

        predictions = SVM_One_vs_All(distances, labels, trainingIndiceList, testingIndiceList )

        # calculate prediction accuracy
        testTrueLabels = [trueLabels[i] for i in testingIndiceList]

        predictions = np.array(predictions)
        testTrueLabels = np.array(testTrueLabels)

        print "Multi-class Accuracy (One vs All): " + str(util.evaluateAccuracy(predictions, testTrueLabels))


        predictions = SVM_One_vs_One(distances, trueLabels, trainingIndiceList, testingIndiceList)
        predictions = np.array(predictions)

        print "Multi-class Accuracy (One vs One): " + str(util.evaluateAccuracy(predictions, testTrueLabels))

        print " "
