__author__ = 'GongLi'
from scipy.io import loadmat
import math
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score

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

def runMKL(distances, labels, trainingIndiceList, testingIndiceList):
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

    aps = []
    for classNum in range(labels.shape[1]):
        thisClassLabels = labels[::, classNum]
        trainingLabels = [thisClassLabels[index] for index in trainingIndiceList]
        testingLabels = [thisClassLabels[index] for index in testingIndiceList]

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
        AP = average_precision_score(testingLabels, testScores)
        aps.append(AP)

    return aps


if __name__ == "__main__":

    distanceOne = loadmat("dist_SIFT_L0.mat")['distMat']
    distanceTwo = loadmat("dist_SIFT_L1.mat")['distMat']

    distances = []
    distances.append(distanceOne)
    distances.append(distanceTwo)
    labels = loadmat("labels.mat")['labels']

    all_aps = []
    for i in range(1,6,1):
        trainingIndices = loadmat(str(i)+".mat")['training_ind']
        trainingIndiceList = []
        testingIndices = loadmat(str(i)+".mat")['test_ind']
        testingIndiceList = []

        # Construct training distances
        for i in range(trainingIndices.shape[0]):
            trainingIndiceList.append(trainingIndices[i][0] - 1)

        for i in range(testingIndices.shape[1]):
            testingIndiceList.append(testingIndices[0][i] - 1)

        aps = runMKL(distances, labels, trainingIndiceList, testingIndiceList)
        all_aps.append(aps)

    all_aps = np.array(all_aps)
    meanAP = np.mean(all_aps)

    rowMean = np.mean(all_aps, axis=1)
    sd = np.std(rowMean)

    print "meanAP: "+str(meanAP)
    print "standard deviation: "+str(sd)