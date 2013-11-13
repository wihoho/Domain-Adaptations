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

def calculateMMD(baseKernls, s):

    numberOfKernels = len(baseKernls)
    h = np.zeros((numberOfKernels))

    for i in range(numberOfKernels):
        kernel = baseKernls[i]
        temp = np.dot(np.transpose(s), kernel)
        h[i] = np.dot(temp, s)

    return h

def train_amkl(training_labels, training_basekernels, mmd_values, fp, lam, C):

    # necessary parameters
    numberOfBaseKernels = len(training_basekernels)
    MAX_ITER = 15
    MAX_ITER_LS = 20
    epsilon = 10 ** (-5)
    tau = 10 ** (-5)
    step_size = (math.sqrt(5) - 1) / 2
    theta = 10 ** (-5)

    h = mmd_values
    h = h.reshape((numberOfBaseKernels, 1))
    coefficients = [1.0 / numberOfBaseKernels] * numberOfBaseKernels
    objectives = []

    sizeFP = len(fp)
    fp = np.array(fp).reshape((sizeFP, 1))
    add_kernel = np.dot(fp, np.transpose(fp)) / lam

    # Adaptive learning
    tempModel, tempObj, q = returnAlpha(h, numberOfBaseKernels, coefficients, training_labels, training_basekernels, add_kernel, C, theta)
    objectives.append(tempObj)

    for i in range(1, MAX_ITER):

        coefficients_new = theta * (np.linalg.solve(np.dot(h, np.transpose(h)) + epsilon * np.eye(numberOfBaseKernels), q.reshape((numberOfBaseKernels, 1))))

        # Project coefficients_new on feasible set M
        coefficients_newTwo = []
        for i in range(coefficients_new.shape[0]):
            if coefficients_new[i][0] < 0:
                coefficients_newTwo.append(0)
                continue
            coefficients_newTwo.append(coefficients_new[i][0])

        tempSum = sum(coefficients_newTwo)
        for i in range(len(coefficients_newTwo)):
            coefficients_newTwo[i] /= tempSum

        coefficients_new = coefficients_newTwo

        coefficients_new = np.array(coefficients_new)
        coefficients = np.array(coefficients)

        # run with new coefficients
        eta_t = step_size

        coefficients_cur = [eta_t * coefficients_new[i] + (1 - eta_t) * coefficients[i] for i in range(len(coefficients_new))]
        coefficients_cur = np.array(coefficients_cur)

        tempModel, tempObj, q = returnAlpha(h, numberOfBaseKernels, coefficients_cur, training_labels, training_basekernels,add_kernel, C, theta)

        iter_ls = 0
        while tempObj > objectives[-1] and iter_ls < MAX_ITER_LS:
            eta_t = step_size * eta_t
            coefficients_cur = eta_t * coefficients_new + (1.0 - eta_t) * coefficients
            tempModel, tempObj, q = returnAlpha(h, numberOfBaseKernels, coefficients_cur, training_labels, training_basekernels,add_kernel, C, theta)

            iter_ls += 1

        if iter_ls == MAX_ITER_LS:
            break

        coefficients = coefficients_cur
        objectives.append(tempObj)
        finalSVMmodel = tempModel

        if abs(objectives[-1] - objectives[-2]) <= tau:
            break

    return coefficients, finalSVMmodel, objectives

def returnAlpha(h, n_baseKernels, coefficients, labels, baseKernels, add_kernel, C, theta):

    trainingKernels = returnKernel(coefficients, baseKernels, add_kernel)
    SVMModel = SVC(kernel="precomputed")
    SVMModel.fit(trainingKernels, labels)

    SVs = SVMModel.support_
    dualCoeffs = SVMModel.dual_coef_

    alpha = [0] * len(labels)
    for i in range(dualCoeffs.shape[1]):
        temp = dualCoeffs[0][i]
        if temp < 0:
            temp = 0 - temp
        alpha[SVs[i]] = temp

    y_alpha = [alpha[i] * labels[i] for i in range(len(alpha))]

    q = np.zeros((n_baseKernels))
    for i in range(n_baseKernels):
        baseKernel = baseKernels[i]

        q[i] = 0.5 * np.dot(np.dot(np.transpose(y_alpha), baseKernel), y_alpha)

    h = h.reshape((1, n_baseKernels))
    coefficients = np.array(coefficients).reshape((n_baseKernels, 1))
    temp = 0.5 * (np.dot(h, coefficients)) ** 2

    q = q.reshape((1, n_baseKernels))
    y_alpha = np.array(y_alpha).reshape((1, len(labels)))
    obj = temp + theta * (sum(alpha) - np.dot(q, coefficients) - 0.5 * np.dot(np.dot(y_alpha, add_kernel), np.transpose(y_alpha)))

    return SVMModel, obj[0][0], q

def returnKernel(coefficients, baseKernels, add_kernel):

    resultKernel = coefficients[0] * baseKernels[0]

    for i in range(1, len(coefficients), 1):
        resultKernel += coefficients[i] * baseKernels[i]

    resultKernel += add_kernel

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

    aps = []
    scores = []
    for classNum in range(labels.shape[1]):
        decisionValues = np.zeros((labels.shape[0])).reshape(1, labels.shape[0]) # class * number of kernels
        thisClassLabels = labels[::, classNum]
        trainingLabels = [thisClassLabels[index] for index in trainingIndiceList]
        testingLabels = [thisClassLabels[index] for index in testingIndiceList]

        # MMD
        s = np.zeros((baseKernels[0].shape[0]))
        s[0:195] =  -1.0 / 195
        s[195:] = 1.0 / 906
        s = s.reshape((baseKernels[0].shape[0], 1))
        mmd_values = calculateMMD(baseKernels, s)

        # AMKL
        trainingBaseKernls = []
        testingBaseKernels = []

        for baseKernel in baseKernels:
            tempKernel = sliceArray(baseKernel, trainingIndiceList)
            trainingBaseKernls.append(tempKernel)

            tempKernel = baseKernel[np.ix_(testingIndiceList, trainingIndiceList)]
            testingBaseKernels.append(tempKernel)


        trainingFP = [0] * len(trainingIndiceList)
        testingFP = [0] * len(testingIndiceList)

        coefficients, SVMmodel, objectives = train_amkl(trainingLabels, trainingBaseKernls, mmd_values, trainingFP, 20.0, 1)
        test_addKernel = np.dot(np.array(testingFP).reshape((len(testingIndiceList), 1)), np.array(trainingFP).reshape((1, len(trainingIndiceList)))) / 20.0
        testKernels = returnKernel(coefficients, testingBaseKernels, test_addKernel)

        testScores = SVMmodel.decision_function(testKernels)
        scores.append(testScores.flatten())

    # Find the label with the largest score
    scores = np.vstack(scores)
    ranks = np.argmax(scores, axis=0)

    labelSet = ["birthday", "parade", "picnic", "show", "sports", "wedding"]
    predictLabels = [labelSet[i] for i in ranks]

    return predictLabels


def SVM_One_vs_One(distances, semanticLabels, auxiliaryIndices, targetTrainIndices, targetTestIndices):

    combinedTrainIndces = auxiliaryIndices + targetTrainIndices

    # Construct base kernels
    baseKernels = []
    for i in range(len(distances)):
        distance = distances[i]
        distance = distance ** 2
        trainingDistances = sliceArray(distance, combinedTrainIndces)

        # Define kernel parameters
        gramma0 = 1.0 / np.mean(trainingDistances)
        kernel_params = [gramma0 *(2 ** index) for index in range(-3, 2, 1)]

        # Construct base kernels & pre-learned classifier
        baseKernel = constructBaseKernels(["rbf", "lap", "isd","id"], kernel_params, distance)
        baseKernels.append(baseKernel)

    # Build a classifier for each pair
    labelSet = ["birthday", "parade", "picnic", "show", "sports", "wedding"]
    pairs = util.paris(labelSet)

    stackOfPredictions = []
    for pair in pairs:

        positiveClass = pair[0]
        negativeClass = pair[1]

        thisTargetTrainIndices = []
        thisTargetBinaryLabels = []

        for i in targetTrainIndices:
            if semanticLabels[i] in pair:
                thisTargetTrainIndices.append(i)

                if semanticLabels[i] == positiveClass:
                    thisTargetBinaryLabels.append(1)
                elif semanticLabels[i] == negativeClass:
                    thisTargetBinaryLabels.append(-1)

        thisAuxiIndices = []
        thisAuxiBinaryLabels = []
        for i in auxiliaryIndices:
            if semanticLabels[i] in pair:
                thisAuxiIndices.append(i)

                if semanticLabels[i] == positiveClass:
                    thisAuxiBinaryLabels.append(1)
                elif semanticLabels[i] == negativeClass:
                    thisAuxiBinaryLabels.append(-1)

        thisCombinedTrain = thisTargetTrainIndices + thisAuxiIndices
        thisCombinedLabel = thisTargetBinaryLabels + thisAuxiBinaryLabels

        # MMD
        AllKernels = []
        for setKernels in baseKernels:
            AllKernels += setKernels

        s = np.zeros((AllKernels[0].shape[0]))
        numAuxiliaryDomain = len(thisAuxiIndices)
        numTargetDomain = len(thisTargetTrainIndices + targetTestIndices)

        for i in thisAuxiIndices:
            s[i] = - 1.0 / numAuxiliaryDomain
        for i in thisTargetTrainIndices + targetTestIndices:
            s[i] = 1.0 / numTargetDomain

        s = s.reshape((AllKernels[0].shape[0], 1))
        mmd_values = calculateMMD(AllKernels, s)

        # A-MKL
        trainingBaseKernels = []
        testingBaseKernels = []

        for setOfBaseKernel in baseKernels:
            for baseKernel in setOfBaseKernel:

                tempKernel = baseKernel[np.ix_(thisCombinedTrain, thisCombinedTrain)]
                trainingBaseKernels.append(tempKernel)

                tempKernel = baseKernel[np.ix_(targetTestIndices, thisCombinedTrain )]
                testingBaseKernels.append(tempKernel)

        trainingFP = [0] * len(thisCombinedTrain)
        testingFP = [0] * len(targetTestIndices)

        coefficients, SVMmodel, objectives = train_amkl(thisCombinedLabel, trainingBaseKernels, mmd_values, trainingFP, 20.0, 1)
        test_addKernel = np.dot(np.array(testingFP).reshape((len(targetTestIndices), 1)), np.array(trainingFP).reshape((1, len(thisCombinedTrain)))) / 20.0
        testKernels = returnKernel(coefficients, testingBaseKernels, test_addKernel)

        testScores = SVMmodel.decision_function(testKernels)
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

        targetTrainingIndices = []
        auxiliaryTrainingIndices = []
        for i in trainingIndiceList:
            if i <= 194:
                targetTrainingIndices.append(i)
            else:
                auxiliaryTrainingIndices.append(i)

        predictions = SVM_One_vs_All(distances, labels, trainingIndiceList, testingIndiceList)

        # calculate prediction accuracy
        testTrueLabels = [trueLabels[i] for i in testingIndiceList]
        predictions = np.array(predictions)
        testTrueLabels = np.array(testTrueLabels)

        print "Multi-class Accuracy (One vs All): " + str(util.evaluateAccuracy(predictions, testTrueLabels))


        predictions = SVM_One_vs_One(distances, trueLabels, auxiliaryTrainingIndices, targetTrainingIndices, testingIndiceList)
        predictions = np.array(predictions)
        print "Multi-class Accuracy (One vs One): " + str(util.evaluateAccuracy(predictions, testTrueLabels))

        print ""
