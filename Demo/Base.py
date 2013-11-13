__author__ = 'GongLi'

from SimpleSVM import runSVM_T
import random
from scipy.io import loadmat
import Utility as util
import numpy as np
import A_MKL
from sklearn.svm import SVC


# This method gives out the indices of Youtube videos for training and testing according to numberFromEachClass
def randomYoutubeIndices(numberFromEachClass):

    betterIndices = util.loadObject("correctVideos.pkl")

    assert numberFromEachClass < 80
    birthday = [i for i in range(195, 346) if i in betterIndices]
    parade = [i for i in range(346, 465) if i in betterIndices]
    picnic = [i for i in range(465, 550) if i in betterIndices]
    show = [i for i in range(550, 750) if i in betterIndices]
    sports = [i for i in range(750, 1010) if i in betterIndices]
    wedding = [i for i in range(1010, 1101) if i in betterIndices]

    trainIndices = []
    trainIndices += random.sample(birthday, numberFromEachClass)
    trainIndices += random.sample(parade, numberFromEachClass)
    trainIndices += random.sample(picnic, numberFromEachClass)
    trainIndices += random.sample(show, numberFromEachClass)
    trainIndices += random.sample(sports, numberFromEachClass)
    trainIndices += random.sample(wedding, numberFromEachClass)

    testIndices = [i for i in betterIndices]
    for i in trainIndices:
        testIndices.remove(i)

    return trainIndices, testIndices

# This method classifies Youtube videos using SVM_T
def simpleYoutubeClassification(distances, YoutubeTrainIndices, YoutubeTestIndices):

    binaryLabels = loadmat("/Users/GongLi/PycharmProjects/DomainAdaption/labels.mat")['labels']
    aps, predictions = runSVM_T(distances, binaryLabels, YoutubeTrainIndices, YoutubeTestIndices)

    trueLabels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/All/Normal/all_labels_Level0.pkl")
    originalLabels = [trueLabels[i] for i in YoutubeTestIndices]

    # Give out accuracy
    predictions = np.array(predictions)
    originalLabels = np.array(originalLabels)

    correctIndices = []
    for i in range(len(predictions)):
        if predictions[i] == originalLabels[i]:
            correctIndices.append(YoutubeTestIndices[i])

    util.storeObject("one.pkl", correctIndices)


    return str(util.evaluateAccuracy(predictions, originalLabels))

# This method classifies Youtube videos using A-MKL
def adaptiveMKLYoutubeClassification(distances, YoutubeTrainIndices, YoutubeTestIndices):

    binaryLabels = loadmat("/Users/GongLi/PycharmProjects/DomainAdaption/labels.mat")['labels']
    aps, predictions = AdaptiveMKL(distances, binaryLabels, [i for i in range(0,195)], YoutubeTrainIndices, YoutubeTestIndices)

    trueLabels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/All/Normal/all_labels_Level0.pkl")
    originalLabels = [trueLabels[i] for i in YoutubeTestIndices]

    # Give out accuracy
    predictions = np.array(predictions)
    originalLabels = np.array(originalLabels)

    return str(util.evaluateAccuracy(predictions, originalLabels))


# This method rewrites A-MKL
def AdaptiveMKL(distances, binaryLabels, auxiliaryIndices, targetTrainIndices, targetTestIndices):

    combinedTrainIndices = auxiliaryIndices + targetTrainIndices
    # base kernels
    baseKernels = []
    for i in range(len(distances)):
        distance = distances[i]
        distance = distance ** 2
        trainingDistances = distance[np.ix_(combinedTrainIndices, combinedTrainIndices)]

        gamma = 1.0 / np.mean(trainingDistances)
        kernel_params = [gamma * (2 ** index) for index in range(-3, 2, 1)]

        baseKernel = A_MKL.constructBaseKernels(["rbf", "lap", "isd","id"], kernel_params, distance)
        baseKernels.append(baseKernel)

    aps = []
    scores = []
    for classNum in range(binaryLabels.shape[1]):
        thisClassLabels = binaryLabels[::, classNum]
        trainingLabels = [thisClassLabels[index] for index in combinedTrainIndices]
        testingLabels = [thisClassLabels[index] for index in targetTestIndices]

        # pre-learned classifiers
        all_fp = []
        for m in range(len(baseKernels)):
            setOfBaseKernels = baseKernels[m]

            decisionValues = []
            for baseKernel in setOfBaseKernels:
                Ktrain = baseKernel[np.ix_(combinedTrainIndices, combinedTrainIndices)]
                Ktest = baseKernel[::, combinedTrainIndices]

                clf = SVC(kernel="precomputed")
                clf.fit(Ktrain, trainingLabels)

                dv = clf.decision_function(Ktest)
                dv = dv.reshape(1, Ktest.shape[0])
                decisionValues.append(dv)

            decisionValues = np.vstack(decisionValues)
            averageDV = np.mean(decisionValues, axis = 0)
            all_fp.append(averageDV)

        all_fp = np.vstack(all_fp)

        # MMD
        AllKernels = []
        for setKernels in baseKernels:
            AllKernels += setKernels

        s = np.zeros((AllKernels[0].shape[0]))
        numAuxiliaryDomain = len(auxiliaryIndices)
        targetIndices = targetTestIndices + targetTrainIndices
        numTargetDomain = len(targetTrainIndices)

        for i in auxiliaryIndices:
            s[i] = -1.0 / numAuxiliaryDomain

        for i in targetIndices:
            s[i] = 1.0 / numTargetDomain

        s = s.reshape((AllKernels[0].shape[0], 1))
        mmd_values = A_MKL.calculateMMD(AllKernels, s)

        # AMKL
        trainingBaseKernels = []
        testingBaseKernels = []

        for setOfBaseKernel in baseKernels:
            for baseKernel in setOfBaseKernel:

                tempKernel = baseKernel[np.ix_(combinedTrainIndices, combinedTrainIndices)]
                trainingBaseKernels.append(tempKernel)

                tempKernel = baseKernel[np.ix_(targetTestIndices, combinedTrainIndices)]
                testingBaseKernels.append(tempKernel)

        sizeOfDistances = len(baseKernels)
        rows = [i for i in range(sizeOfDistances)]
        trainingFP = all_fp[np.ix_(rows, combinedTrainIndices)]
        testingFP = all_fp[np.ix_(rows, targetTestIndices)]

        coefficients, SVMmodel, objectives = A_MKL.train_amkl(trainingLabels, trainingBaseKernels, mmd_values, trainingFP, 20.0, 1)
        test_addKernel = np.dot(np.transpose(testingFP), trainingFP) / 20.0
        testKernels = A_MKL.returnKernel(coefficients, testingBaseKernels, test_addKernel)

        testScores = SVMmodel.decision_function(testKernels)
        AP = util.averagePrecision(testScores, testingLabels)

        aps.append(AP)
        scores.append(testScores.flatten())

    # Find the label with the largest score
    scores = np.vstack(scores)
    ranks = np.argmax(scores, axis=0)

    labelSet = ["birthday", "parade", "picnic", "show", "sports", "wedding"]
    predictLabels = [labelSet[i] for i in ranks]

    return aps, predictLabels

if __name__ == "__main__":


    distanceOne = loadmat("/Users/GongLi/PycharmProjects/DomainAdaption/dist_SIFT_L0.mat")['distMat']
    distanceTwo = loadmat("/Users/GongLi/PycharmProjects/DomainAdaption/dist_SIFT_L1.mat")['distMat']
    distanceThree = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc1000/All/GMM_ALL_Distance.pkl")

    distances = []
    distances.append(distanceOne)
    distances.append(distanceTwo)
    # distances.append(distanceThree)


    for i in range(5):

        trainIndices, testIndices = randomYoutubeIndices(10)
        print "SVM_T: " + simpleYoutubeClassification(distances, trainIndices, testIndices)
        print "SVM_AT: " + simpleYoutubeClassification(distances, trainIndices + [i for i in range(0, 195)], testIndices)
        #
        print "A-MKL: " + adaptiveMKLYoutubeClassification(distances, trainIndices, testIndices)

        print " "


        indiceOne = util.loadObject("correctVideos.pkl")

        indiceTwo = util.loadObject("one.pkl")

        for i in indiceTwo:
            if i not in indiceOne:
                print i
