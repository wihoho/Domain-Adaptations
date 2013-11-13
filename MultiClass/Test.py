import Utility as util
from scipy.io import loadmat
import numpy as np
import SVM_T
import SVM_AT
import AdaptiveSVM
import AMKL
import DTMKL
import FR
import MKL
import xlwt
import time

import random

# This method gives out the indices of Youtube videos for training and testing according to numberFromEachClass
def randomKodakIndices(numberFromEachClass):


    birthday = [i for i in range(0, 16)]
    parade = [i for i in range(16, 30)]
    picnic = [i for i in range(30, 36)]
    show = [i for i in range(36, 93)]
    sports = [i for i in range(93, 168)]
    wedding = [i for i in range(168, 195)]

    trainIndices = []
    trainIndices += random.sample(birthday, numberFromEachClass)
    trainIndices += random.sample(parade, numberFromEachClass)
    trainIndices += random.sample(picnic, numberFromEachClass)
    trainIndices += random.sample(show, numberFromEachClass)
    trainIndices += random.sample(sports, numberFromEachClass)
    trainIndices += random.sample(wedding, numberFromEachClass)

    testIndices = [i for i in range(195)]
    for k in trainIndices:
        testIndices.remove(k);


    return trainIndices, testIndices






if __name__ == "__main__":

    runningTimeWb = xlwt.Workbook()
    runningTimeWs = runningTimeWb.add_sheet("Running Time")

    runningTimeWs.write(0,0, "SVM_T (One vs All)")
    runningTimeWs.write(1,0, "SVM_T (One vs One)")

    runningTimeWs.write(2,0, "SVM_AT (One vs All)")
    runningTimeWs.write(3,0, "SVM_AT (One vs One)")

    runningTimeWs.write(4,0, "FR (One vs All)")
    runningTimeWs.write(5,0, "FR (One vs One)")

    runningTimeWs.write(6,0, "A_SVM (One vs All)")
    runningTimeWs.write(7,0, "A_SVM (One vs One)")

    runningTimeWs.write(8,0, "MKL (One vs All)")
    runningTimeWs.write(9,0, "MKL (One vs One)")

    runningTimeWs.write(10,0, "DTSVM (One vs All)")
    runningTimeWs.write(11,0, "DTSVM (One vs One)")

    runningTimeWs.write(12,0, "A_MKL (One vs All)")
    runningTimeWs.write(13,0, "A_MKL (One vs One)")


    accuracyWs = runningTimeWb.add_sheet("Accuracy")
    accuracyWs.write(0,0, "SVM_T (One vs All)")
    accuracyWs.write(1,0, "SVM_T (One vs One)")

    accuracyWs.write(2,0, "SVM_AT (One vs All)")
    accuracyWs.write(3,0, "SVM_AT (One vs One)")

    accuracyWs.write(4,0, "FR (One vs All)")
    accuracyWs.write(5,0, "FR (One vs One)")

    accuracyWs.write(6,0, "A_SVM (One vs All)")
    accuracyWs.write(7,0, "A_SVM (One vs One)")

    accuracyWs.write(8,0, "MKL (One vs All)")
    accuracyWs.write(9,0, "MKL (One vs One)")

    accuracyWs.write(10,0, "DTSVM (One vs All)")
    accuracyWs.write(11,0, "DTSVM (One vs One)")

    accuracyWs.write(12,0, "A_MKL (One vs All)")
    accuracyWs.write(13,0, "A_MKL (One vs One)")


    distanceThree = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc1000/All/GMM_ALL_Distance.pkl")

    distances = []
    distances.append(distanceThree)

    BinaryLabels = loadmat("/Users/GongLi/PycharmProjects/DomainAdaption/labels.mat")['labels']
    trueLabels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/All/Normal/all_labels_Level0.pkl")

    all_aps = []
    for iteration in range(1,6,1):



        trainingIndices = loadmat("/Users/GongLi/PycharmProjects/DomainAdaption/"+str(iteration)+".mat")['training_ind']
        trainingIndiceList = []
        testingIndices = loadmat("/Users/GongLi/PycharmProjects/DomainAdaption/"+str(iteration)+".mat")['test_ind']
        testingIndiceList = []

        # Construct indices
        for x in range(trainingIndices.shape[0]):
            trainingIndiceList.append(trainingIndices[x][0] - 1)

        for x in range(testingIndices.shape[1]):
            testingIndiceList.append(testingIndices[0][x] - 1)

        targetTrainingIndices = []
        auxiliaryTrainingIndices = []
        for x in trainingIndiceList:
            if x <= 194:
                targetTrainingIndices.append(x)
            else:
                auxiliaryTrainingIndices.append(x)


        # targetTrainingIndices, testingIndiceList = randomKodakIndices(3)
        # auxiliaryTrainingIndices = [i for i in range(195, 1101,1)]
        # trainingIndiceList = targetTrainingIndices + auxiliaryTrainingIndices


        # SVM_T
        startTime = time.time()
        predictions = SVM_T.runSVM_T_One_vs_All(distances, BinaryLabels, targetTrainingIndices, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(0, iteration, endTime - startTime)

        testTrueLabels = [trueLabels[i] for i in testingIndiceList]

        predictions = np.array(predictions)
        testTrueLabels = np.array(testTrueLabels)

        allAccuracy = util.evaluateAccuracy(predictions, testTrueLabels)
        print "SVM_T (One vs All): " + str(allAccuracy)
        accuracyWs.write(0, iteration, allAccuracy)

        startTime = time.time()
        predictions = SVM_T.runSVM_T_One_vs_One(distances, trueLabels, targetTrainingIndices, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(1, iteration, endTime - startTime)

        oneAccuracy = util.evaluateAccuracy(predictions, testTrueLabels)
        print "SVM_T (One vs One): " + str(oneAccuracy)
        accuracyWs.write(1,iteration,oneAccuracy)


        # SVM_AT
        startTime = time.time()
        predictions = SVM_AT.SVM_One_vs_All(distances, BinaryLabels, trainingIndiceList, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(2, iteration, endTime - startTime)

        testTrueLabels = [trueLabels[i] for i in testingIndiceList]

        predictions = np.array(predictions)
        testTrueLabels = np.array(testTrueLabels)

        allAccuracy = util.evaluateAccuracy(predictions, testTrueLabels)
        print "SVM_AT (One vs All): " + str(allAccuracy)
        accuracyWs.write(2,iteration,allAccuracy)

        startTime = time.time()
        predictions = SVM_AT.SVM_One_vs_One(distances, trueLabels, trainingIndiceList, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(3, iteration, endTime - startTime)

        oneAccuracy = util.evaluateAccuracy(predictions, testTrueLabels)
        print "SVM_AT (One vs Onee): " + str(oneAccuracy)
        accuracyWs.write(3,iteration,oneAccuracy)

        # FR
        startTime = time.time()
        predictions = FR.runSVM_T_One_vs_All (distances, BinaryLabels, trainingIndiceList, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(4, iteration, endTime - startTime)

        testTrueLabels = [trueLabels[i] for i in testingIndiceList]

        predictions = np.array(predictions)
        testTrueLabels = np.array(testTrueLabels)

        allAccuracy = util.evaluateAccuracy(predictions, testTrueLabels)
        print "FR (One vs All): " + str(allAccuracy)
        accuracyWs.write(4,iteration,allAccuracy)

        startTime = time.time()
        predictions = FR.SVM_One_vs_One(distances, trueLabels, trainingIndiceList, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(5, iteration, endTime - startTime)

        oneAccuracy = util.evaluateAccuracy(predictions, testTrueLabels)
        print "FR (One vs Onee): " + str(oneAccuracy)
        accuracyWs.write(5,iteration,oneAccuracy)

        # A-SVM
        startTime = time.time()
        predictions = AdaptiveSVM.SVM_One_vs_All(distances, BinaryLabels, auxiliaryTrainingIndices,targetTrainingIndices, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(6, iteration, endTime - startTime)

        testTrueLabels = [trueLabels[i] for i in testingIndiceList]

        predictions = np.array(predictions)
        testTrueLabels = np.array(testTrueLabels)

        allAccuracy = util.evaluateAccuracy(predictions, testTrueLabels)
        print "AdaptiveSVM (One vs All): " + str(allAccuracy)
        accuracyWs.write(6,iteration,allAccuracy)

        startTime = time.time()
        predictions = AdaptiveSVM.SVM_One_vs_One(distances, trueLabels, auxiliaryTrainingIndices,targetTrainingIndices, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(7, iteration, endTime - startTime)

        oneAccuracy = util.evaluateAccuracy(predictions, testTrueLabels)
        print "AdaptiveSVM (One vs Onee): " + str(oneAccuracy)
        accuracyWs.write(7,iteration,oneAccuracy)

        # MKL
        startTime = time.time()
        predictions = MKL.SVM_One_vs_All(distances, BinaryLabels, trainingIndiceList, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(8, iteration, endTime - startTime)

        testTrueLabels = [trueLabels[i] for i in testingIndiceList]

        predictions = np.array(predictions)
        testTrueLabels = np.array(testTrueLabels)

        allAccuracy = util.evaluateAccuracy(predictions, testTrueLabels)
        print "MKL (One vs All): " + str(allAccuracy)
        accuracyWs.write(8,iteration,allAccuracy)

        startTime = time.time()
        predictions = MKL.SVM_One_vs_One(distances, trueLabels, trainingIndiceList, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(9, iteration, endTime - startTime)

        oneAccuracy = util.evaluateAccuracy(predictions, testTrueLabels)
        print "MKL (One vs Onee): " + str(oneAccuracy)
        accuracyWs.write(9,iteration,oneAccuracy)

        # DTSVM
        startTime = time.time()
        predictions = DTMKL.SVM_One_vs_All(distances, BinaryLabels, trainingIndiceList, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(10, iteration, endTime - startTime)

        testTrueLabels = [trueLabels[i] for i in testingIndiceList]

        predictions = np.array(predictions)
        testTrueLabels = np.array(testTrueLabels)

        allAccuracy = util.evaluateAccuracy(predictions, testTrueLabels)
        print "DTMKL (One vs All): " + str(allAccuracy)
        accuracyWs.write(10,iteration,allAccuracy)

        startTime = time.time()
        predictions = DTMKL.SVM_One_vs_One(distances, trueLabels, auxiliaryTrainingIndices,targetTrainingIndices, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(11, iteration, endTime - startTime)

        oneAccuracy = util.evaluateAccuracy(predictions, testTrueLabels)
        print "DTMKL (One vs Onee): " + str(oneAccuracy)
        accuracyWs.write(11,iteration,oneAccuracy)

        # A-MKL
        startTime = time.time()
        predictions = AMKL.SVM_One_vs_All(distances, BinaryLabels, auxiliaryTrainingIndices,targetTrainingIndices, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(12, iteration, endTime - startTime)

        testTrueLabels = [trueLabels[i] for i in testingIndiceList]

        predictions = np.array(predictions)
        testTrueLabels = np.array(testTrueLabels)

        allAccuracy = util.evaluateAccuracy(predictions, testTrueLabels)
        print "AMKL (One vs All): " + str(allAccuracy)
        accuracyWs.write(12,iteration,allAccuracy)

        startTime = time.time()
        predictions = AMKL.SVM_One_vs_One(distances, trueLabels, auxiliaryTrainingIndices,targetTrainingIndices, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(13, iteration, endTime - startTime)

        oneAccuracy = util.evaluateAccuracy(predictions, testTrueLabels)
        print "AMKL (One vs Onee): " + str(oneAccuracy)
        accuracyWs.write(13,iteration,oneAccuracy)

    runningTimeWb.save("Result.xls")
