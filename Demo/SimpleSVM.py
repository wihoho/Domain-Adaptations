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

def runSVM_T(distances, labels, targetTrainingIndice, targetTestingIndice):

    # Construct base kernels
    baseKernels = []
    for i in range(len(distances)):
        distance = distances[i]
        distance = distance ** 2
        trainingDistances = sliceArray(distance, targetTrainingIndice)

        # Define kernel parameters
        gramma0 = 1.0 / np.mean(trainingDistances)
        kernel_params = [gramma0 *(2 ** index) for index in range(-3, 2, 1)]

        # Construct base kernels & pre-learned classifier
        baseKernel = constructBaseKernels(["rbf", "lap", "isd","id"], kernel_params, distance)
        baseKernels += baseKernel

    aps = []
    scores = []
    for classNum in range(labels.shape[1]):
        thisClassLabels = labels[::, classNum]
        TrainingLabels = [thisClassLabels[index] for index in targetTrainingIndice]
        testingLabels = [thisClassLabels[index] for index in targetTestingIndice]

        finalTestScores = np.zeros((len(targetTestingIndice))).reshape((1, len(targetTestingIndice)))


        for m in range(len(baseKernels)):
            baseKernel = baseKernels[m]
            Ktrain = sliceArray(baseKernel, targetTrainingIndice)
            Ktest = baseKernel[np.ix_(targetTestingIndice , targetTrainingIndice)]

            clf = SVC(kernel="precomputed")
            clf.fit(Ktrain, TrainingLabels)

            dv = clf.decision_function(Ktest)
            finalTestScores = np.vstack((finalTestScores, dv.reshape((1, len(targetTestingIndice)))))

        # Fuse final scores together
        finalTestScores = finalTestScores[1:]
        tempFinalTestScores = 1.0 / (1 + math.e **(-finalTestScores))
        finalTestScores = np.mean(tempFinalTestScores, axis = 0)

        AP = util.averagePrecision(finalTestScores, testingLabels)
        aps.append(AP)

        scores.append(finalTestScores)

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
    # distances.append(distanceOne)
    # distances.append(distanceTwo)
    distances.append(distanceThree)
    labels = loadmat("/Users/GongLi/PycharmProjects/DomainAdaption/labels.mat")['labels']

    trueLabels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/All/Normal/all_labels_Level0.pkl")

    all_aps = []
    for i in range(1,6,1):
        trainingIndices = loadmat("/Users/GongLi/PycharmProjects/DomainAdaption/"+str(i)+".mat")['training_ind']
        trainingIndiceList = []
        testingIndices = loadmat("/Users/GongLi/PycharmProjects/DomainAdaption/"+str(i)+".mat")['test_ind']
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

        aps, predictions = runSVM_T(distances, labels, targetTrainingIndices, testingIndiceList)
        all_aps.append(aps)


        # calculate prediction accuracy
        testTrueLabels = [trueLabels[i] for i in testingIndiceList]

        predictions = np.array(predictions)
        testTrueLabels = np.array(testTrueLabels)

        print "Multi-class Accuracy: " + str(util.evaluateAccuracy(predictions, testTrueLabels))

    all_aps = np.array(all_aps)
    meanAP = np.mean(all_aps)

    rowMean = np.mean(all_aps, axis=1)
    sd = np.std(rowMean)

    print "meanAP: "+str(meanAP)
    print "standard deviation: "+str(sd)

