__author__ = 'GongLi'

import random
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

def getTrainVideos(num):

    birthdayIndices = [i for i in range(195, 346, 1)]
    paradeIndices = [i for i in range(346, 465, 1)]
    picnicIndices = [ i for i in range(465, 550, 1)]
    showIndices = [i for i in range(550, 750, 1)]
    sportsIndices = [i for i in range(750, 1010, 1)]
    weddingIndices = [i for i in range(1010, 1101, 1)]

    trainIndices = []
    trainIndices.append(random.sample(birthdayIndices, num))
    trainIndices.append(random.sample(paradeIndices, num))
    trainIndices.append(random.sample(picnicIndices, num))
    trainIndices.append(random.sample(showIndices, num))
    trainIndices.append(random.sample(sportsIndices, num))
    trainIndices.append(random.sample(weddingIndices, num))

    return trainIndices


def getTestVideos(trainIndices, num):

    birthdayIndices = [i for i in range(195, 346, 1)]
    paradeIndices = [i for i in range(346, 465, 1)]
    picnicIndices = [ i for i in range(465, 550, 1)]
    showIndices = [i for i in range(550, 750, 1)]
    sportsIndices = [i for i in range(750, 1010, 1)]
    weddingIndices = [i for i in range(1010, 1101, 1)]

    testIndices = []

    temp = birthdayIndices
    for i in trainIndices[0]:
        temp.remove(i)
    testIndices.append(random.sample(temp, num))


    temp = paradeIndices
    for i in trainIndices[1]:
        temp.remove(i)
    testIndices.append(random.sample(temp, num))

    temp = picnicIndices
    for i in trainIndices[2]:
        temp.remove(i)
    testIndices.append(random.sample(temp, num))

    temp = showIndices
    for i in trainIndices[3]:
        temp.remove(i)
    testIndices.append(random.sample(temp, num))

    temp = sportsIndices
    for i in trainIndices[4]:
        temp.remove(i)
    testIndices.append(random.sample(temp, num))

    temp = weddingIndices
    for i in trainIndices[5]:
        temp.remove(i)
    testIndices.append(random.sample(temp, num))

    return testIndices

def runSVM_T(distances, labels, targetTrainingIndice, targetTestingIndice):

    temp = []
    for i in targetTrainingIndice:
        for j in i:
            temp.append(j)
    targetTrainingIndice = temp

    temp = []
    for i in targetTestingIndice:
        for j in i:
            temp.append(j)
    targetTestingIndice = temp

    # Construct base kernels
    baseKernels = []
    for i in range(len(distances)):
        distance = distances[i]
        distance = distance ** 2
        trainingDistances = distance[np.ix_(targetTrainingIndice, targetTrainingIndice)]

        # Define kernel parameters
        gramma0 = 1.0 / np.mean(trainingDistances)
        # kernel_params = [gramma0 *(2 ** index) for index in range(-3, 2, 1)]
        kernel_params = []
        kernel_params.append(gramma0)

        # Construct base kernels & pre-learned classifier
        baseKernel = constructBaseKernels(["rbf", "lap", "isd","id"], kernel_params, distance)
        baseKernels += baseKernel

    TrainingLabels = [labels[index] for index in targetTrainingIndice]
    TestingLabels = [labels[index] for index in targetTestingIndice]

    for m in range(len(baseKernels)):
        baseKernel = baseKernels[m]

        Ktrain = baseKernel[np.ix_(targetTrainingIndice, targetTrainingIndice)]
        Ktest = baseKernel[np.ix_(targetTestingIndice , targetTrainingIndice)]

        clf = SVC(kernel="precomputed")
        clf.fit(Ktrain, TrainingLabels)

        prediction = clf.predict(Ktest)
        correct = sum(1.0 * (prediction == TestingLabels))
        accuracy = correct / len(TestingLabels)

        print str(accuracy)


if __name__ == "__main__":

    distance = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/All/Normal/all_DistanceMatrix_Level0.pkl")
    labels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/All/Normal/all_labels_Level0.pkl")
    distances = []
    distances.append(distance)


    trainIndices = getTrainVideos(80)
    testIndices = getTestVideos(trainIndices, 3)

    runSVM_T(distances, labels, trainIndices, testIndices)