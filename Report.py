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

def runSVM_T(distances, labels, targetTrainingIndice, targetTestingIndice, kernels):

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
        baseKernel = constructBaseKernels(kernels, kernel_params, distance)
        baseKernels += baseKernel

    aps = []
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

    return aps


def evaluate(distances, labels, kernels):

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
    # ["rbf", "lap", "isd","id"]
        aps = runSVM_T(distances, labels, targetTrainingIndices, testingIndiceList, kernels)
        all_aps.append(aps)

    all_aps = np.array(all_aps)
    meanAP = np.mean(all_aps)

    rowMean = np.mean(all_aps, axis=1)
    sd = np.std(rowMean)

    print str(kernels) +": "+ str(meanAP) +u"\u00B1"+ str(sd)

if __name__ == "__main__":

    labels = loadmat("labels.mat")['labels']

    kernelChoices = []
    kernelChoices.append(["rbf"])
    kernelChoices.append(["lap"])
    kernelChoices.append(["isd"])
    kernelChoices.append(["id"])
    kernelChoices.append(["rbf", "lap", "isd","id"])



    spherical128 = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/GMM/Spherical Covariance/All 128/All_GMM_distances.pkl")
    spherical_64 = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/GMM/Spherical Covariance/PCA64_Spherical_GMM_n_iteration50_KodakDistances.pkl")
    full_128 = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/GMM/Full Covariance/128_Full_GMM_n_iteration50_KodakDistances.pkl")
    full_64 = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/GMM/Full Covariance/PCA64_FUll_GMM_n_iteration50_KodakDistances.pkl")

    candidates_dis = []
    candidates_dis.append(spherical128)
    candidates_dis.append(spherical_64)
    candidates_dis.append(full_128)
    candidates_dis.append(full_64)


    for dis in candidates_dis:

        distances = []
        distances.append(dis)

        for k in kernelChoices:
            evaluate(distances, labels, k)

        print "-------------------------------"
        print " "
