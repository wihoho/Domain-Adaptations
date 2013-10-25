__author__ = 'GongLi'

from scipy.io import loadmat
import AMKL
import DTMKL
import FR
import MKL
import SVM_AT
import SVM_T
import AdaptiveSVM
import numpy as np
import xlwt
import time
import Utility as util

def evaluate(distanceOne, excelName, labels):

    SVM_T__aps = []
    SVM_AT_aps = []
    FR_aps = []
    A_SVM_aps = []
    MKL_aps = []
    DT_SVM_aps = []
    A_MKL_aps = []

    runningTimeWb = xlwt.Workbook()
    runningTimeWs = runningTimeWb.add_sheet("Running Time")

    runningTimeWs.write(0,0, "SVM_T")
    runningTimeWs.write(1,0, "SVM_AT")
    runningTimeWs.write(2,0, "FR")
    runningTimeWs.write(3,0, "A_SVM")
    runningTimeWs.write(4,0, "MKL")
    runningTimeWs.write(5,0, "DTSVM")
    runningTimeWs.write(6,0, "A_MKL")

    runningTimeIndex = 1

    for i in range(1,6,1):
        print excelName+": "+ str(i)

        trainingIndices = loadmat(str(i)+".mat")['training_ind']
        trainingIndiceList = []
        testingIndices = loadmat(str(i)+".mat")['test_ind']
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

        startTime = time.time()
        aps = SVM_T.runSVM_T(distanceOne, labels, targetTrainingIndices, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(0, runningTimeIndex, endTime - startTime)
        SVM_T__aps.append(aps)

        startTime = time.time()
        aps = SVM_AT.runSVM_AT(distanceOne, labels, auxiliaryTrainingIndices, targetTrainingIndices, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(1, runningTimeIndex, endTime - startTime)
        SVM_AT_aps.append(aps)

        startTime = time.time()
        aps = FR.runSVM_AT(distanceOne, labels, auxiliaryTrainingIndices, targetTrainingIndices, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(2, runningTimeIndex, endTime - startTime)
        FR_aps.append(aps)

        startTime = time.time()
        aps = AdaptiveSVM.runASVM(distanceOne, labels, auxiliaryTrainingIndices, targetTrainingIndices, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(3, runningTimeIndex, endTime - startTime)
        A_SVM_aps.append(aps)

        startTime = time.time()
        aps = MKL.runMKL(distanceOne, labels, trainingIndiceList, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(4, runningTimeIndex, endTime - startTime)
        MKL_aps.append(aps)

        startTime = time.time()
        aps = DTMKL.runDTMKL(distanceOne, labels, trainingIndiceList, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(5, runningTimeIndex, endTime - startTime)
        DT_SVM_aps.append(aps)

        startTime = time.time()
        aps = AMKL.runAMKL(distanceOne, labels, trainingIndiceList, testingIndiceList)
        endTime = time.time()
        runningTimeWs.write(6, runningTimeIndex, endTime - startTime)
        A_MKL_aps.append(aps)

        runningTimeIndex += 1


    runningTimeWb.save("RunningTime.xls")
    # Write into excel file
    wb = xlwt.Workbook()
    ws = wb.add_sheet("SVM_T")
    SVM_T_apsArray = np.array(SVM_T__aps)
    for i in range(SVM_T_apsArray.shape[0]):
        for j in range(SVM_T_apsArray.shape[1]):
            ws.write(i, j, SVM_T_apsArray[i][j])

    meanAP = np.mean(SVM_T_apsArray)
    rowMean = np.mean(SVM_T_apsArray, axis=1)
    sd = np.std(rowMean)
    ws.write(i+1, 0, meanAP)
    ws.write(i+1, 1, sd)

    print "SVM_T: " +str(meanAP)

    ws = wb.add_sheet("SVM_AT")
    tempArray = np.array(SVM_AT_aps)
    for i in range(tempArray.shape[0]):
        for j in range(tempArray.shape[1]):
            ws.write(i, j, tempArray[i][j])

    meanAP = np.mean(tempArray)
    rowMean = np.mean(tempArray, axis=1)
    sd = np.std(rowMean)
    ws.write(i+1, 0, meanAP)
    ws.write(i+1, 1, sd)

    print "SVM_AT: " +str(meanAP)


    ws = wb.add_sheet("FR")
    tempArray = np.array(FR_aps)
    for i in range(tempArray.shape[0]):
        for j in range(tempArray.shape[1]):
            ws.write(i, j, tempArray[i][j])

    meanAP = np.mean(tempArray)
    rowMean = np.mean(tempArray, axis=1)
    sd = np.std(rowMean)
    ws.write(i+1, 0, meanAP)
    ws.write(i+1, 1, sd)

    print "FR: " +str(meanAP)

    ws = wb.add_sheet("A-SVM")
    tempArray = np.array(A_SVM_aps)
    for i in range(tempArray.shape[0]):
        for j in range(tempArray.shape[1]):
            ws.write(i, j, tempArray[i][j])

    meanAP = np.mean(tempArray)
    rowMean = np.mean(tempArray, axis=1)
    sd = np.std(rowMean)
    ws.write(i+1, 0, meanAP)
    ws.write(i+1, 1, sd)

    print "A-SVM: " +str(meanAP)


    ws = wb.add_sheet("MKL")
    tempArray = np.array(MKL_aps)
    for i in range(tempArray.shape[0]):
        for j in range(tempArray.shape[1]):
            ws.write(i, j, tempArray[i][j])

    meanAP = np.mean(tempArray)
    rowMean = np.mean(tempArray, axis=1)
    sd = np.std(rowMean)
    ws.write(i+1, 0, meanAP)
    ws.write(i+1, 1, sd)

    print "MKL: " +str(meanAP)


    ws = wb.add_sheet("DTSVM")
    tempArray = np.array(DT_SVM_aps)
    for i in range(tempArray.shape[0]):
        for j in range(tempArray.shape[1]):
            ws.write(i, j, tempArray[i][j])

    meanAP = np.mean(tempArray)
    rowMean = np.mean(tempArray, axis=1)
    sd = np.std(rowMean)
    ws.write(i+1, 0, meanAP)
    ws.write(i+1, 1, sd)

    print "DTSVM: " +str(meanAP)


    ws = wb.add_sheet("A-MKL")
    tempArray = np.array(A_MKL_aps)
    for i in range(tempArray.shape[0]):
        for j in range(tempArray.shape[1]):
            ws.write(i, j, tempArray[i][j])

    meanAP = np.mean(tempArray)
    rowMean = np.mean(tempArray, axis=1)
    sd = np.std(rowMean)
    ws.write(i+1, 0, meanAP)
    ws.write(i+1, 1, sd)

    print "A-MKL: " +str(meanAP)

    wb.save(excelName)

if __name__ == "__main__":

    GmmDis = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/GMM/Spherical Covariance/All 128/All_GMM_distances.pkl")
    EMDNormalDis = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/All/Normal/all_DistanceMatrix_Level0.pkl")
    EMDSoftWeightDis = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/All/SoftWeight/all_softWeight_distance.pkl")
    Tfc_Level0 = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/All/tfc_all_distance.pkl")


    EMDLevelOneUnalignedDis = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelOne/Unaligned_LevelOne_All_Distance.pkl")
    EMDLevelOneAlignedDis = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelOne/Aligned_LevelOne_All_Distance.pkl")

    Voc1000_GaussianAssignment = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc1000/All/GMM_ALL_Distance.pkl")
    Voc1000_LevelZero = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc1000/All/LevelZero_ALL_Distance.pkl")

    tempLabels = loadmat("labels.mat")['labels']


    distances = []
    # distances.append(Voc1000_GaussianAssignment)
    # distances.append(EMDLevelOneAlignedDis)
    # distances.append(Voc1000_LevelZero)
    distances.append(Tfc_Level0)

    evaluate(distances, "tfcHistogramsLevel0.xls", tempLabels)








