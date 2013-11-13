__author__ = 'GongLi'

import Utility as util
import TestAll
from scipy.io import loadmat


if __name__ == "__main__":

    # KodakLevelZeros = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/Kodak/Normal/Kodak_distanceMatrix_version2.pkl")
    # KodakLevelOneUnaligned = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelOne/Kodak/KodakDistanceLevelOneUnAligned.pkl")
    # KodakLevelOneAligned = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelOne/Kodak/KodakDistanceLevelOneAligned.pkl")
    #
    # meanAP, std = evaluateVoc1000.predefinedIndices(KodakLevelZeros)
    # print "L0\t" +   str(meanAP) +u" \u00B1 "+str(std)
    #
    # meanAP, std = evaluateVoc1000.predefinedIndices(KodakLevelOneUnaligned)
    # print "L1 Unaligned\t" +   str(meanAP) +u" \u00B1 "+str(std)
    #
    # meanAP, std = evaluateVoc1000.predefinedIndices(KodakLevelOneAligned)
    # print "L1 Aligned\t" +   str(meanAP) +u" \u00B1 "+str(std)



    LevelZero = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/All/Normal/all_DistanceMatrix_Level0.pkl")
    unalignedLevelOne = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelOne/Unaligned_LevelOne_All_Distance.pkl")
    alignedLevelOne = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelOne/Aligned_LevelOne_All_Distance.pkl")

    tempLabels = loadmat("labels.mat")['labels']

    distances0 = []
    distances0.append(LevelZero)
    TestAll.evaluate(distances0, "2013.10.17/LevelZero.xls", tempLabels)

    unaligned_distances = []
    unaligned_distances.append(unalignedLevelOne)
    TestAll.evaluate(unaligned_distances, "2013.10.17/Unaligned_LevelOne.xls", tempLabels)

    aligned_distances = []
    aligned_distances.append(alignedLevelOne)
    TestAll.evaluate(aligned_distances, "2013.10.17/Aligned_LevelOne.xls", tempLabels)


