__author__ = 'GongLi'

import evaluateVoc1000
import Utility as util

if __name__ == "__main__":

    KodakLevelZeros = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelZero/Kodak/Normal/Kodak_distanceMatrix_version2.pkl")
    KodakLevelOneUnaligned = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelOne/Kodak/KodakDistanceLevelOneUnAligned.pkl")
    KodakLevelOneAligned = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc2500/LevelOne/Kodak/KodakDistanceLevelOneAligned.pkl")

    meanAP, std = evaluateVoc1000.predefinedIndices(KodakLevelZeros)
    print "L0\t" +   str(meanAP) +u" \u00B1 "+str(std)

    meanAP, std = evaluateVoc1000.predefinedIndices(KodakLevelOneUnaligned)
    print "L1 Unaligned\t" +   str(meanAP) +u" \u00B1 "+str(std)

    meanAP, std = evaluateVoc1000.predefinedIndices(KodakLevelOneAligned)
    print "L1 Aligned\t" +   str(meanAP) +u" \u00B1 "+str(std)