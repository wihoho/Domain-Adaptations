__author__ = 'GongLi'

import Utility as util
import numpy as np

# originalDistance = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/Kodak_distanceMatrix_version2.pkl")
# currentDistance = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/DifferentWeight/txxDistance.pkl")



ghtLabels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/DifferentWeight/ghtLabels.pkl")
tfxLabels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/DifferentWeight/tfxLabels.pkl")
txxLabels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/DifferentWeight/txxLabels.pkl")

originalLabels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/KodakLabelsLevel0.pkl")

if ghtLabels == tfxLabels == txxLabels == originalLabels:
    print "Great!!"