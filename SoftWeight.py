__author__ = 'GongLi'


if __name__ == "__main__":
    import SVM_T
    import Utility as util

    originalKodakDistances = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/Kodak_distanceMatrix_version2.pkl")
    originalLabels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/LevelZero/Kodak_labels_version2.pkl")

    softWeightKodakDistance = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/NewMethods/KodakSoftWeight_DistanceMatrix_Level0.pkl")
    softWeightLabels = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/NewMethods/KodakSoftWeight_labels_Level0.pkl")

    # Check whether these two labels are the same
    if originalLabels == softWeightLabels:
        print "Labels are the same"

    # Perform classification
    




