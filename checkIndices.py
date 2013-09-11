__author__ = 'GongLi'


from scipy.io import loadmat
import pickle

distanceOne = loadmat("dist_SIFT_L0.mat")['distMat']
labels = loadmat("labels.mat")['labels']


file = open("LevelZero/all_labels_Level0.pkl", "rb")
myLabels = pickle.load(file)

birthdayList = [i for i in range(len(myLabels)) if myLabels[i] == "birthday"]
paradeList = [i for i in range(len(myLabels)) if myLabels[i] == "parade"]
showList = [i for i in range(len(myLabels)) if myLabels[i] == "show"]
sportsList = [i for i in range(len(myLabels)) if myLabels[i] == "sports"]
weddingList = [i for i in range(len(myLabels)) if myLabels[i] == "wedding"]
picnicList = [i for i in range(len(myLabels)) if myLabels[i] == "picnic"]


print "Yes"

