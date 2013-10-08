import numpy as np
import Utility as util

score = np.array([0, 1, -1, 0, 0])
label = np.array([-1, 1, 1, -1, -1])

ap = util.averagePrecision(score, label)

print ap
