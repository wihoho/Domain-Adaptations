from sklearn.metrics import average_precision_score
import numpy as np

score = np.array([0.8, 0.4, 0.35, 0.1])
label = np.array([1, 0 , 1, 0])

ap = average_precision_score(label, score)

print ap
