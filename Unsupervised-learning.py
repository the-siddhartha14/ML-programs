
import numpy as np
from sklearn.cluster import KMeans

X = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11]
])

model = KMeans(n_clusters=2)

model.fit(X)

labels = model.labels_

print("Cluster Labels:", labels)
