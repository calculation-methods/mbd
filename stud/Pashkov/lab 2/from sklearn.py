from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

dataset = make_blobs(n_samples=200, centers = 4,n_features = 2, cluster_std = 1.6, random_state = 50)
print (dataset)
points = dataset[0]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
clusters = kmeans.cluster_centers_
kmeans.fit(points)

plt.scatter(dataset[0][:,0],dataset[0][:,1])