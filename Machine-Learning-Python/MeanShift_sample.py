import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use('ggplot')

# here we are just using an example with initial 3 centers
centers = [[1,1,1], [5,5,5], [3,10,10]]
# make blob function of sklearn is used to generate clusters, cluster_std is for standard deviation from the cluster centers
X, _ = make_blobs(n_samples = 1000, centers = centers, cluster_std = 1.0)

ms = MeanShift()
ms.fit(X)

# we obtain the final labels and cluster centres after mean shift
labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(cluster_centers)

colors = 10*['r','g','b','c','k','y','m']
# plotting a 3d graph
fig = plt.figure()
# the first argument of add_subplot is for size and position of the graph
ax = fig.add_subplot(111, projection = '3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

ax.scatter(cluster_centers[:,0], cluster_centers[:,1], cluster_centers[:,2], marker='x', color='k', linewidths=5, s=150, zorder=10)
plt.show()
