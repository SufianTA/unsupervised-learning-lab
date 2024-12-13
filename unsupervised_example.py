# unsupervised-learning-lab/unsupervised_example.py

from sklearn.cluster import KMeans
import numpy as np

# Generate synthetic data
X = np.random.randn(100, 2)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

print("Cluster centers:", kmeans.cluster_centers_)
