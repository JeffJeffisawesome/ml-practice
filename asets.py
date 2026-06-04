import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# A1
url = "https://cs.joensuu.fi/sipu/datasets/a1.txt"

# Download and format the data
df = pd.read_csv(url, sep=r'\s+', header=None, names=['x', 'y'])

# Initialize K-Means. We know A1 has 20 clusters, so we set n_clusters=20.
kmeans = KMeans(n_clusters=20, random_state=42, n_init="auto")

# Fit the algorithm and label the data
df['cluster_label'] = kmeans.fit_predict(df[['x', 'y']])

# Create a scatter plot of the clusters
plt.figure(figsize=(10, 8))
plt.scatter(df['x'], df['y'], c=df['cluster_label'], cmap='tab20', s=5)
plt.title("K-Means Clustering on A1 Dataset (20 Clusters)")
plt.show()