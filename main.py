import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# STEP 1: Fetch and Load the Data
# ==========================================
# We will use the 's1.txt' dataset. It is a text file with two columns (X and Y coordinates)
url = "https://cs.joensuu.fi/sipu/datasets/s1.txt"

# Read the data directly from the web. '\s+' handles unknown amounts of spaces/tabs.
df = pd.read_csv(url, sep=r'\s+', header=None, names=['x', 'y'])

print("Data Loaded Successfully!")
print(df.head())


# ==========================================
# STEP 2: The Clustering Approach (Unsupervised)
# ==========================================
# The S-sets consist of exactly 15 clusters. Let's use K-Means to find them.
print("\n--- Running K-Means Clustering ---")

# Initialize K-Means asking for 15 clusters
kmeans = KMeans(n_clusters=15, random_state=42, n_init="auto")

# Fit the model and assign a cluster number to every data point
df['cluster_label'] = kmeans.fit_predict(df[['x', 'y']])

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(df['x'], df['y'], c=df['cluster_label'], cmap='tab20', s=10)
plt.title("K-Means Clustering on S1 Dataset")
plt.show()


# ==========================================
# STEP 3: The Classification Approach (Supervised)
# ==========================================
print("\n--- Running Random Forest Classification ---")

# To build a classifier, we need a "target" to predict. 
# We will use the cluster labels we just found as our "ground truth" target.
X = df[['x', 'y']]         # Features (Inputs)
y = df['cluster_label']    # Target (Outputs)

# Split the data: 80% to train the classifier, 20% to test it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest Classifier
classifier = RandomForestClassifier(random_state=42)

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Ask the classifier to predict labels for the unseen test data
predictions = classifier.predict(X_test)

# Check the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Classifier Accuracy: {accuracy * 100:.2f}%")