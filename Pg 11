import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Step 1: Read the dataset
df = pd.read_csv("mall_customers.csv")

# Step 2: Display the dataset
print("Dataset:")
print(df)

# Step 3: Display first five records
print("\nFirst Five Records:")
print(df.head())

# Step 4: Display dataset information
print("\nDataset Information:")
print(df.info())

# Step 5: Select features for clustering
X = df[["AnnualIncome", "SpendingScore"]]

# Step 6: Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Step 7: Train the model
kmeans.fit(X)

# Step 8: Get cluster labels
df["Cluster"] = kmeans.labels_

# Step 9: Display clustered dataset
print("\nClustered Dataset:")
print(df)

# Step 10: Display cluster centroids
print("\nCluster Centroids:")
print(kmeans.cluster_centers_)

# Step 11: Visualize the clusters
plt.scatter(
    df["AnnualIncome"],
    df["SpendingScore"],
    c=df["Cluster"]
)

# Plot centroids
centers = kmeans.cluster_centers_

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means Customer Clustering")

plt.show()
