import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df_scaled = pd.read_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_features_scaled.csv"
)

# -------------------------------
# Transactional features only
# -------------------------------
transactional_cols = ['total_items', 'discount%', 'weekday', 'hour', 'num_orders']

X_transactional = df_scaled[transactional_cols].copy()


# Scaled
scaler = StandardScaler()
X_scaled_transactional = scaler.fit_transform(X_transactional)

# Elbow method 
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled_transactional)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method - Transactional Features')
plt.show()


# Fit K-Means with chosen k
k_transactional = 4  # Replaced after inspecting elbow 
kmeans_transactional = KMeans(n_clusters=k_transactional, random_state=42)
transactional_labels = kmeans_transactional.fit_predict(X_scaled_transactional)


# Merge labels
df_transactional = df_scaled.copy()
df_transactional['transactional_cluster'] = transactional_labels

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_transactional)

plt.figure(figsize=(6,4))
plt.scatter(X_pca[:,0], X_pca[:,1], c=transactional_labels, cmap='tab10')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Transactional Clusters (PCA)')
plt.show()


df_transactional[['transactional_cluster']].to_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_clusters_transactional.csv",
    index=False
)

print("Transactional K-Means clusters saved successfully!")
