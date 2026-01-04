import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df_scaled = pd.read_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_features_scaled.csv"
)

# Behavioral features only
behavioral_cols = [
    'Food%', 'Fresh%', 'Drinks%', 'Home%',
    'Beauty%', 'Health%', 'Baby%', 'Pets%', 
]

X_behavioral = df_scaled[behavioral_cols].copy()

# Scaled datad
scaler = StandardScaler()
X_scaled_behavioral = scaler.fit_transform(X_behavioral)


# Elbow method
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled_behavioral)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method - Behavioral Features')
plt.show()

# Fit K-Means with chosen k
k_behavioral = 7 # Replaced after inspecting elbow
kmeans_behavioral = KMeans(n_clusters=k_behavioral, random_state=42)
behavioral_labels = kmeans_behavioral.fit_predict(X_scaled_behavioral)


# Merge labels back
df_behavioral = df_scaled.copy()
df_behavioral['behavioral_cluster'] = behavioral_labels

# -------------------------------
# PCA for visualization
# -------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_behavioral)

plt.figure(figsize=(6,4))
plt.scatter(X_pca[:,0], X_pca[:,1], c=behavioral_labels, cmap='tab10')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Behavioral Clusters (PCA)')
plt.show()

# Save cluster labels
df_behavioral[['behavioral_cluster']].to_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_clusters_behavioral.csv",
    index=False
)

print("Behavioral K-Means clusters saved successfully!")
