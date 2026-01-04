import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1️⃣ Load cluster labels and raw features
# -------------------------------
df_clusters = pd.read_csv(
    r"C:\Users\adria\OneDrive\Pulpit\Customer_Segmentation\customer_clusters_top3_labeled.csv"
)
df_raw = pd.read_csv(
    r"C:\Users\adria\OneDrive\Pulpit\Customer_Segmentation\customer_segmentation.csv"
)

# Merge cluster labels with original features
df_merged = df_raw.merge(df_clusters[['customer', 'cluster_label']], on='customer', how='left')

# -------------------------------
# 2️⃣ Define behavioral features
# -------------------------------
features = ['Food%', 'Fresh%', 'Drinks%', 'Home%', 'Beauty%', 'Baby%', 'Pets%']

# -------------------------------
# 3️⃣ Compute mean per cluster
# -------------------------------
cluster_summary = df_merged.groupby('cluster_label')[features].mean()

# -------------------------------
# 4️⃣ Plot heatmap
# -------------------------------
plt.figure(figsize=(12,6))
sns.heatmap(cluster_summary, annot=True, fmt=".1f", cmap='YlGnBu', cbar_kws={'label': 'Average %'})
plt.title('Average Feature Contribution per Cluster', fontsize=14)
plt.ylabel('Cluster Label', fontsize=12)
plt.xlabel('Feature', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
