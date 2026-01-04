import pandas as pd
import matplotlib.pyplot as plt

# Load labeled dataset
df = pd.read_csv(r"C:\Users\adria\OneDrive\Pulpit\Customer_Segmentation\customer_clusters_top3_labeled.csv")

# Count customers per cluster label
cluster_counts = df['cluster_label'].value_counts()

# Plot
plt.figure(figsize=(10,6))
cluster_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Customers per Cluster', fontsize=14)
plt.ylabel('Number of Customers')
plt.xlabel('Cluster Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
