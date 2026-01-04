import pandas as pd


df = pd.read_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_segmentation_final.csv"
)

# Behavioral features
behavioral_cols = [
    'Food%', 'Fresh%', 'Drinks%', 'Home%',
    'Beauty%', 'Health%', 'Baby%', 'Pets%'
]


# Compute cluster summary (mean values)
cluster_summary = df.groupby('behavioral_cluster')[behavioral_cols].mean()


# Calculate relative contribution / proportion
cluster_prop = cluster_summary.div(cluster_summary.sum(axis=1), axis=0)

# Map features to labels
feature_to_label = {
    'Food%': 'Food',
    'Drinks%': 'Drinks',
    'Fresh%': 'Fresh',
    'Home%': 'Home',
    'Beauty%': 'Beauty',
    'Health%': 'Health',
    'Baby%': 'Baby',
    'Pets%': 'Pet'
}

# Get top 3 features per cluster
def get_top3_features(row):
    top_features = row.sort_values(ascending=False).head(3).index.tolist()
    top_labels = [feature_to_label.get(f, f) for f in top_features]
    return top_labels

cluster_top3 = cluster_prop.apply(get_top3_features, axis=1)


# Assign labels to customers in three separate columns
df['behavioral_top1_label'] = df['behavioral_cluster'].map(lambda x: cluster_top3[x][0])
df['behavioral_top2_label'] = df['behavioral_cluster'].map(lambda x: cluster_top3[x][1])
df['behavioral_top3_label'] = df['behavioral_cluster'].map(lambda x: cluster_top3[x][2])


#Check counts per top1 label
label_counts = df.groupby('behavioral_top1_label')['behavioral_cluster'].count()
print("\nCustomer counts per top1 behavioral label:")
print(label_counts)


df.to_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_segmentation_behavioral_labeled_top3.csv",
    index=False
)

print("\nBehavioral clusters labeled successfully! Top 3 features stored in separate columns.")
