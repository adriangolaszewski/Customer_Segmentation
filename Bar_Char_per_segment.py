import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_segmentation_behavioral_labeled_top3.csv"
)


behavioral_cols = [
    'Food%', 'Fresh%', 'Drinks%', 'Home%',
    'Beauty%', 'Health%', 'Baby%', 'Pets%'
]

cluster_profile = (
    df.groupby('behavioral_cluster')[behavioral_cols]
      .mean()
)

# Normalize to proportions
cluster_profile = cluster_profile.div(
    cluster_profile.sum(axis=1),
    axis=0
)


cluster_labels = (
    df.groupby('behavioral_cluster')
      .first()[['behavioral_top1_label', 'behavioral_top2_label', 'behavioral_top3_label']]
)


for cluster in cluster_profile.index:

    label = (
        f"{cluster_labels.loc[cluster, 'behavioral_top1_label']} | "
        f"{cluster_labels.loc[cluster, 'behavioral_top2_label']} | "
        f"{cluster_labels.loc[cluster, 'behavioral_top3_label']}"
    )

    plt.figure(figsize=(8, 4))
    cluster_profile.loc[cluster].plot(kind='bar')

    plt.title(f"Customer Segment: {label}")
    plt.ylabel("Average Purchase Share")
    plt.xlabel("Product Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
