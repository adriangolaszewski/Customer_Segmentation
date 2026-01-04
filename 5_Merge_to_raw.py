import pandas as pd

df_raw = pd.read_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_segmentation.csv"
)

df_behavioral = pd.read_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_clusters_behavioral.csv"
)



# Merge cluster labels into original data
df_merged = df_raw.copy()
df_merged['behavioral_cluster'] = df_behavioral['behavioral_cluster']


print(df_merged.head())
print("\nColumns:", df_merged.columns)


df_merged.to_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_segmentation_final.csv",
    index=False
)

print("\nMerged dataset with behavioral and transactional clusters saved successfully!")
