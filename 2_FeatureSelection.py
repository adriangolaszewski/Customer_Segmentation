import pandas as pd

# Load the dataset (use raw string for Windows path)
df = pd.read_csv(
   "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_segmentation.csv"
)
df.columns
columns_to_drop = [
    'customer',   
    'order',      
    'labels',     
    'class'       
]

df_features = df.drop(columns=columns_to_drop)
df_features.head()
df_features.dtypes
percentage_cols = [
    'Food%', 'Fresh%', 'Drinks%', 'Home%',
    'Beauty%', 'Health%', 'Baby%', 'Pets%'
]

df_features[percentage_cols].describe()
selected_features = df_features.columns.tolist()
selected_features
# Save
df_features.to_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_features_for_clustering.csv",
    index=False
)
