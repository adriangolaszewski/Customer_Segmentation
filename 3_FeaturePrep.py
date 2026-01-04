import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the cleaned feature dataset from Step 3
df_features = pd.read_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_features_for_clustering.csv"
)

# Inspect the features
df_features.head()
df_features.dtypes
# Check feature range
print(df_features.describe().loc[['min','max']])
# Cap discount at 0
df_features['discount%'] = df_features['discount%'].clip(lower=0)

# Feature Scaling
features_to_scale = df_features.columns.tolist()

X = df_features[features_to_scale]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for convenience
X_scaled_df = pd.DataFrame(X_scaled, columns=features_to_scale)
X_scaled_df.head()

# Save scaled features for clustering
X_scaled_df.to_csv(
     "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_features_scaled.csv",
    index=False
)
