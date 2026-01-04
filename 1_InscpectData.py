import pandas as pd

# Load the dataset (use raw string for Windows path)
df = pd.read_csv(
   "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_segmentation.csv"
)

# Preview data
df.head()
# Shape of the dataset
print("Rows, Columns:", df.shape)

# Column names
print("\nColumns:")
print(df.columns)

# Data types & missing values
df.info()
df.isnull().sum()
df.describe()
# Number of unique customers
print("Unique customers:", df['customer'].nunique())

# Check if one row = one customer
df['customer'].value_counts().head()
category_cols = [
    'Food%', 'Fresh%', 'Drinks%', 'Home%',
    'Beauty%', 'Health%', 'Baby%', 'Pets%'
]

df[category_cols].describe()
