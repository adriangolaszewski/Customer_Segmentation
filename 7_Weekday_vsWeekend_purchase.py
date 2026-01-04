import pandas as pd

# Load your dataset
df = pd.read_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_segmentation_transactional_labeled_top1.csv"
)

# Define if each purchase is on a weekend or weekday
df['is_weekend'] = df['weekday'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

# Count total purchases per type
purchase_counts = df['is_weekend'].value_counts()

# Number of days
num_weekdays = 5
num_weekend_days = 2

# Average purchases per day
avg_weekday = purchase_counts.get('Weekday', 0) / num_weekdays
avg_weekend = purchase_counts.get('Weekend', 0) / num_weekend_days

# Calculate percentage difference (how much more per weekend day vs a weekday)
percent_more_weekend = ((avg_weekend - avg_weekday) / avg_weekday) * 100

print(f"Average purchases per weekday: {avg_weekday:.2f}")
print(f"Average purchases per weekend day: {avg_weekend:.2f}")
print(f"Weekend days have {percent_more_weekend:.2f}% more purchases than a weekday.")
