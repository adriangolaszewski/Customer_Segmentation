import pandas as pd

df = pd.read_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\cluster_behavioral_time_discount_summary.csv"
)

total_weekend_pct = df['pct_weekend'].mean()
total_weekday_pct = df['pct_weekday'].mean()

print("Overall percentage of customers buying during weekend:", round(total_weekend_pct*100, 2), "%")
print("Overall percentage of customers buying during weekday:", round(total_weekday_pct*100, 2), "%")

summary_df = pd.DataFrame({
    'Period': ['Weekend', 'Weekday'],
    'Percentage': [total_weekend_pct, total_weekday_pct]
})

summary_df.to_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\overall_weekday_weekend_summary.csv",
    index=False
)
