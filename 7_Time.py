import pandas as pd

df = pd.read_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_segmentation_transactional_labeled_top1.csv"
)


# Define time
def get_time_of_day(hour):
    if 6 <= hour < 14:
        return 'Morning'
    elif 14 <= hour < 22:
        return 'Evening'
    else:
        return 'Night'

df['time_of_day'] = df['hour'].apply(get_time_of_day)


# Remove duplicates
unique_cust_time = df[['customer','time_of_day']].drop_duplicates()

# Count customers per time_of_day
time_counts = unique_cust_time['time_of_day'].value_counts()

# Calculate percentages
total_customers = unique_cust_time['customer'].nunique()
time_percentages = (time_counts / total_customers * 100).reset_index()
time_percentages.columns = ['Time_of_Day', 'Percentage_of_Customers']


time_percentages.to_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\overall_customer_time_pct.csv",
    index=False
)

print("Overall customer time-of-day percentages:")
print(time_percentages)
