import pandas as pd

df = pd.read_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_segmentation_behavioral_labeled_top3.csv"
)


# Define day and time segments

# Weekend / Weekday
df['is_weekend'] = df['weekday'].isin([5,6])  

# Time of day
def time_of_day(hour):
    if 6 <= hour < 14:
        return 'morning'
    elif 14 <= hour < 22:
        return 'evening'
    else:
        return 'night'

df['time_segment'] = df['hour'].apply(time_of_day)


#  Discount quartiles

q1 = df['discount%'].quantile(0.25)
q2 = df['discount%'].quantile(0.5)
q3 = df['discount%'].quantile(0.75)

def discount_group(x):
    if x <= q1:
        return 'Low'
    elif x <= q2:
        return 'Medium-Low'
    elif x <= q3:
        return 'Medium-High'
    else:
        return 'High'

df['discount_group'] = df['discount%'].apply(discount_group)

# Aggregate per cluster

# % Weekend / Weekday buyers
daytype_pct = df.groupby('behavioral_top1_label')['is_weekend'].mean().rename('pct_weekend')
daytype_pct = daytype_pct.to_frame()
daytype_pct['pct_weekday'] = 1 - daytype_pct['pct_weekend']

# % by time of day
time_pct = df.groupby(['behavioral_top1_label','time_segment']).size().unstack(fill_value=0)
time_pct = time_pct.div(time_pct.sum(axis=1), axis=0)
time_pct.columns = [f'pct_{c}' for c in time_pct.columns]

# % by discount group
discount_pct = df.groupby(['behavioral_top1_label','discount_group']).size().unstack(fill_value=0)
discount_pct = discount_pct.div(discount_pct.sum(axis=1), axis=0)
discount_pct.columns = [f'pct_discount_{c}' for c in discount_pct.columns]


# Merge all percentages into one table
cluster_summary_pct = daytype_pct.join(time_pct).join(discount_pct)
cluster_summary_pct = cluster_summary_pct.reset_index()


cluster_summary_pct.to_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\cluster_behavioral_time_discount_summary.csv",
    index=False
)

print("Cluster-level % by day type, time of day, and discount group saved successfully!")
print(cluster_summary_pct)
