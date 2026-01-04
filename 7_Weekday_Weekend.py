import pandas as pd


df = pd.read_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_segmentation_behavioral_labeled_top3.csv"
)


# Create weekend/weekday indicator

df['is_weekend'] = df['weekday'].isin([5, 6])


# Calculate average discount per cluster for weekend vs weekday

cluster_discount_comparison = df.groupby(['behavioral_top1_label', 'is_weekend'])['discount%'].mean().unstack()

cluster_discount_comparison.columns = ['Weekday_avg_discount', 'Weekend_avg_discount']


#  Check if weekend buyers tend to buy with bigger discounts

cluster_discount_comparison['Weekend_higher_discount'] = cluster_discount_comparison['Weekend_avg_discount'] > cluster_discount_comparison['Weekday_avg_discount']



print("Average discount for each cluster (weekday vs weekend) and if weekend buyers get higher discounts:")
print(cluster_discount_comparison)


cluster_discount_comparison.to_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\cluster_weekend_discount_analysis.csv"
)

