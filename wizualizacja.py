import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px


df = pd.read_csv(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\customer_segmentation_behavioral_labeled_top3.csv"
)


behavioral_cols = [
    'Food%', 'Fresh%', 'Drinks%', 'Home%',
    'Beauty%', 'Health%', 'Baby%', 'Pets%'
]

X = df[behavioral_cols]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# PCA 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]



color_map = {
    "Fresh":  "#7CFC9A",   # bright green
    "Drinks": "#A8F07A",   # light green
    "Home":   "#D9F57A",   # yellow-green
    "Baby":   "#F5F08A",   # soft yellow
    "Food":   "#F7E59C",   # warm yellow
    "Beauty": "#F7B7A3",   # peach / light pink
    "Pet":    "#F28CA3"    # pink
}


fig = px.scatter(
    df,
    x='PC1',
    y='PC2',
    color='behavioral_top1_label',
    color_discrete_map=color_map,
    size='num_orders',
    hover_data=[
        'customer',
        'num_orders',
        'behavioral_top2_label',
        'behavioral_top3_label'
    ],
    title='Customer Behavioral Segments (Purchase Preferences)',
    width=900,
    height=600
)


fig.update_traces(
    marker=dict(
        opacity=0.85,
        line=dict(width=0.5, color='white')
    )
)

fig.update_layout(
    legend_title_text='Primary Purchase Category',
    xaxis_title='Behavioral Dimension 1',
    yaxis_title='Behavioral Dimension 2',
    template='simple_white',
    plot_bgcolor="#F3FEFF",   # chart area
    paper_bgcolor="#FFFFFF",  # entire canvas
)


fig.show()

fig.write_html(
    "C:\\Users\\adria\\OneDrive\\Pulpit\\Customer_Segmentation\\behavioral_clusters_contrast.html",
    include_plotlyjs='cdn',
    full_html=True
)
