import streamlit as st
import pandas as pd
import numpy as np
from create_fake_data import create_fake_dataset
from data_model import data_model as box_data_model
from clustering.k_means import perform_kmeans
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")


def df_return():
    df = create_fake_dataset()
    return df


sections = {
    'Clusters 4 Operators': 'section-1'
}

st.header("KMeans Data Performance")

num_clusters = "Automatic"
if num_clusters == "Automatic":
    num_clusters = None

fake_data = df_return()
if "engineering_df" not in st.session_state:
    engineering_df = pd.read_excel("Engineering_data.xlsx")
    st.session_state["engineering_df"] = engineering_df
    st.write(fake_data)

st.header("Clusters 4 Operators", anchor=sections['Clusters 4 Operators'])
df = df_return()

num_clusters = st.selectbox('Number of clusters', ["Automatic", 2, 3, 4, 5, 6, 7, 8, 9, 10], index=3)

if num_clusters == "Automatic":
    num_clusters = None

columns = df.columns.tolist()
cluster_columns = st.multiselect('Select columns for clustering', columns, default=columns[2:6])
fake_data = df[cluster_columns].values

cluster_labels, cluster_centers, inertia, optimal_k = perform_kmeans(fake_data, num_clusters)
center_df = pd.DataFrame(cluster_centers, columns=cluster_columns)
cluster_names = [f'Cluster {i}' for i in range(optimal_k)]
center_df['Name'] = cluster_names
engineering_df = pd.read_excel("Engineering_data.xlsx")  # check df session control
cluster_df = center_df

st.subheader('Cluster Centers', anchor='cluster-centers')
cluster_df["index_num"] = cluster_df.index
figures = []

for column in cluster_columns:
    fig = px.scatter(cluster_df, x="index_num", y=column, color="Name")
    fig.update_traces(marker={'size': 15})

    # Add horizontal lines for target, min, and max values from the target DataFrame
    if column in engineering_df['name'].values:
        target_value = engineering_df.loc[engineering_df['name'] == column, 'target'].values[0]
        min_value = engineering_df.loc[engineering_df['name'] == column, 'min'].values[0]
        max_value = engineering_df.loc[engineering_df['name'] == column, 'max'].values[0]

        fig.add_shape(type="line", x0=0, y0=target_value, x1=len(cluster_df[column]) - 1, y1=target_value,
                      line=dict(color="green", width=1), name='target')
        fig.add_shape(type="line", x0=0, y0=min_value, x1=len(cluster_df[column]) - 1, y1=min_value,
                      line=dict(color="violet", width=1), name='min')
        fig.add_shape(type="line", x0=0, y0=max_value, x1=len(cluster_df[column]) - 1, y1=max_value,
                      line=dict(color="red", width=1), name='max')

    # Update layout
    fig.update_layout(title=f"Cluster Centers for {column}",
                      xaxis_title="Cluster Index",
                      yaxis_title=column)
    # Add the figure to the list
    figures.append(fig)

# Display the figures in a row format
for fig in figures:
    st.plotly_chart(fig)

# Calculate max and min values for each cluster
cluster_max_values = []
cluster_min_values = []

for i in range(optimal_k):
    cluster_data = fake_data[cluster_labels == i]  # Get data points belonging to cluster i
    cluster_max = np.max(cluster_data, axis=0)  # Calculate max values for each feature in the cluster
    cluster_min = np.min(cluster_data, axis=0)  # Calculate min values for each feature in the cluster
    cluster_max_values.append(cluster_max)
    cluster_min_values.append(cluster_min)

# Convert max and min values to DataFrame
max_df = pd.DataFrame(cluster_max_values, columns=cluster_columns)
min_df = pd.DataFrame(cluster_min_values, columns=cluster_columns)

# Add cluster names
max_df['Cluster'] = [f'Cluster {i}' for i in range(optimal_k)]
min_df['Cluster'] = [f'Cluster {i}' for i in range(optimal_k)]

# # Display max and min values
# st.subheader('Max and Min Values for Each Cluster')
# st.write("Max Values:")
# st.write(max_df)
# st.write("Min Values:")
# st.write(min_df)

# Define a color palette for clusters
colors = px.colors.qualitative.Set1

# Create subplots
fig = go.Figure()

# Plot min and max values for each column
for i, column in enumerate(cluster_columns):
    for j in range(optimal_k):
        fig.add_trace(go.Bar(x=[f'Cluster {j} - Min', f'Cluster {j} - Max'],
                             y=[min_df[column][j], max_df[column][j]],
                             name=column,
                             marker_color=colors[j]))

# Update layout
fig.update_layout(title="Min and Max Values for Each Cluster",
                  xaxis_title="Cluster",
                  yaxis_title="Value",
                  barmode='group')

st.plotly_chart(fig)