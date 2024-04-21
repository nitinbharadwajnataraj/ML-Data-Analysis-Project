import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from data_model import data_model
from create_fake_data import create_fake_dataset
from Compute_fit import compute_fit, count_yes_no
from k_means import perform_kmeans

st.set_page_config(layout="wide")

def df_return():
    df = create_fake_dataset()
    return df


def color_code(val):
    if val == 'OK':
        color = 'green'
    elif val == 'NOK':
        color = 'red'
    elif val == 'Transition':
        color = 'lightblue'
    elif val == 'Clearance':
        color = 'yellow'
    else:
        color = 'grey'
    return f'background-color: {color}'
def df_fitting_and_evaluation():
    df = df_return()
    df["fitting"] = df["box_hole_diameter"] - df["cylinder_diameter"]

    # Using & instead of 'and'
    condition1 = (df["fitting"] <= 1) & (df["fitting"] >= -1)
    condition2 = (df["fitting"] > 1)

    # Assigning values based on conditions
    df.loc[condition1, "Evaluation"] = 'OK'
    df.loc[condition1, "fitting_group"] = 'Transition'
    df.loc[condition2, "Evaluation"] = 'NOK'
    df.loc[condition2, "fitting_group"] = 'Clearance'
    df.loc[~(condition1 | condition2), "Evaluation"] = 'NOK'
    df.loc[~(condition1 | condition2), "fitting_group"] = 'Excess'

    styled_df = df.style.applymap(color_code, subset=['Evaluation', 'fitting_group'])

    return df, styled_df


def df_bar_chart_Evaluation():
    df1, df2 = df_fitting_and_evaluation()
    count_ok, count_nok = 0, 0
    for val in df1["Evaluation"]:
        if val == "OK":
            count_ok += 1
        elif val == 'NOK':
            count_nok += 1

    data = {'Category': ['OK','NOK'], 'Value': [count_ok, count_nok]}
    df = pd.DataFrame(data)

    fig = go.Figure(data=[go.Bar(x=df['Category'], y=df['Value'])])
    fig.update_layout(title='Evaluation Results', xaxis_title='Category', yaxis_title='Value')
    st.plotly_chart(fig)


def df_bar_chart_fitting_group():
    df1, df2 = df_fitting_and_evaluation()
    count_transition, count_clearance, count_excess = 0, 0, 0
    for val in df1["fitting_group"]:
        if val == "Excess":
            count_excess += 1
        elif val == "Clearance":
            count_clearance += 1
        elif val == "Transition":
            count_transition += 1

    data = {'Category': ['Transition', 'Clearance', 'Excess'], 'Value': [count_transition, count_clearance, count_excess]}
    df = pd.DataFrame(data)

    fig = go.Figure(data=[go.Bar(x=df['Category'], y=df['Value'])])
    fig.update_layout(title='Fitting Groups', xaxis_title='Category', yaxis_title='Value')
    st.plotly_chart(fig)

def scatter_plot():
    df_1 = compute_fit()
    df_1 = df_1.drop(columns=['ID'], axis=1)
    col1, col2 = st.columns([0.3,0.7])
    with col1:
        selected_shape = st.selectbox("Select shape for scatter plot", ["Box", "Cylinder"], key="select_shape")

        if selected_shape:
            if selected_shape == "Box":
                related_columns = ["box_hole_diameter", "box_hole_depth"]
            elif selected_shape == "Cylinder":
                related_columns = ["cylinder_diameter", "cylinder_height"]
            else:
                related_columns = []

            selected_column = st.selectbox("Select a column for scatter plot", related_columns, key="select_column")

            yes_color = '#FF7F0E'
            no_color = 'grey'

        with col2:
            if selected_column:
                st.write("Scatter Plot:")
                fig = go.Figure()

                colors = [yes_color if val == 'yes' else no_color for val in df_1["check"]]
                fig.add_trace(go.Scatter(x=df_1.index, y=df_1[selected_column], mode='markers', name=selected_column,
                                         marker=dict(color=colors)))

                min_value = data_model[selected_shape][selected_column]["min_value"]
                max_value = data_model[selected_shape][selected_column]["max_value"]
                target_value = data_model[selected_shape][selected_column]["target_value"]

                fig.add_shape(type="line", x0=df_1.index.min(), y0=min_value, x1=df_1.index.max(), y1=min_value,
                              line=dict(color="red", width=1, dash="solid"))
                fig.add_shape(type="line", x0=df_1.index.min(), y0=max_value, x1=df_1.index.max(), y1=max_value,
                              line=dict(color="red", width=1, dash="solid"))
                fig.add_shape(type="line", x0=df_1.index.min(), y0=target_value, x1=df_1.index.max(), y1=target_value,
                              line=dict(color="lime", width=1, dash="solid"))

                fig.add_annotation(
                    x=df_1.index.min(),
                    y=min_value,
                    text="Minimum Value",
                    showarrow=False,
                    font=dict(size=12, color="red")
                )
                fig.add_annotation(
                    x=df_1.index.min(),
                    y=max_value,
                    text="Maximum Value",
                    showarrow=False,
                    font=dict(size=12, color="red")
                )
                fig.add_annotation(
                    x=df_1.index.min(),
                    y=target_value,
                    text="Target Value",
                    showarrow=False,
                    font=dict(size=12, color="lime")
                )

                st.plotly_chart(fig)
            else:
                st.warning("Please select a column for the scatter plot.")

def box_plot():
    df = df_return()
    fig = go.Figure()
    for column in df.columns[1:]:
        fig.add_trace(go.Box(y=df[column], name=column))
    fig.update_layout(title='Box Plot(Fake_data understanding)', xaxis_title='Parameters', yaxis_title='Values')
    st.plotly_chart(fig)

def bar_chart():
    df_1, count_yes, count_no = count_yes_no()
    df2,df3 = df_fitting_and_evaluation()
    st.dataframe(df3, width=800)
    distinct_check_value = list(set(df_1['check']))
    data = {'Category': distinct_check_value, 'Value': [count_yes, count_no]}
    df = pd.DataFrame(data)

    fig = go.Figure(data=[go.Bar(x=df['Category'], y=df['Value'])])
    fig.update_layout(title='Bar Chart', xaxis_title='Category', yaxis_title='Value')
    st.plotly_chart(fig)

def kmeans_info_popover():
    st.popover("K-Means analysis provides insights into data clustering. It helps in identifying patterns and grouping similar data points together. Select the number of clusters and features for clustering to visualize the results.")

def kmeans():
    sections = {'Clusters 4 Operators': 'section-1'}

    st.header("Synthetic Data")

    num_clusters = "Automatic"
    if num_clusters == "Automatic":
        num_clusters = None

    fake_data = df_return()
    if "engineering_df" not in st.session_state:
        engineering_df = pd.read_excel("engineered_data_for_cluster.xlsx")
        st.session_state["engineering_df"] = engineering_df
        st.write(fake_data)

    st.header("Cluster Analysis", anchor=sections['Clusters 4 Operators'])
    df = df_return()

    # import streamlit as st
    #
    # popover = st.popover("Filter items")
    # red = popover.checkbox("Show red items.", True)
    # blue = popover.checkbox("Show blue items.", True)
    #
    # if red:
    #     st.write(":red[This is a red item.]")
    # if blue:
    #     st.write(":blue[This is a blue item.]")

    # popover = st.popover('Number of clusters')
    # automatic = popover.checkbox('Automatic')
    automatic_clusters = st.checkbox("Automatic", False)
    if automatic_clusters:
        num_clusters = None
    else:
        with st.popover(""):
            st.markdown("Hello World 👋")
            name = st.text_input("What's your name?")

        st.write("Your name:", name)
        num_clusters = st.popover('Number of clusters', [2, 3, 4, 5, 6, 7, 8, 9, 10], index=1)

    if num_clusters == "Automatic":
        num_clusters = None

    df = df.drop(columns=['ID'], axis=1)
    columns = df.columns.tolist()
    cluster_columns = st.multiselect('Select columns for clustering', columns, default=columns[2:6])
    fake_data = df[cluster_columns].values

    cluster_labels, cluster_centers, inertia, optimal_k = perform_kmeans(fake_data, num_clusters)
    center_df = pd.DataFrame(cluster_centers, columns=cluster_columns)
    cluster_names = [f'Cluster {i}' for i in range(optimal_k)]
    center_df['Name'] = cluster_names
    engineering_df = pd.read_excel("engineered_data_for_cluster.xlsx")  # check df session control
    cluster_df = center_df

    st.subheader('Cluster Centers', anchor='cluster-centers')
    cluster_df["index_num"] = cluster_df.index
    figures = []

    for column in cluster_columns:
        fig = px.scatter(cluster_df, x="index_num", y=column, color="Name")
        fig.update_traces(marker={'size': 15})

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

        fig.update_layout(title=f"Cluster Centers for {column}", xaxis_title="Cluster Index", yaxis_title=column)
        figures.append(fig)

    for fig in figures:
        st.plotly_chart(fig)

    cluster_max_values = []
    cluster_min_values = []

    for i in range(optimal_k):
        cluster_data = fake_data[cluster_labels == i]
        cluster_max = np.max(cluster_data, axis=0)
        cluster_min = np.min(cluster_data, axis=0)
        cluster_max_values.append(cluster_max)
        cluster_min_values.append(cluster_min)

    max_df = pd.DataFrame(cluster_max_values, columns=cluster_columns)
    min_df = pd.DataFrame(cluster_min_values, columns=cluster_columns)

    max_df['Cluster'] = [f'Cluster {i}' for i in range(optimal_k)]
    min_df['Cluster'] = [f'Cluster {i}' for i in range(optimal_k)]

    fig = go.Figure()

    for i, column in enumerate(cluster_columns):
        for j in range(optimal_k):
            fig.add_trace(go.Bar(x=[f'Cluster {j} - Min', f'Cluster {j} - Max'],
                                 y=[min_df[column][j], max_df[column][j]],
                                 name=column,
                                 marker_color=px.colors.qualitative.Set1[j]))

    fig.update_layout(title="Min and Max Values for Each Cluster",
                      xaxis_title="Cluster",
                      yaxis_title="Value",
                      barmode='group')

    st.plotly_chart(fig)
def clustering_for_evaluation():
    cluster_mapping = {}
    df = df_return()
    for index, row in df.iterrows():
        paired_with: int = row['Paired_With']  # Paired with resembles the ID column of the paired part
        # Get the cluster of the paired part
        paired_cluster = df.loc[df['ID'] == paired_with, 'Cluster'].values[0]
        cluster_mapping[paired_with] = paired_cluster
        # Create a new column to store the cluster label of the paired part
        df['Paired Cluster'] = df['Paired_With'].map(cluster_mapping)

    # Sample input dataframe
    data = {
        'Cluster_A': df['Cluster'].values,
        'Cluster_B': df['Paired Cluster'].values,
        'Evaluation': df['Evaluation'].values
    }

    contingency_df = pd.DataFrame(data)

    # Create a list of unique clusters
    clusters = list(set(contingency_df['Cluster_A']) | set(contingency_df['Cluster_B']))

    # Initialize a dictionary to store combined cluster occurrences
    combined_cluster_dict = {"Cluster A": [], "Cluster B": [], 'Combined_Cluster': [], 'Occurrence': [],
                             'Num_OK': [],
                             'Num_NOK': [], 'OK_Ratio': [], "Cluster_A_Name": [], "Cluster_B_Name": []}

    # Iterate through each unique combination of clusters
    for i, cluster_a in enumerate(clusters):
        for cluster_b in clusters[i:]:
            combined_cluster = f'{cluster_a}_{cluster_b}' if cluster_a <= cluster_b else f'{cluster_b}_{cluster_a}'
            occurrence = 0
            num_ok = 0
            num_nok = 0

            # Count occurrences, num_ok, and num_nok for the combination
            for index, row in contingency_df.iterrows():
                if (row['Cluster_A'] == cluster_a and row['Cluster_B'] == cluster_b) or \
                        (row['Cluster_A'] == cluster_b and row['Cluster_B'] == cluster_a):
                    occurrence += 1
                    if row['Evaluation'] == 'OK':
                        num_ok += 1
                    else:
                        num_nok += 1
            # Calculate OK ratio
            ok_ratio = num_ok / occurrence if occurrence > 0 else None

            # Append data to the dictionary
            combined_cluster_dict['Cluster A'].append(
                cluster_a[-1:])  # Append only the number of the cluster for sorting purposes
            combined_cluster_dict['Cluster_A_Name'].append(cluster_a)
            combined_cluster_dict['Cluster B'].append(cluster_b[-1:])
            combined_cluster_dict['Cluster_B_Name'].append(cluster_b)
            combined_cluster_dict['Combined_Cluster'].append(combined_cluster)
            combined_cluster_dict['Occurrence'].append(occurrence)
            combined_cluster_dict['Num_OK'].append(num_ok)
            combined_cluster_dict['Num_NOK'].append(num_nok)
            combined_cluster_dict['OK_Ratio'].append(ok_ratio)

    # Create the second dataframe
    second_df = pd.DataFrame(combined_cluster_dict)
    # sort the dataframe
    second_df = second_df.sort_values(by=['Cluster A', 'Cluster B'], ascending=[True, True])
    second_df = second_df.reset_index(drop=True)

    st.dataframe(second_df, use_container_width=True)

    # Create a matrix to hold the OK ratio values
    matrix = pd.DataFrame(index=second_df['Cluster_A_Name'].unique(), columns=second_df['Cluster_B_Name'].unique())

    # Populate the matrix with OK ratio values
    for index, row in second_df.iterrows():
        clusters = row['Combined_Cluster'].split('_')
        matrix.loc[clusters[0], clusters[1]] = row['OK_Ratio']

    # Plot heatmap using Plotly Express
    fig = px.imshow(matrix.values.tolist(),
                    labels=dict(x="Cluster 2", y="Cluster 1", color="OK Ratio"),
                    x=matrix.columns,
                    y=matrix.index,
                    zmin=0.0,
                    zmax=1.0,
                    text_auto=True,
                    color_continuous_scale='rdylgn')

    fig.update_layout(
        title='OK Ratio Heatmap of Cluster Pairs'
    )

    st.subheader('OK Ratio Heatmap of Cluster Pairs', anchor='heatmap')
    st.plotly_chart(fig)

def main():
    st.markdown('<h1 style="text-align: center;">Box and Cylinder Analysis</h1>', unsafe_allow_html=True)

    sections = {'Bar-Chart': 'Bar-Chart', 'Plot': 'Plot', 'K-Means': 'k-means'}

    st.sidebar.title('PMV4')
    selected_nav = st.sidebar.selectbox("Navigate to", list(sections.keys()), key='navigation')

    if selected_nav:
        st.session_state.selected_nav = sections[selected_nav]

    nav = st.session_state.selected_nav

    if nav == 'Bar-Chart':
        bar_chart()
        df_bar_chart_Evaluation()
        df_bar_chart_fitting_group()
    elif nav == 'Plot':
        tab1, tab2 = st.tabs(["Box-Plot", "Scatter-Plot"])
        with tab1:
            st.header("Box-Plot")
            box_plot()
        with tab2:
            st.header("Scatter-Plot")
            scatter_plot()
    elif nav == 'k-means':
        kmeans_info_popover()
        kmeans()
        # clustering_for_evaluation()

if __name__ == "__main__":
    main()