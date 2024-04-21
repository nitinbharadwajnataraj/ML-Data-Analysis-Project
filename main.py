import plotly.figure_factory as ff
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from data_model import data_model
from Compute_fit import compute_fit, count_yes_no
from clustering.k_means import perform_kmeans
from Decision_Tress import Decision_Tress
st.set_page_config(layout="wide")



def color_code(val):
    if val == 'OK':
        color = 'rgba(0, 255, 0, 0.3)'
    elif val == 'NOK':
        color = 'rgba(255, 0, 0, 0.3)'
    elif val == 'Transition':
        color = 'lightblue'
    elif val == 'Clearance':
        color = 'yellow'
    else:
        color = 'rgba(255, 165, 0, 0.3)'
    return f'background-color: {color}'

def color_code2(val):
    if (val <= 1) and (val >= -1):
        color = 'lightblue'
    elif val > 1:
        color = 'yellow'
    elif val < -1:
        color = 'rgba(255, 165, 0, 0.3)'
    return f'background-color: {color}'


def df_fitting_and_evaluation():
    df = pd.read_excel("fake_data.xlsx")
    df["fitting_distance"] = df["box_hole_diameter"] - df["cylinder_diameter"]

    # Using & instead of 'and'
    condition1 = (df["fitting_distance"] <= 1) & (df["fitting_distance"] >= -1)
    condition2 = (df["fitting_distance"] > 1)

    # Assigning values based on conditions
    df.loc[condition1, "Evaluation"] = 'OK'
    df.loc[condition1, "fitting_group"] = 'Transition'
    df.loc[condition2, "Evaluation"] = 'NOK'
    df.loc[condition2, "fitting_group"] = 'Clearance'
    df.loc[~(condition1 | condition2), "Evaluation"] = 'NOK'
    df.loc[~(condition1 | condition2), "fitting_group"] = 'Excess'

    styled_df = df.style.map(color_code, subset=['Evaluation', 'fitting_group' ])
    styled_df = styled_df.map(color_code2, subset=['fitting_distance'])

    return df, styled_df


def df_bar_chart_Evaluation():
    df1, df2 = df_fitting_and_evaluation()
    # st.dataframe(df2, width=800)
    count_ok, count_nok = 0, 0
    for val in df1["Evaluation"]:
        if val == 'OK':
            count_ok += 1
        elif val == 'NOK':
            count_nok += 1

    data = {'Category': ['OK', 'NOK'], 'Value': [count_ok, count_nok]}
    df = pd.DataFrame(data)

    fig = go.Figure(data=[go.Bar(x=df['Category'], y=df['Value'])])
    fig.update_layout(title='Evaluation Results', xaxis_title='Category', yaxis_title='Value')
    st.plotly_chart(fig)


def df_bar_chart_fitting_group():
    df1, df2 = df_fitting_and_evaluation()
    count_transition, count_clearance, count_excess = 0, 0, 0
    for val in df1["fitting_group"]:
        if val == 'Excess':
            count_excess += 1
        elif val == 'Clearance':
            count_clearance += 1
        elif val == 'Transition':
            count_transition += 1

    data = {'Category': ['Transition', 'Clearance', 'Excess'],
            'Value': [count_transition, count_clearance, count_excess]}
    df = pd.DataFrame(data)

    fig = go.Figure(data=[go.Bar(x=df['Category'], y=df['Value'])])
    fig.update_layout(title='Fitting Groups', xaxis_title='Category', yaxis_title='Value')
    st.plotly_chart(fig)


def scatter_plot():
    df_1 = compute_fit()
    df_1 = df_1.drop(columns=['ID'], axis=1)
    col1, col2 = st.columns([0.3, 0.7])
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
    df = pd.read_excel("fake_data.xlsx")
    fig = go.Figure()
    for column in df.columns[1:]:
        fig.add_trace(go.Box(y=df[column], name=column))
    fig.update_layout(title='Box Plot(Fake_data understanding)', xaxis_title='Parameters', yaxis_title='Values')
    st.plotly_chart(fig)


def bar_chart():
    df_1, count_yes, count_no = count_yes_no()
    df2, df3 = df_fitting_and_evaluation()
    st.dataframe(df3, width=800)
    distinct_check_value = list(set(df_1['check']))
    data = {'Category': distinct_check_value, 'Value': [count_yes, count_no]}
    df = pd.DataFrame(data)

    fig = go.Figure(data=[go.Bar(x=df['Category'], y=df['Value'])])
    fig.update_layout(title='Bar Chart', xaxis_title='Category', yaxis_title='Value')
    st.plotly_chart(fig)


def kmeans_info_popover():
    st.popover(
        "K-Means analysis provides insights into data clustering. It helps in identifying patterns and grouping similar data points together. Select the number of clusters and features for clustering to visualize the results.")


def kmeans():
    sections = {'Clusters 4 Operators': 'section-1'}

    fake_data = pd.read_excel("fake_data.xlsx")
    if "engineering_df" not in st.session_state:
        engineering_df = pd.read_excel("Engineering_data.xlsx")
        st.session_state["engineering_df"] = engineering_df
        #st.write(fake_data)

    st.header("Cluster Analysis", anchor=sections['Clusters 4 Operators'])
    df = pd.read_excel("fake_data.xlsx")
    with st.popover("Number of Clusters"):
        automatic_clusters = st.checkbox("Automatic", False)
        if automatic_clusters:
            num_clusters = None
        else:
            num_clusters = st.selectbox('Number of clusters', [2, 3, 4, 5, 6, 7, 8, 9, 10], index=1)

    num_clusters_write = num_clusters
    if num_clusters_write is None:
        num_clusters_write = " Automatic"

    st.write("Selected number of Clusters:", num_clusters_write)
    df = df.drop(columns=['ID'], axis=1)
    columns = df.columns.tolist()
    cluster_columns = st.multiselect('Select columns for clustering', columns, default=columns[2:6])
    fake_data = df[cluster_columns].values

    try:
        cluster_labels, cluster_centers, inertia, optimal_k = perform_kmeans(fake_data, num_clusters)
        center_df = pd.DataFrame(cluster_centers, columns=cluster_columns)
        cluster_names = [f'Cluster {i}' for i in range(optimal_k)]
        center_df['Name'] = cluster_names
        engineering_df = pd.read_excel("Engineering_data.xlsx")  # check df session control
        cluster_df = center_df

        st.subheader('Cluster Centers', anchor='cluster-centers')
        cluster_df["index_num"] = cluster_df.index
        figures = {}

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
            figures[column] = fig

        selected_graph = st.selectbox("Select Graph", cluster_columns)
        st.plotly_chart(figures[selected_graph])

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

    except ValueError as e:
        st.error("Please select atleast one column for clustering")

def fitting_group_visualisation():
    st.header("Synthetic Data")
    main_df, df1 = df_fitting_and_evaluation()
    st.dataframe(df1, width=1000)
    st.header("K-means")
    col1, col2 = st.columns(2)
    with col1:
        # kmeans clustering for fitting distance
        df_for_clustering = main_df.drop(columns=['ID', 'Evaluation'])
        fitting_distance_data = df_for_clustering['fitting_distance'].values.reshape(-1, 1)
        # Specify the number of clusters
        n_clusters = 3

        # Initialize KMeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        # Fit the model to the data
        kmeans.fit(fitting_distance_data)

        # Add cluster labels to the DataFrame
        df_for_clustering['cluster_label'] = kmeans.labels_
        # st.write(df_for_clustering)
        # if df_for_clustering['']
        # df_for_clustering['cluster_name_cm'] =

        # Display the cluster centers
        # st.write("Cluster centers:")
        # st.write(kmeans.cluster_centers_)
        centers = []
        for x in kmeans.cluster_centers_:
            for y in x:
                centers.append(y)
        print(centers)
        # Plotting using Plotly
        fig = px.scatter(df_for_clustering, y='fitting_distance', color='cluster_label',
                         title='KMeans Clustering based on Fitting Distance')

        # Plotting cluster centers
        fig.add_scatter(y=centers, x=[500] * n_clusters, mode='markers',
                        marker=dict(color='black', size=10), name='Cluster Centers')

        # st.write(df_for_clustering['cluster_label'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # fitting group visualization kmeans
        df_vis = main_df.loc[:, ['ID', 'fitting_distance', 'fitting_group']]
        # st.dataframe(df_vis)
        # Plotting using Plotly
        fig = px.scatter(df_vis, x='ID', y='fitting_distance', color='fitting_group',
                         title='Fitting Distance vs Number of Points',
                         labels={'ID': 'Number of Points', 'fitting_distance': 'Fitting Distance'})

        # Customize the layout
        fig.update_layout(showlegend=True)

        # confusion_matrix(y_true, y_pred)

        st.plotly_chart(fig, use_container_width=True)


def fitting_group_visualisation_dbscan():
    main_df, df1 = df_fitting_and_evaluation()
    df_vis = main_df.loc[:, ['ID', 'fitting_distance', 'fitting_group']]

    st.header("DBSCAN")
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=0.2, min_samples=3).fit(df_vis[['fitting_distance']])

    # Add cluster labels to the DataFrame
    cluster_labels = clustering.labels_
    unique_clusters = np.unique(cluster_labels)
    if -1 in unique_clusters:
        unique_clusters = unique_clusters[1:]
    label_mapping = {cluster: i for i, cluster in enumerate(unique_clusters)}
    df_vis['cluster_label'] = cluster_labels
    df_vis['cluster_label'] = df_vis['cluster_label'].map(label_mapping).fillna(-1).astype(int)
    # st.write(df_vis)
    # Plotting using Plotly
    fig = px.scatter(df_vis, x='ID', y='fitting_distance', color='cluster_label',
                     title='Fitting Distance vs Number of Points (DBSCAN Clustering)',
                     labels={'ID': 'Number of Points', 'fitting_distance': 'Fitting Distance',
                             'cluster_label': 'Cluster'})

    # Plotting using Plotly
    fig = px.scatter(df_vis, x='ID', y='fitting_distance', color='cluster_label',
                     title='Fitting Distance vs Number of Points (DBSCAN Clustering)',
                     labels={'ID': 'Number of Points', 'fitting_distance': 'Fitting Distance',
                             'cluster_label': 'Cluster'})

    # Customize the layout
    fig.update_layout(showlegend=True)

    # Show the plot
    st.plotly_chart(fig)

def decision_tree_viz():
    preci_value, recall_value, accuracy_value, classification_report_val, confusion_matrix_test = Decision_Tress()
    tab1, tab2 = st.tabs(["Confusion-Matrix", "Evaluation-Metrics"])
    preci_value = round(preci_value, 4)
    recall_value = round(recall_value, 4)
    accuracy_value = round(accuracy_value, 4)

    with tab1:
        # Create a DataFrame with label mappings
        confusion_matrix_df = pd.DataFrame(confusion_matrix_test, index=['Transition', 'Excess', 'Clearance'],
                                       columns=['Transition', 'Excess', 'Clearance'])

        # Define the labels for rows and columns
        labels = ['Clearance', 'Excess', 'Transition']

        # Create the confusion matrix plot using Plotly
        fig = ff.create_annotated_heatmap(z=confusion_matrix_df.values, x=labels, y=labels, colorscale='Blues')

        # Add title and axis labels
        fig.update_layout(title='Confusion Matrix',
                          xaxis=dict(title='Actual  Values'),
                          yaxis=dict(title='Predicted Values'))

        # Display the Plotly figure in Streamlit
        st.plotly_chart(fig)
        # confusion_matrix_df = pd.concat([pd.DataFrame(confusion_matrix_df)], axis=1)
        with tab2:
            st.header("Evaluation-Metrics")

            # Accuracy Progress Bar
            # Accuracy Progress Bar
            accuracy_color = "#87CEEB"  # Pastel Blue
            progress_text = f"<div class='progress-bar-text'>Accuracy Value</div>"
            st.write(progress_text, unsafe_allow_html=True)
            st.markdown("""
                <style>
                .progress-container.accuracy {
                    position: relative;
                    width: 100%;
                    background-color: #f0f0f0;
                    border-radius: 5px;
                    padding: 2px;
                }
                .progress-bar.accuracy {
                    position: relative;
                    width: 0%;
                    height: 30px;
                    background-color: """ + accuracy_color + """;
                    text-align: center;
                    line-height: 30px;
                    color: white;
                    border-radius: 5px;
                }
                .progress-bar-text {
                    position: absolute;
                    top: 300%;
                    left :7%;
                    transform: translate(-50%, -50%);
                
                    color: black;  /* Change the color as needed */
                }
                </style>
                """, unsafe_allow_html=True)

            # Calculate percentage for Accuracy
            percentage_accuracy = accuracy_value * 100
            percentage_accuracy = round(percentage_accuracy, 2)
            # Display Accuracy progress bar
            st.markdown(f"""
                <div class="progress-container accuracy">
                    <div class="progress-bar accuracy" style="width: {percentage_accuracy}%;">
                        {percentage_accuracy}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Precision Progress Bar
            precision_color = "#FFD700"  # Pastel Yellow
            progress_text = f"<div style='position:relative;padding-top: 10px;'>Precision Value</div>"
            st.write(progress_text, unsafe_allow_html=True)
            st.markdown("""
                        <style>
                        .progress-container.precision {
                            width: 100%;
                            background-color: #f0f0f0;
                            border-radius: 5px;
                            padding: 2px; /* Increased padding for more space */
                        }
                        .progress-bar.precision {
                            width: 0%;
                            height: 30px;
                            background-color: """ + precision_color + """;
                            text-align: center;
                            line-height: 30px;
                            color: white;
                            border-radius: 5px;
                        }
                        </style>
                        """, unsafe_allow_html=True)

            # Calculate percentage for Precision
            percentage_preci = preci_value * 100
            percentage_preci = round(percentage_preci, 2)
            # Display Precision progress bar
            st.markdown(f"""
                        <div class="progress-container precision">
                            <div class="progress-bar precision" style="width: {percentage_preci}%">
                                {percentage_preci}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # Recall Progress Bar
            recall_color = "#FFA07A"  # Pastel Salmon
            progress_text = f"<div style='position:relative;padding-top: 10px;'>Recall Value</div>"

            st.write(progress_text, unsafe_allow_html=True)
            st.markdown("""
                        <style>
                        .progress-container.recall {
                            width: 100%;
                            background-color: #f0f0f0;
                            border-radius: 5px;
                            padding: 2px; /* Increased padding for more space */
                        }
                        .progress-bar.recall {
                            width: 0%;
                            height: 30px;
                            background-color: """ + recall_color + """;
                            text-align: center;
                            line-height: 30px;
                            color: white;
                            border-radius: 5px;
                        }
                        </style>
                        """, unsafe_allow_html=True)

            # Calculate percentage for Recall
            percentage_recall = recall_value * 100
            percentage_recall = round(percentage_recall, 2)
            # Display Recall progress bar
            st.markdown(f"""
                        <div class="progress-container recall">
                            <div class="progress-bar recall" style="width: {percentage_recall}%">
                                {percentage_recall}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        # Accuracy Progress Bar


    # st.write(preci_value, recall_value, accuracy_value)
def main():
    st.markdown('<h1 style="text-align: center;">Box and Cylinder Analysis</h1>', unsafe_allow_html=True)

    sections = {'Bar-Chart': 'Bar-Chart', 'Plot': 'Plot', 'K-Means': 'k-means', 'Decision-Trees':'Decision-Trees'}

    st.sidebar.title('PMV4')
    selected_nav = st.sidebar.selectbox("Navigate to", list(sections.keys()), key='navigation')

    if selected_nav:
        st.session_state.selected_nav = sections[selected_nav]

    nav = st.session_state.selected_nav

    if nav == 'Bar-Chart':
        # bar_chart()
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
        fitting_group_visualisation()
        fitting_group_visualisation_dbscan()
        kmeans()
    elif nav == 'Decision-Trees':
        decision_tree_viz()




if __name__ == "__main__":
    main()