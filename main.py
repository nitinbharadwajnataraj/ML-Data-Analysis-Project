import plotly.figure_factory as ff
from io import BytesIO
from base64 import b64encode
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.cluster import KMeans, DBSCAN
from data_model import data_model
from Compute_fit import compute_fit, count_yes_no
from clustering.k_means import perform_kmeans
from Decision_Tress import Decision_Tress
import joblib

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

def actual_target_values():
    df = pd.read_excel("fake_data.xlsx")

def df_fitting_and_evaluation():
    df = pd.read_excel("fake_data.xlsx")
    df["fitting_distance"] = df["box_hole_diameter"] - df["cylinder_diameter"]

    # Using & instead of 'and'
    condition1 = (df["fitting_distance"] <= 1) & (df["fitting_distance"] >= -1)
    condition2 = (df["fitting_distance"] > 1)
    # condition3 = (df["bed_distance"])
    predicted_df = pd.read_excel("predicted_data.xlsx")
    predicted_df.index = df.index
    df['Prediction'] = predicted_df['Prediction']
    # Assigning values based on conditions
    df.loc[condition1, "Evaluation"] = 'OK'
    df.loc[condition1, "fitting_group"] = 'Transition'
    df.loc[condition2, "Evaluation"] = 'NOK'
    df.loc[condition2, "fitting_group"] = 'Clearance'
    df.loc[~(condition1 | condition2), "Evaluation"] = 'NOK'
    df.loc[~(condition1 | condition2), "fitting_group"] = 'Excess'

    # df_to_write = df.merge(predicted_df[["Prediction"]], on="ID", how="left")
    styled_df = df.style.map(color_code, subset=['Evaluation', 'fitting_group', 'Prediction'])
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
        if selected_column:
            st.write("Scatter Plot:")
            fig = go.Figure()
            # Separate traces for `yes` and `no` to enable legends
            yes_points = df_1[df_1["check"] == 'yes']
            no_points = df_1[df_1["check"] == 'no']

            # Add `yes` points
            fig.add_trace(go.Scatter(
                x=yes_points.index,
                y=yes_points[selected_column],
                mode='markers',
                name='OK '+'for the shape '+selected_shape+' and column '+selected_column,  # Legend entry for 'Yes'
                marker=dict(color=yes_color)
            ))

            # Add `no` points
            fig.add_trace(go.Scatter(
                x=no_points.index,
                y=no_points[selected_column],
                mode='markers',
                name='NOK '+'for the shape '+selected_shape+' and column '+selected_column,  # Legend entry for 'No'
                marker=dict(color=no_color)
            ))

            #colors = [yes_color if val == 'yes' else no_color for val in df_1["check"]]
            #fig.add_trace(go.Scatter(x=df_1.index, y=df_1[selected_column], mode='markers', name=selected_column,
                                     #marker=dict(color=colors)))

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
                text="Min",
                showarrow=False,
                font=dict(size=12, color="red")
            )
            fig.add_annotation(
                x=df_1.index.min(),
                y=max_value,
                text="Max",
                showarrow=False,
                font=dict(size=12, color="red")
            )
            fig.add_annotation(
                x=df_1.index.min(),
                y=target_value,
                text="Target",
                showarrow=False,
                font=dict(size=12, color="lime")
            )

            st.plotly_chart(fig,use_container_width=True)
        else:
            st.warning("Please select a column for the scatter plot.")


def box_plot():
    df = pd.read_excel("fake_data.xlsx")
    fig = go.Figure()
    for column in df.columns[1:]:
        fig.add_trace(go.Box(y=df[column], name=column))
    fig.update_layout(title='Box Plot(Understanding Synthetic Data)', xaxis_title='Parameters', yaxis_title='Values')
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
        # st.write(fake_data)

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
    cluster_columns = st.multiselect('Select columns for clustering', columns, default=columns[0:4])
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
                fig.add_annotation(x=-0.25, y=target_value, text="Target", showarrow=False, font=dict(color="green"))
                fig.add_annotation(x=-0.25, y=min_value, text="Min", showarrow=False, font=dict(color="violet"))
                fig.add_annotation(x=-0.25, y=max_value, text="Max", showarrow=False, font=dict(color="red"))

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
    #st.header("Synthetic Data")
    main_df, df1 = df_fitting_and_evaluation()
    #st.dataframe(df1, width=1000)
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


# MATERIAL_SUPPLIER_MAPPING = {
#     'Supplier_A': 0,
#     'Supplier_B': 1,
#     'Supplier_C': 2
# }



def rename_dataframe_columns(df):
    # Rename the columns to match the expected feature names
    df = df.rename(columns={
        'Box hole diameter': 'box_hole_diameter',
        'Box hole depth': 'box_hole_depth',
        'Cylinder diameter': 'cylinder_diameter',
        'Cylinder height': 'cylinder_height',
        'Wire diameter': 'wire_diameter',
        'Bed distance': 'bed_distance',
        # 'Material Supplier': 'Material_seller'
    })
    return df

def decision_tree_viz():
    preci_value, recall_value, accuracy_value, classification_report_val, confusion_matrix_test = Decision_Tress()
    tab0, tab1, tab2, tab4 = st.tabs(
        ["User Prediction", "Confusion-Matrix", "Evaluation-Metrics", "Decision Tree Visualization"])
    preci_value = round(preci_value, 4)
    recall_value = round(recall_value, 4)
    accuracy_value = round(accuracy_value, 4)
    with tab0:
        option = st.radio("Select input method", ("Manual Input", "Upload Excel File"))

        if option == "Manual Input":
            with st.form("prediction_form"):
                st.write("Enter the input values:")
                number1 = st.number_input("Enter Box hole diameter", step=0.1, format="%.2f", value=30.099909)
                number2 = st.number_input("Enter Box hole depth", step=0.1, format="%.2f", value=37.075133)
                number3 = st.number_input("Enter Cylinder diameter", step=0.1, format="%.2f", value=29.515288)
                number4 = st.number_input("Enter Cylinder height", step=0.1, format="%.2f", value=28.470196)
                number5 = st.number_input("Enter Wire diameter", step=0.1, format="%.2f", value=1.724816)
                number6 = st.number_input("Enter Bed distance", step=0.1, format="%.2f", value=0.181060)

                # material_supplier = st.selectbox("Select Material Supplier",
                #                                  options=list(MATERIAL_SUPPLIER_MAPPING.keys()))

                submitted = st.form_submit_button("Make Prediction")
                if submitted:
                    # material_supplier_value = MATERIAL_SUPPLIER_MAPPING[material_supplier]
                    df_input_val = pd.DataFrame(
                        [[number1, number2, number3, number4, number5, number6]],
                        columns=['Box hole diameter', 'Box hole depth', 'Cylinder diameter',
                                 'Cylinder height', 'Wire diameter', 'Bed distance'])
                    df_input_val = rename_dataframe_columns(df_input_val)
                    prediction = predict_input(df_input_val)
                    display_prediction(prediction)

        elif option == "Upload Excel File":
            uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["xlsx", "xls", "csv"])
            if uploaded_file is not None:
                file_ext = uploaded_file.name.split('.')[-1]
                if file_ext.lower() in ['xlsx', 'xls']:
                    df_input_val = pd.read_excel(uploaded_file)
                elif file_ext.lower() == 'csv':
                    df_input_val = pd.read_csv(uploaded_file)
                else:
                    st.error("Unsupported file format. Please upload an Excel (xlsx/xls) or CSV (csv) file.")
                    return


                # if 'Material Supplier' in df_input_val.columns:
                #     df_input_val['Material Supplier'] = df_input_val['Material Supplier'].apply(
                #         lambda x: MATERIAL_SUPPLIER_MAPPING[x] if isinstance(x,
                #                                                              str) and x in MATERIAL_SUPPLIER_MAPPING else x
                #     )
                #     st.write("Data uploaded:")
                #     st.write(df_input_val)
                if st.button("Make Prediction"):
                    df_input_val = rename_dataframe_columns(df_input_val)
                    predictions = predict_input(df_input_val)
                    prediction_labels = ['NOK' if pred == 0 else 'OK' for pred in predictions]
                    df_input_val['Prediction'] = prediction_labels

                    def apply_color(val):
                        if val == 'OK':
                            color = 'background-color: rgba(0, 255, 0, 0.3)'
                        else:
                            color = 'background-color: rgba(255, 0, 0, 0.3)'
                        return color

                    df_input_val_styled = df_input_val.style.applymap(apply_color, subset=['Prediction'])

                    st.write("Predictions:")
                    st.dataframe(df_input_val_styled)
            else:
                st.error("Please upload an Excel (xlsx/xls) or CSV (csv) file.")
                # Provide a template for download
                st.markdown(get_table_download_link(), unsafe_allow_html=True)

    with tab1:
        # confusion_matrix_df = pd.DataFrame(confusion_matrix_test, index=['Transition', 'Excess', 'Clearance'],
        #                                columns=['Transition', 'Excess', 'Clearance'])
        confusion_matrix_df = pd.DataFrame(confusion_matrix_test, index=['OK', 'NOK'],
                                           columns=['OK', 'NOK'])
        # Define the labels for rows and columns
        labels = ['OK', 'NOK']

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
        df_bar_chart_Evaluation()
        df_bar_chart_fitting_group()

    with tab4:
        st.image("decision_tree_graphviz.png")


def load_model():
    # Load the saved decision tree model
    decision_tree_model = joblib.load('decision_tree_model.joblib')
    return decision_tree_model


def predict_input(df_input_val):
    decision_tree_model = load_model()
    predictions = decision_tree_model.predict(df_input_val)
    return predictions


def display_prediction(predictions):
    if predictions == 0:
        predictions_val = "NOK"
        color = "red"
    else:
        predictions_val = "OK"
        color = "green"

    predict = "Prediction: "
    # st.write("Prediction is ",
    #          unsafe_allow_html=True,
    #          )
    # Adding some color to the output for better visualization
    st.markdown(f' <p  style="color:{color};font-size:20px;">{predict}{predictions_val}</p>', unsafe_allow_html=True)


def get_table_download_link():
    df = pd.DataFrame({
        'Box hole diameter': [30.671085, 30.211838, 29.208569],
        'Box hole depth': [33.816286, 32.237568, 35.959047],
        'Cylinder diameter': [30.139838, 30.926533, 31.071050],
        'Cylinder height': [32.814482, 29.192808, 32.021252],

        'Wire diameter': [1.724487, 1.788960, 2.685131],
        'Bed distance': [0.055709, 0.169561, 0.197313]

    })

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    encoded_data = b64encode(processed_data).decode()
    download_link = f'<a href="data:application/octet-stream;base64,{encoded_data}" download="data.xlsx">Download Sample Excel file</a>'
    return download_link


def main():
    st.markdown('<h1 style="text-align: center;">Box and Cylinder Analysis</h1>', unsafe_allow_html=True)

    sections = {'Data Understanding': 'Data Understanding', 'K-Means': 'k-means', 'Decision Tree': 'Decision Tree'}

    # Define the options for the navigation menu
    options = list(sections.keys())

    # Use the option_menu for sidebar menu
    with st.sidebar:
        selected_nav = option_menu("PMV4 Analytics", options, default_index=0)

    # Map the selected option to the corresponding section
    selected_section = sections.get(selected_nav)

    # Display content based on the selected section
    if selected_section == 'Bar-Chart':
        df_bar_chart_Evaluation()
        df_bar_chart_fitting_group()

    elif selected_section == 'Data Understanding':
        st.header('Synthetic Dataset')
        main_df, df1 = df_fitting_and_evaluation()
        main_df.drop(columns=['fitting_distance','Prediction', 'Evaluation','fitting_group'], inplace=True)
        st.dataframe(main_df, hide_index=True, width=1250)
        st.header('Box-Cylinder Model Range ')
        df_engineering_data_from_xlsx = pd.read_excel('Engineering_data.xlsx')
        st.dataframe(df_engineering_data_from_xlsx, hide_index=True)
        tab1, tab2 = st.tabs(["Box-Plot", "Scatter-Plot"])
        with tab1:
            st.header("Box-Plot")
            box_plot()
        with tab2:
            st.header("Scatter-Plot")
            scatter_plot()

    elif selected_section == 'k-means':
        kmeans_info_popover()
        fitting_group_visualisation()
        #fitting_group_visualisation_dbscan()
        kmeans()

    elif selected_section == 'Decision Tree':
        st.header("Predictions for Synthetic Dataset")
        main_df, df1 = df_fitting_and_evaluation()
        st.dataframe(df1,hide_index=True,width=1250)
        decision_tree_viz()

if __name__ == "__main__":
    main()
