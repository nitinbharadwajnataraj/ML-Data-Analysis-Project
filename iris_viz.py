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
from Probabilistic_DT import Probabilistic_Decision_Tree
from Iris_PDT import Probabilistic_Decision_Tree_Iris, df_fitting_and_evaluation_iris
import joblib
import streamlit_flow
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout
import json
import json as json_lib 
from openai import OpenAI

def rename_dataframe_columns(df):
    # Rename the columns to match the expected feature names
    df = df.rename(columns={
        'SepalLengthCm': 'Sepal Length',
        'SepalWidthCm': 'Sepal Width ',
        'PetalLengthCm': 'Petal Length',
        'PetalWidthCm': 'Petal Width'
    })
    return df

def load_model_iris_pdt():
    # Load the saved decision tree model
    iris_probabilistic_decision_tree_model = joblib.load('probabilistic_decision_tree_model_Iris.joblib')
    return iris_probabilistic_decision_tree_model

def predict_input_pdt(df_input_val):
    probabilistic_decision_tree_model = load_model_iris_pdt()
    predictions = probabilistic_decision_tree_model.predict(df_input_val)
    return predictions

def display_prediction(predictions):
    if predictions == 0:
        predictions_val = "Iris-Setosa"
        color = "red"
    elif predictions == 1:
        predictions_val = "Iris-Versicolor"
        color = "green"
    else:
        predictions_val = "Iris-Virginica"
        color = "blue"

    predict = "Prediction: "
    st.markdown(f' <p  style="color:{color};font-size:20px;">{predict}{predictions_val}</p>', unsafe_allow_html=True)

def get_table_download_link():
    df = pd.DataFrame({
        'SepalLengthCm': [4.9, 4.7, 7.0, 6.3],
        'SepalWidthCm': [3.0, 3.2, 3.2, 3.3],
        'PetalLengthCm': [1.4, 1.6, 4.7, 6.0],
        'PetalWidthCm': [0.2, 0.2, 1.4, 2.5]
    })

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    encoded_data = b64encode(processed_data).decode()
    download_link = f'<a href="data:application/octet-stream;base64,{encoded_data}" download="data.xlsx">Download Sample Excel file</a>'
    return download_link

def llm_analysis(json,sess_state):
    if f"llm_answer_{sess_state}" not in st.session_state:
        st.session_state[f"llm_answer_{sess_state}"] = None
    st.header("AI-Powered Analysis")
    prompt = st.text_area("Describe your Question:", 
    placeholder="Example: How many nodes leads to OK leaf Node\n", key=f"prompt_input_{sess_state}")

    if st.button("Generate AI-Based Analysis"):
        client = OpenAI(api_key="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", base_url="https://api.perplexity.ai")
        answer = generate_analysis_from_llm(prompt, client, json)
        if answer:
            st.session_state[f"llm_answer_{sess_state}"] = answer
    if st.session_state[f"llm_answer_{sess_state}"]:
        st.markdown("### LLM Answer:")
        st.write(st.session_state[f"llm_answer_{sess_state}"])
            

def generate_analysis_from_llm(prompt, client, tree_json):
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj

    # Clean and stringify the JSON
    cleaned_json = convert_numpy(tree_json)
    tree_str = json.dumps(cleaned_json, indent=2)

    # Create the system prompt with JSON inside as plain text
    system_prompt = (
        "You are an AI specializing in analysing JSON-based decision trees. "
        "Your task is to answer user questions based only with the provided decision tree below.\n\n"
        "Here is the decision tree:\n" + tree_str + "You should only give the answer from the context of the provided decision tree.\n\n "
        "DO not provide the json structure and id in the answer. Let it be a natural language and do not use the word json or id in the answer"
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt + "according to the provided decision tree"
        }
    ]

    try:
        response = client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=messages
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error generating Analysis: {e}")
        return None

def iris_probabilistic_decision_tree_viz(depth):
    preci_value, recall_value, accuracy_value, classification_report_val, confusion_matrix_test, dtc, feature_names = Probabilistic_Decision_Tree_Iris(depth)
    tab0, tab1, tab2, tab4, tab5 = st.tabs(
        ["User Prediction", "Confusion-Matrix", "Evaluation-Metrics", "Iris PDT Visualization", "Analysis"])
    preci_value = round(preci_value, 4)
    recall_value = round(recall_value, 4)
    accuracy_value = round(accuracy_value, 4)
    with tab0:
        option = st.radio("Select input method", ("Manual Input", "Upload Excel File"))

        if option == "Manual Input":
            with st.form("prediction_form"):
                st.write("Enter the input values:")
                number1 = st.number_input("Enter Sepal Length in CM", step=0.1, format="%.2f", value=5.1)
                number2 = st.number_input("Enter Sepal Width in CM", step=0.1, format="%.2f", value=3.5)
                number3 = st.number_input("Enter Petal Length in CM", step=0.1, format="%.2f", value=1.4)
                number4 = st.number_input("Enter Petal Width in CM", step=0.1, format="%.2f", value=0.2)

                # material_supplier = st.selectbox("Select Material Supplier",
                #                                  options=list(MATERIAL_SUPPLIER_MAPPING.keys()))

                submitted = st.form_submit_button("Make Prediction")
                if submitted:
                    # material_supplier_value = MATERIAL_SUPPLIER_MAPPING[material_supplier]
                    df_input_val = pd.DataFrame(
                        [[number1, number2, number3, number4]],
                        columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
                    #df_input_val = rename_dataframe_columns(df_input_val)
                    prediction = predict_input_pdt(df_input_val)
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
                    #df_input_val = rename_dataframe_columns(df_input_val)
                    predictions = predict_input_pdt(df_input_val)
                    prediction_labels = ['Iris-Setosa' if pred == 0 else 'Iris-Versicolor' if pred == 1 else 'Iris-Virginica' for pred in predictions]
                    df_input_val['Prediction'] = prediction_labels

                    def apply_color(val):
                        if val == 'Iris-Setosa':
                            color = 'background-color: rgba(255, 0, 0, 0.3)'
                        elif val == 'Iris-Versicolor':
                            color = 'background-color: rgba(0, 255, 0, 0.3)'
                        else:
                            color = 'background-color: rgba(0, 0, 255, 0.3)'
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
        confusion_matrix_df = pd.DataFrame(confusion_matrix_test, index=['Setosa', 'Versicolor', 'Virginica'],
                                           columns=['Setosa', 'Versicolor', 'Virginica'])
        # Define the labels for rows and columns
        labels = ['Setosa', 'Versicolor', 'Virginica']

        # Create the confusion matrix plot using Plotly
        fig = ff.create_annotated_heatmap(z=confusion_matrix_df.values, x=labels, y=labels, colorscale='Blues')

        # Add title and axis labels
        fig.update_layout(title='Confusion Matrix for Test Results',
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
        #df_bar_chart_Evaluation_PDT()
        #df_bar_chart_fitting_group_PDT()

    with tab4:
        def visualize_iris_probabilistic_decision_tree(dtc, feature_names):
            nodes = []
            edges = []

            def node_id(idx): return f"node_{idx}"

            node_content_map = {}
            edge_label_map = {}

            path_filter = st.selectbox("🔍 Filter & Show Paths", ["All", "Setosa", "Versicolor", "Virginica"])
            class_map = {"Setosa": 0, "Versicolor": 1, "Virginica": 2}

            def keep_subtree(node_idx, target_class):
                """Check if any leaf node under this node leads to the target class"""
                is_leaf = dtc.tree_.feature[node_idx] == -2
                if is_leaf:
                    values = dtc.tree_.value[node_idx][0]
                    predicted_class = dtc.classes_[values.argmax()]
                    return predicted_class == target_class
                left = dtc.tree_.children_left[node_idx]
                right = dtc.tree_.children_right[node_idx]
                return keep_subtree(left, target_class) or keep_subtree(right, target_class)

            def traverse(node_idx, depth=0, pos_x=100, pos_y=100):
                if path_filter in ["Setosa", "Versicolor", "Virginica"]:
                    if not keep_subtree(node_idx, class_map[path_filter]):
                        return

                is_leaf = dtc.tree_.feature[node_idx] == -2
                nid = node_id(node_idx)

                if is_leaf:
                    values = dtc.tree_.value[node_idx][0]
                    predicted_class = dtc.classes_[values.argmax()]
                    label = f"Predict: {'Setosa' if predicted_class == 0 else 'Versicolor' if predicted_class == 1 else 'Virginica'}"
                    color = "#90EE90" if predicted_class == 1 else "#FFCCCB"
                    color_map = {
                            0: "#FFCCCB",  # light red
                            1: "#90EE90",  # light green
                            2: "yellow"   
                        }
                    color = color_map.get(predicted_class, "#FFFFFF")
                else:
                    feature = feature_names[dtc.tree_.feature[node_idx]]
                    threshold = dtc.tree_.threshold[node_idx]
                    label = f"{feature} <= {threshold:.2f}"
                    color = "#ADD8E6"

                style = {
                    "backgroundColor": color,
                    "border": "2px solid black",
                    "opacity": "1.0"
                }

                nodes.append(StreamlitFlowNode(
                    id=nid,
                    pos=(pos_x, pos_y),
                    data={"content": label},
                    node_type="default",
                    source_position="right",
                    target_position="left",
                    draggable=True,
                    style=style
                ))
                node_content_map[nid] = label

                if not is_leaf:
                    left = dtc.tree_.children_left[node_idx]
                    right = dtc.tree_.children_right[node_idx]

                    spacing_x = 500
                    spacing_y = 300

                    if path_filter == "All" or keep_subtree(left, class_map.get(path_filter, None)):
                        traverse(left, depth + 1, pos_x + spacing_x, pos_y - spacing_y // (depth + 1))
                        edge_id = f"{nid}-{node_id(left)}"
                        edges.append(StreamlitFlowEdge(
                            id=edge_id,
                            source=nid,
                            target=node_id(left),
                            label="True",
                            animated=True,
                            style={"stroke": "green", "strokeWidth": "2px"}
                        ))
                        edge_label_map[edge_id] = "True"

                    if path_filter == "All" or keep_subtree(right, class_map.get(path_filter, None)):
                        traverse(right, depth + 1, pos_x + spacing_x, pos_y + spacing_y // (depth + 1))
                        edge_id = f"{nid}-{node_id(right)}"
                        edges.append(StreamlitFlowEdge(
                            id=edge_id,
                            source=nid,
                            target=node_id(right),
                            label="False",
                            animated=True,
                            style={"stroke": "red", "strokeWidth": "2px"}
                        ))
                        edge_label_map[edge_id] = "False"

            traverse(0)

            st.markdown("### 🌳 Decision Tree Visualization")

            if (
                "tree_flow_state_prob_dt_iris" not in st.session_state or
                st.session_state.get("last_filter_prob_dt_iris") != path_filter or
                st.session_state.get("last_depth_prob_dt_iris") != dtc.get_depth()
            ):
                st.session_state.tree_flow_state_prob_dt_iris = StreamlitFlowState(nodes, edges)
                st.session_state.last_filter_prob_dt_iris = path_filter
                st.session_state.last_depth_prob_dt_iris = dtc.get_depth()

            updated_state = streamlit_flow(
                'decision_tree_flow',
                st.session_state.tree_flow_state_prob_dt_iris,
                fit_view=True,
                get_node_on_click=True,
                get_edge_on_click=True
            )

            selected_id = updated_state.selected_id
            if selected_id in node_content_map:
                st.success(f"Clicked Node Content: {node_content_map[selected_id]}")
            elif selected_id in edge_label_map:
                st.success(f"Clicked Edge Label: {edge_label_map[selected_id]}")
            else:
                st.info("Click on a node or edge to see its value.")
            # Extract full tree as JSON
            def extract_tree_json(node_idx):
                is_leaf = dtc.tree_.feature[node_idx] == -2
                if is_leaf:
                    values = dtc.tree_.value[node_idx][0]
                    predicted_class = dtc.classes_[values.argmax()]
                    return {
                        "id": node_idx,
                        "type": "leaf",
                        "prediction": "Setosa" if predicted_class == 0 else "Versicolor" if predicted_class == 1 else "Virginica",
                        "samples": int(sum(values)),
                        "class_distribution": values.tolist()
                    }
                else:
                    feature = feature_names[dtc.tree_.feature[node_idx]]
                    threshold = dtc.tree_.threshold[node_idx]
                    left_idx = dtc.tree_.children_left[node_idx]
                    right_idx = dtc.tree_.children_right[node_idx]
                    return {
                        "id": node_idx,
                        "type": "split",
                        "feature": feature,
                        "threshold": threshold,
                        "left": extract_tree_json(left_idx),
                        "right": extract_tree_json(right_idx)
                    }
            tree_json = extract_tree_json(0)
            return tree_json
        
        preci_value, recall_value, accuracy_value, classification_report_val, confusion_matrix_test, dtc, feature_names = Probabilistic_Decision_Tree_Iris(depth)
        json = visualize_iris_probabilistic_decision_tree(dtc, feature_names)
        llm_analysis(json, "iris")
        
    with tab5:
        st.header("Analyse via Image")

        # Import required libraries
        from streamlit_cropperjs import st_cropperjs
        import base64
        from io import BytesIO
        import time
        from PIL import Image
        import numpy as np

        # Initialize session state for tree screenshot
        if 'tree_screenshot_iris' not in st.session_state:
            st.session_state['tree_screenshot_iris'] = None

        # Image upload section
        uploaded_file = st.file_uploader(
            "Upload an Image (only .jpg, .jpeg, .png allowed)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            # Read the image and convert to a format suitable for cropperjs
            image_bytes = uploaded_file.getvalue()
            st.session_state['original_image_iris'] = image_bytes
            
            # Display and crop image using cropperjs
            cropped_img = st_cropperjs(
                img_file=uploaded_file,
                box_color='red',
                aspect_ratio=None,
                return_type='bytes'
            )
            
            if cropped_img:
                # Show the cropped image
                st.image(cropped_img, caption="Processed Image", use_container_width=True)
                
                # Save the cropped image for download
                st.session_state['cropped_image_iris'] = cropped_img
                
                # Add download button for the cropped image
                st.download_button(
                    label="Download Processed Image",
                    data=cropped_img,
                    file_name=f"processed_image_{int(time.time())}.png",
                    mime="image/png"
                )
                
                # Add image analysis functionality here
                st.info("Image analysis will be performed here. You can add your image processing code.")
                
                # Example placeholder for image analysis results
                with st.expander("Image Analysis Results", expanded=False):
                    st.write("Your image analysis results will appear here.")

        st.markdown("---")

        # Tree Canvas Management
        if 'show_tree_canvas_pdt_iris' not in st.session_state:
            st.session_state['show_tree_canvas_pdt_iris'] = False

        # Toggle button for tree canvas
        if not st.session_state['show_tree_canvas_pdt_iris']:
            if st.button("Create your own Tree"):
                st.session_state['show_tree_canvas_pdt_iris'] = True
                st.rerun()
        else:
            if st.button("Close Tree"):
                st.session_state['show_tree_canvas_pdt_iris'] = False
                st.rerun()

        # Tree Canvas Section
        if st.session_state['show_tree_canvas_pdt_iris']:
            # Initialize canvas state if not present
            if 'canvas_state_dt_iris' not in st.session_state:
                st.session_state.canvas_state_dt_iris = StreamlitFlowState([], [])

            # Create a container for our tree with a unique key for capturing
            tree_container = st.container()
            with tree_container:
                # Apply custom CSS to remove padding/margin
                st.markdown("""
                    <style>
                        #flow-container {
                            background-color: white;
                            padding: 10px;
                            margin: 0;
                            border-radius: 8px;
                            border: 1px solid #ddd;
                        }
                        
                        .element-container {
                            margin-top: 0;
                            padding-top: 0;
                        }
                    </style>
                """, unsafe_allow_html=True)
                
                # This div will be our capture target
                st.markdown('<div id="flow-container">', unsafe_allow_html=True)
                st.info('Right click on the canvas to add Nodes and Edges')

                # Draw flow component
                st.session_state.canvas_state_dt_iris = streamlit_flow(
                    key='fully_interactive_flow',
                    state=st.session_state.canvas_state_dt_iris,
                    fit_view=True,
                    show_controls=True,
                    allow_new_edges=True,
                    animate_new_edges=True,
                    layout=TreeLayout("right"),
                    enable_pane_menu=True,
                    enable_edge_menu=True,
                    enable_node_menu=True,
                )

                # Debug section to understand node structure
                if st.checkbox("Debug Node Structure", False):
                    st.subheader("Node Structure Information")
                    
                    node_debug_list = []
                    for node in st.session_state.canvas_state_dt_iris.nodes:
                        # Get node data fields
                        data_fields = {}
                        if hasattr(node, 'data') and isinstance(node.data, dict):
                            data_fields = node.data
                        
                        # Extract what would be displayed in visualization
                        display_label = None
                        for field in ['label', 'text', 'content', 'name', 'title', 'value']:
                            if field in data_fields and data_fields[field]:
                                display_label = data_fields[field]
                                break
                                
                        # Add to debug list
                        node_debug_list.append({
                            "Node ID": node.id,
                            "Display Label": display_label,
                            "Data Fields": data_fields
                        })
                    
                    # Show the node information
                    for i, node_info in enumerate(node_debug_list):
                        with st.expander(f"Node {i+1}: {node_info['Display Label'] or node_info['Node ID']}", expanded=False):
                            st.json(node_info)

                st.markdown('</div>', unsafe_allow_html=True)

            # Metrics display
            col1, col2 = st.columns(2)
            col1.metric("Nodes", len(st.session_state.canvas_state_dt_iris.nodes))
            col2.metric("Edges", len(st.session_state.canvas_state_dt_iris.edges))

            st.markdown("---")

                                # Create a PNG representation of the tree (server-side)
            if len(st.session_state.canvas_state_dt_iris.nodes) > 0:
                try:
                    import matplotlib.pyplot as plt
                    import networkx as nx
                    import re
                    
                    # Function to clean node IDs
                    def clean_node_id(node_id):
                        # Remove common ReactFlow prefixes
                        if isinstance(node_id, str):
                            # Remove prefixes like "node_", "st-flow-node_" etc.
                            clean_id = re.sub(r'^(node_|st-flow-node_)', '', node_id)
                            # Shorten long IDs (often hash-like strings)
                            if len(clean_id) > 12:
                                clean_id = clean_id[:8] + "..."
                            return clean_id
                        return str(node_id)
                    
                    # Create a graph from the tree data
                    G = nx.DiGraph()
                    
                    # Debug the node structure to understand what's available
                    node_debug_info = {}
                    
                    # Improved node label extraction
                    for node in st.session_state.canvas_state_dt_iris.nodes:
                        node_id = str(node.id)
                        
                        # Initialize default label as None
                        label = None
                        
                        # Extract label from ReactFlow node structure
                        if hasattr(node, 'data') and isinstance(node.data, dict):
                            # ReactFlow typically stores the visible text content in these fields
                            # Try all common places where ReactFlow stores node labels
                            for field in ['label', 'text', 'content', 'name', 'title', 'value']:
                                if field in node.data and node.data[field]:
                                    label = node.data[field]
                                    break
                            
                            # For nodes with HTML content, try to extract the text
                            if 'html' in node.data and node.data['html']:
                                try:
                                    html_content = node.data['html']
                                    # Simple text extraction from HTML (can be improved if needed)
                                    import re
                                    text_content = re.sub(r'<[^>]+>', '', html_content).strip()
                                    if text_content:
                                        label = text_content
                                except Exception:
                                    pass
                            
                            # Store all node attributes for debugging
                            node_debug_info[node_id] = {
                                'data': node.data,
                                'extracted_label': label
                            }
                        
                        # If still no label found, look directly at node attributes
                        if not label and hasattr(node, 'label'):
                            label = node.label
                            
                        # If still no label found, check if there's a 'text' field directly on the node
                        if not label and hasattr(node, 'text'):
                            label = node.text
                            
                        # If still no label found or it's empty, use a simplified node ID as label
                        if not label:
                            # Clean up the node ID for display
                            label = clean_node_id(node_id)
                            if not label:
                                label = f"Node {len(G.nodes) + 1}"
                        
                        # Add node with its clean label as an attribute
                        G.add_node(node_id, label=label)
                    
                    # Log debug info to help diagnose
                    st.session_state['node_debug_info_iris'] = node_debug_info
                    
                    # Add edges
                    for edge in st.session_state.canvas_state_dt_iris.edges:
                        G.add_edge(edge.source, edge.target)
                    
                    # Create a figure and draw the graph
                    plt.figure(figsize=(12, 8), facecolor='white')
                    
                    # Use better layout for trees
                    # Try different layouts to see which works best
                    if len(G.nodes) > 1:
                        try:
                            # For tree-like structures, hierarchical layout works best
                            pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB')
                        except Exception:
                            try:
                                # Fall back to Sugiyama layout for directed graphs
                                pos = nx.multipartite_layout(G, subset_key=lambda node: G.in_degree(node))
                            except Exception:
                                # Last resort: spring layout
                                pos = nx.spring_layout(G, k=0.5, iterations=100)
                    else:
                        # With just one node, spring layout is fine
                        pos = nx.spring_layout(G, k=0.15, iterations=20)
                    
                    # Draw nodes with better styling
                    nx.draw_networkx_nodes(G, pos, 
                                          node_color='lightblue', 
                                          node_size=3000, 
                                          edgecolors='black', 
                                          alpha=0.8)
                    
                    # Draw edges with arrows - make them visible
                    nx.draw_networkx_edges(G, pos, 
                                          arrows=True,
                                          arrowsize=20,
                                          width=2,
                                          edge_color='black',
                                          alpha=0.8)
                    
                    # Use ONLY the extracted clean labels for node labels
                    clean_labels = {node_id: G.nodes[node_id]['label'] for node_id in G.nodes}
                    nx.draw_networkx_labels(G, pos, 
                                           labels=clean_labels, 
                                           font_size=12, 
                                           font_family='sans-serif',
                                           font_weight='bold')
                    
                    # Better save handling with background and improved resolution
                    plt.tight_layout()
                    plt.axis('off')  # Turn off axis
                    
                    # Add a title that includes the number of nodes
                    plt.title(f"Decision Tree ({len(G.nodes)} nodes)", fontsize=16)
                    
                    # Save with white background and higher DPI for better quality
                    buf = BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', 
                               dpi=150, facecolor='white')
                    buf.seek(0)
                    plt.close()
                    
                    # Save the visualization to session state
                    st.session_state['tree_screenshot_dt_iris'] = buf.getvalue()
                    
                    # Show a preview of the download image
                    st.subheader("Preview of Downloadable Tree Image")
                    st.image(buf, caption="Tree Visualization", use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating tree visualization: {str(e)}")

            # Export functionality
            tree_data = {
                "nodes": [node.__dict__ for node in st.session_state.canvas_state_dt_iris.nodes],
                "edges": [edge.__dict__ for edge in st.session_state.canvas_state_dt_iris.edges]
            }

            # Make sure json is properly imported and not shadowed
            tree_json = json_lib.dumps(tree_data, indent=4)
            
            # Download buttons section
            st.subheader("Download Options")
            col1, col2 = st.columns(2)
            
            # JSON Download button
            col1.download_button(
                label="Download Tree Structure (JSON)",
                data=tree_json,
                file_name="my_tree_structure.json",
                mime="application/json"
            )
            
            # Image Download button (only show if we have a tree visualization)
            if st.session_state['tree_screenshot_iris'] is not None:
                col2.download_button(
                    label="Download Tree Image",
                    data=st.session_state['tree_screenshot_iris'],
                    file_name=f"tree_visualization_{int(time.time())}.png",
                    mime="image/png"
                )
            
        # Initialize hypothesis state
        if "submitted_hypotheses_iris" not in st.session_state:
            st.session_state["submitted_hypotheses_iris"] = []
        
        # Initialize a reset flag
        if "reset_form_dt_iris" not in st.session_state:
            st.session_state["reset_form_dt_iris"] = False
            
        # Initialize default values for form fields
        if "reset_form_dt_iris" in st.session_state and st.session_state["reset_form_dt_iris"]:
            form_defaults = {
                "failure_desc": "",
                "imp_params": [],
                "failure_name": "",
                "hypo_prob": "Medium",
                "fail_imp": "Medium"
            }
            # Reset the flag
            st.session_state["reset_form_dt_iris"] = False
        else:
            form_defaults = {
                "failure_desc": st.session_state.get("failure_desc_dt_iris", ""),
                "imp_params": st.session_state.get("imp_params_dt_iris", []),
                "failure_name": st.session_state.get("failure_name_dt_iris", ""),
                "hypo_prob": st.session_state.get("hypo_prob_dt_iris", "Medium"),
                "fail_imp": st.session_state.get("fail_imp_dt_iris", "Medium")
            }
        # Domain Hypothesis Form
        with st.container():
            with st.expander("💬 Domain Hypothesis", expanded=False):
                with st.form(key="domain_hypothesis_form_dt"):
                    st.markdown("### 📌 Domain Hypothesis Entry")

                    failure_description = st.text_area(
                        "📝 Describe the failure case", 
                        value=form_defaults["failure_desc"],
                        key="failure_desc_dt_iris"
                    )

                    important_parameters = st.multiselect(
                        "📊 Which parameters are most important?",
                        options=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Other"],
                        default=form_defaults["imp_params"],
                        key="imp_params_dt_iris"
                    )

                    failure_name = st.text_input(
                        "❗ Name this failure (e.g., 'Enlarged Sepal')", 
                        value=form_defaults["failure_name"],
                        key="failure_name_dt_iris"
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        hypothesis_probability = st.selectbox(
                            "🔮 How likely is this hypothesis?",
                            options=["High", "Medium", "Low"],
                            index=["High", "Medium", "Low"].index(form_defaults["hypo_prob"]),
                            key="hypo_prob_dt"
                        )

                    with col2:
                        failure_importance = st.selectbox(
                            "🔥 How important is this failure?",
                            options=["High", "Medium", "Low"],
                            index=["High", "Medium", "Low"].index(form_defaults["fail_imp"]),
                            key="fail_imp_dt"
                        )

                    col1, col2 = st.columns(2)
                    with col1:
                        submitted = st.form_submit_button("Submit Hypothesis")
                    with col2:
                        clear_form = st.form_submit_button("Clear Form")

                    # Handle form submission
                    if submitted:
                        # Validate inputs before submission
                        if not failure_description or not failure_name:
                            st.error("Please fill in the required fields: Description and Failure Name")
                        else:
                            # Get current timestamp using datetime module
                            from datetime import datetime
                            entry = {
                                "Description": failure_description,
                                "Important Parameters": ", ".join(important_parameters) if important_parameters else "None specified",
                                "Failure Name": failure_name,
                                "Hypothesis Probability": hypothesis_probability,
                                "Failure Importance": failure_importance,
                                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.session_state["submitted_hypotheses_iris"].append(entry)
                            st.success("✅ Hypothesis submitted successfully!")

                            st.markdown("#### 🧾 Summary of Hypothesis")
                            for k, v in entry.items():
                                if k != "Timestamp":  # Don't show timestamp in the summary
                                    st.markdown(f"- **{k}**: {v}")
                    
                    # Handle form clearing
                    if clear_form:
                        # Set the reset flag for the next rerun
                        st.session_state["reset_form_dt_iris"] = True
                        st.rerun()

        # Display and export submitted hypotheses
        if st.session_state["submitted_hypotheses_iris"]:
            with st.expander("View All Submitted Hypotheses", expanded=False):
                df_hypo = pd.DataFrame(st.session_state["submitted_hypotheses_iris"])
                st.dataframe(df_hypo)
                
            csv_data = pd.DataFrame(st.session_state["submitted_hypotheses_iris"]).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Download Hypotheses as CSV",
                data=csv_data,
                file_name="domain_hypotheses.csv",
                mime="text/csv"
            )