import graphviz
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
import joblib
import streamlit as st
import streamlit_flow
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout

st.set_page_config(page_title='DT Viz', layout='wide')

def df_fitting_and_evaluation():
    df = pd.read_excel("fake_data.xlsx")
    df["fitting_distance"] = df["box_hole_diameter"] - df["cylinder_diameter"]
    # Using & instead of 'and'
    condition1 = (df["fitting_distance"] <= 1) & (df["fitting_distance"] >= -1)
    condition2 = (df["fitting_distance"] > 1)
    # Assigning values based on conditions
    df.loc[condition1, "Evaluation"] = 'OK'
    # df.loc[condition1, "fitting_group"] = 'Transition'
    df.loc[condition2, "Evaluation"] = 'NOK'
    # df.loc[condition2, "fitting_group"] = 'Clearance'
    df.loc[~(condition1 | condition2), "Evaluation"] = 'NOK'
    # df.loc[~(condition1 | condition2), "fitting_group"] = 'Excess'

    return df


def prepare_DT_df():
    df = df_fitting_and_evaluation()
    #print("Original DataFrame:")
    #print(df.head())  # Check the original DataFrame

    # Drop unnecessary columns
    #print(df.columns)
    df = df.drop(columns=['ID', 'fitting_distance'])
    #print(df)
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    #
    # # Fit and transform the 'fitting_group' column
    df['evaluation_encoded'] = label_encoder.fit_transform(df['Evaluation'])
    #
    # # Mapping of original values to encoded values
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    #print("Label mapping:", label_mapping)

    # Drop the original 'fitting_group' column
    df = df.drop(columns=['Evaluation'])

    #print("DataFrame after preprocessing:")
    #print(df.head())  # Check the DataFrame after preprocessing

    return df


# def visualize_decision_tree(dtc, feature_names):
#     plt.figure(figsize=(50, 55))
#     tree.plot_tree(dtc, feature_names=feature_names, class_names=['OK', 'NOK'], filled=True, rounded=True,
#                    fontsize=12)
#     plt.title("Decision Tree Visualization", fontsize=20)
#     plt.xlabel("Features", fontsize=16)
#     plt.ylabel("Class", fontsize=16)
#     plt.savefig("decision_tree.png")  # Save the plot as an image
#     plt.show()

def visualize_decision_tree(dtc, feature_names):
    nodes = []
    edges = []

    def node_id(node_idx):
        return f"node_{node_idx}"

    # Maps for quick lookup
    node_content_map = {}
    edge_label_map = {}

    def traverse(node_idx, depth=0, pos_x=100, pos_y=100):
        if dtc.tree_.feature[node_idx] != -2:  # not a leaf node
            feature = feature_names[dtc.tree_.feature[node_idx]]
            threshold = dtc.tree_.threshold[node_idx]
            label = f"{feature} <= {threshold:.2f}"
        else:  # leaf node
            classes = dtc.classes_
            values = dtc.tree_.value[node_idx][0]
            predicted_class = classes[values.argmax()]
            if predicted_class == 1:
                label = "Predict: OK"
            else:
                label = "Predict: NOK"

        node = StreamlitFlowNode(
            id=node_id(node_idx),
            pos=(pos_x, pos_y),
            data={'content': label},
            node_type='default',
            source_position='right',
            target_position='left',
            draggable=True
        )
        nodes.append(node)

        node_content_map[node_id(node_idx)] = label

        if dtc.tree_.feature[node_idx] != -2:
            left_child = dtc.tree_.children_left[node_idx]
            right_child = dtc.tree_.children_right[node_idx]

            x_spacing = 500
            y_spacing = 300

            traverse(left_child, depth + 1, pos_x + x_spacing, pos_y - y_spacing // (depth + 1))
            traverse(right_child, depth + 1, pos_x + x_spacing, pos_y + y_spacing // (depth + 1))

            # Add edges
            left_edge_id = f"{node_id(node_idx)}-{node_id(left_child)}"
            right_edge_id = f"{node_id(node_idx)}-{node_id(right_child)}"

            edges.append(StreamlitFlowEdge(
                id=left_edge_id,
                source=node_id(node_idx),
                target=node_id(left_child),
                animated=True,
                label="True"
            ))
            edges.append(StreamlitFlowEdge(
                id=right_edge_id,
                source=node_id(node_idx),
                target=node_id(right_child),
                animated=True,
                label="False"
            ))

            edge_label_map[left_edge_id] = "True"
            edge_label_map[right_edge_id] = "False"

    traverse(0)

    if 'tree_flow_state' not in st.session_state:
        st.session_state.tree_flow_state = StreamlitFlowState(nodes, edges)

    updated_state = streamlit_flow('decision_tree_flow',
                                   st.session_state.tree_flow_state,
                                   fit_view=True,
                                   get_node_on_click=True,
                                   get_edge_on_click=True)

    # Detect if node or edge was clicked
    selected_id = updated_state.selected_id

    if selected_id in node_content_map:
        st.success(f"Clicked Node Content: {node_content_map[selected_id]}")
    elif selected_id in edge_label_map:
        st.success(f"Clicked Edge Label: {edge_label_map[selected_id]}")
    else:
        st.info("Click on a node or edge to see its value.")

def Decision_Tress():
    df = prepare_DT_df()

    X = df.iloc[:, 0:6]
    y = df.iloc[:, 6]
    #print(X, y)
    #print(X, y)

    x_main, x_test, y_main, y_test = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_main, y_main, test_size=0.2, random_state=17, stratify=y_main)

    dtc = DecisionTreeClassifier(criterion='entropy', max_depth=6)
    dtc.fit(x_main, y_main)

    # Save the model
    joblib.dump(dtc, 'decision_tree_model.joblib')
    # Convert class names to strings
    # class_names = df['evaluation_encoded'].unique().astype(str)

    # This is Validation
    y_pred_val = dtc.predict(x_val)
    #print("Confusion Matrics for Validation:")
    #print(confusion_matrix(y_val, y_pred_val))
    #print("classification_report for Validation:")
    #print(classification_report(y_val, y_pred_val))

    # this is for Testing
    y_pred_test = dtc.predict(x_test)
    #print("Confusion Matrics for TEST:")
    #print(confusion_matrix(y_test, y_pred_test))
    #print("classification_report of TEST:")
    #print(classification_report(y_test, y_pred_test))
    #print("Confusion Matrics for TEST:")
    #print(confusion_matrix(y_test, y_pred_test))
    print("classification_report of TEST:")
    print(classification_report(y_test, y_pred_test))
    classification_report_val = classification_report(y_test, y_pred_test)
    confusion_matrix_test = confusion_matrix(y_test, y_pred_test)
    preci_value = precision_score(y_test, y_pred_test, average='weighted')
    recall_value = recall_score(y_test, y_pred_test, average='weighted')
    accuracy_value = accuracy_score(y_test, y_pred_test)

    features = pd.DataFrame(dtc.feature_importances_, index=X.columns)
    feature_names = X.columns
    #print(features.head(6))
    #visualize_decision_tree(dtc, X.columns)
    # Function to visualize Decision Tree

    return preci_value, recall_value, accuracy_value, classification_report_val, confusion_matrix_test, dtc, feature_names


Decision_Tress()