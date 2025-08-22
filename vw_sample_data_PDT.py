import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
import joblib


def df_fitting_and_evaluation_vw_sample():
    df = pd.read_csv("Analysis_Data_augmented_csv.csv", sep=';',decimal=',',on_bad_lines='skip')
    return df


def prepare_DT_df_vw_sample():
    df = df_fitting_and_evaluation_vw_sample()
    #print("Original DataFrame:")
    #print(df.head())

    # Drop unnecessary columns
    #print(df.columns)
    df = df.drop(columns=['pin_position','stator_id','ProduktID', 'Pinbezeichnung','left_pin_id', 'right_pin_id','Drahtprüfung_Ergebnis_x', 'Pin_ID_x','Dachbiegen_Ergebnis_x', 'Pin_Type_x', '3D_Biegen_Ergebnis_x',
       'Abisolieren_eval_x', 'Drahtprüfung_Ergebnis_y', 'Pin_ID_y',
       'Dachbiegen_Ergebnis_y', 'Pin_Type_y', '3D_Biegen_Ergebnis_y'])
    print(df.head())
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    #
    # # Fit and transform the 'fitting_group' column
    df['Ergebnis_encoded'] = label_encoder.fit_transform(df['Ergebnis'])
    #
    # # Mapping of original values to encoded values
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label mapping:", label_mapping)

    # Drop the original 'fitting_group' column
    df = df.drop(columns=['Ergebnis'])
    # Keep only rows that have at least (total columns - 3) non-NaN values
    df = df.dropna(thresh=df.shape[1] - 3)
    #print('columns with obj type')
    #print(df.select_dtypes(include='object').columns)
    #print(df.select_dtypes(include=['number']).columns)
    print("DataFrame after preprocessing:")
    print(df.head())  # Check the DataFrame after preprocessing

    return df


def Probabilistic_Decision_Tree_VW_Sample(depth,selected_to_drop):
    df = prepare_DT_df_vw_sample()
    if selected_to_drop:
        df = df.drop(columns=selected_to_drop)
    #print(df.head())
    X = df.drop(columns=['Ergebnis_encoded'])
    y = df['Ergebnis_encoded']
    #print('x: ',X)
    #print('y: ',y)
    x_main, x_test, y_main, y_test = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_main, y_main, test_size=0.2, random_state=17, stratify=y_main)

    # Create a Decision Tree with probabilistic behavior
    dtc = DecisionTreeClassifier(
    criterion='entropy',
    random_state=0,
    class_weight="balanced",
    max_depth=depth
)
    dtc.fit(x_main, y_main)

    # Save the model
    joblib.dump(dtc, 'probabilistic_decision_tree_model_VW_Sample.joblib')

    # Instead of normal predict, use predict_proba + argmax to simulate probabilistic prediction
    y_proba_val = dtc.predict_proba(x_val)
    y_pred_val = y_proba_val.argmax(axis=1)

    y_proba_test = dtc.predict_proba(x_test)
    y_pred_test = y_proba_test.argmax(axis=1)

    print(y.value_counts())

    print("classification_report of TEST:")
    print(classification_report(y_test, y_pred_test))

    classification_report_val = classification_report(y_test, y_pred_test)
    confusion_matrix_test = confusion_matrix(y_test, y_pred_test)
    preci_value = precision_score(y_test, y_pred_test, average='weighted')
    recall_value = recall_score(y_test, y_pred_test, average='weighted')
    accuracy_value = accuracy_score(y_test, y_pred_test)

    features = pd.DataFrame(dtc.feature_importances_, index=X.columns)
    feature_names = X.columns

    return preci_value, recall_value, accuracy_value, classification_report_val, confusion_matrix_test, dtc, feature_names

#Probabilistic_Decision_Tree_VW_Sample(5)
