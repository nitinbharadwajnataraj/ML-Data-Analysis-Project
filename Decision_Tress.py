import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import joblib


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

    df= df_fitting_and_evaluation()

    print("Original DataFrame:")
    print(df.head())  # Check the original DataFrame

    # Drop unnecessary columns

    print(df.columns)
    df = df.drop(columns=['ID', 'fitting_distance'])
    print(df)

    df = df.drop(columns=['ID', 'Evaluation', 'fitting_distance'])


    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    #
    # # Fit and transform the 'fitting_group' column
    df['evaluation_encoded'] = label_encoder.fit_transform(df['Evaluation'])
    #
    # # Mapping of original values to encoded values
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label mapping:", label_mapping)

    # Drop the original 'fitting_group' column
    df = df.drop(columns=['Evaluation'])

    print("DataFrame after preprocessing:")
    print(df.head())  # Check the DataFrame after preprocessing

    df = df.drop(columns=['fitting_group'])

    print("DataFrame after preprocessing:")
    print(df.head())  # Check the DataFrame after preprocessing

    return df

def visualize_decision_tree(dtc, feature_names):
    plt.figure(figsize=(25, 20))
    tree.plot_tree(dtc, feature_names=feature_names, class_names=['OK', 'NOK'], filled=True, rounded=True,
                   fontsize=12)
    plt.title("Decision Tree Visualization", fontsize=20)
    plt.xlabel("Features", fontsize=16)
    plt.ylabel("Class", fontsize=16)
    plt.savefig("decision_tree.png")  # Save the plot as an image
    plt.show()


def Decision_Tress():
    df = prepare_DT_df()


    X = df.iloc[:, 0:7]
    y = df.iloc[:, 7]
    print(X, y)
    x_main, x_test, y_main, y_test = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_main, y_main, test_size=0.2, random_state=17, stratify=y_main)

    dtc = DecisionTreeClassifier(criterion='entropy', max_depth=6)
    dtc.fit(x_main, y_main)

    # Save the model
    joblib.dump(dtc, 'decision_tree_model.joblib')
    # Convert class names to strings
    # class_names = df['evaluation_encoded'].unique().astype(str)


    X = df.iloc[:, 0:4]
    y = df.iloc[:, 4]
    x_main,x_test,y_main,y_test=train_test_split(X,y, test_size=0.2, random_state=17,stratify=y)
    x_train, x_val,y_train,y_val=train_test_split(x_main,y_main, test_size=0.2, random_state=17, stratify=y_main)

    dtc = DecisionTreeClassifier(criterion='entropy', max_depth=6)
    dtc.fit(x_main, y_main)

    # Convert class names to strings
    class_names = df['fitting_group_encoded'].unique().astype(str)


    #This is Validation

    y_pred_val = dtc.predict(x_val)
    print("Confusion Matrics for Validation:")
    print(confusion_matrix(y_val, y_pred_val))
    print("classification_report for Validation:")
    print(classification_report(y_val, y_pred_val))


    y_pred_test = dtc.predict(x_test)
    print("Confusion Matrics for TEST:")
    print(confusion_matrix(y_test, y_pred_test))
    print("classification_report of TEST:")
    print(classification_report(y_test, y_pred_test))
    classification_report_val = classification_report(y_test, y_pred_test)
    confusion_matrix_test = confusion_matrix(y_test, y_pred_test)
    preci_value = precision_score(y_test, y_pred_test, average='weighted')
    recall_value = recall_score(y_test, y_pred_test, average='weighted')
    preci_value = precision_score(y_test, y_pred_test,  average='weighted')
    recall_value = recall_score(y_test, y_pred_test,  average='weighted')
    accuracy_value = accuracy_score(y_test, y_pred_test)

    features = pd.DataFrame(dtc.feature_importances_, index=X.columns)
    print(features.head(6))
    visualize_decision_tree(dtc, X.columns)
    # Function to visualize Decision Tree

    return preci_value, recall_value, accuracy_value, classification_report_val, confusion_matrix_test


Decision_Tress()
    # Visualization of the Decision Tree
    #text_representation = tree.export_text(dtc)
    #print(text_representation)
    # with open("decistion_tree.log", "w") as fout:
    #     fout.write(text_representation)

    # fig = plt.figure(figsize=(25, 20))
    # _ = tree.plot_tree(dtc, feature_names=X.columns, filled=True)
    #plt.show()



    return preci_value,recall_value, accuracy_value, classification_report_val, confusion_matrix_test
Decision_Tress()
