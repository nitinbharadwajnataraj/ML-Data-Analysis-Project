import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

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

    return df
from sklearn import tree
def prepare_DT_df():
    df= df_fitting_and_evaluation()
    df.drop(columns=['ID', 'Evaluation'])
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform the 'fitting_group' column
    df['fitting_group_encoded'] = label_encoder.fit_transform(df['fitting_group'])

    # Mapping of original values to encoded values
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label mapping:", label_mapping)

    # Replace 'fitting_group' with 'fitting_group_encoded'
    df.drop(columns=['fitting_group'], inplace=True)
    df.rename(columns={'fitting_group_encoded': 'fitting_group'}, inplace=True)
    df.drop(columns=['Evaluation','ID','fitting_distance'], inplace=True)

    return df


def Decision_Tress():
    df = prepare_DT_df()
    X = df.iloc[:, 0:4]
    y = df.iloc[:, 4]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)
    x_main,x_test,y_main,y_test=train_test_split(X,y, test_size=0.2, random_state=17,stratify=y)
    x_train, x_val,y_train,y_val=train_test_split(x_main,y_main, test_size=0.2, random_state=17, stratify=y_main)
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    dtc = DecisionTreeClassifier(criterion='entropy', max_depth=6)

    #USEless thing
    # l = [0.0,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    # crit = ['gini', 'entropy']
    # dict_ijk = {}
    # for i in range(1,8):
    #     for j in l:
    #         for k in crit:
    #             dtc = DecisionTreeClassifier(criterion='entropy', max_depth=6)
    #             dtc.fit(x_main, y_main)
    #             y_pred_val = dtc.predict(x_val)
    #             y_pred_test = dtc.predict(x_test)
    #             acc_val = accuracy_score(y_val, y_pred_val)
    #             acc_test = accuracy_score(y_test, y_pred_test)
    #
    #             if acc_test > 0.85:
    #                 dict_ijk[(i, j, k)] = acc_test
    #                 # print(f"depth={i}, criterion={k}, ccp_alpha={j}")
    #                 # print("accuracy_report for Validation:")
    #                 # print(acc_val)
    #                 # print("accuracy_report of TEST:")
    #                 # print(acc_test)
    #
    # max_value = max(dict_ijk.values())
    # key_with_max_value = [key for key, value in dict_ijk.items() if value == max_value][0]
    # print(key_with_max_value, max_value)

    dtc.fit(x_main, y_main)
    #This is Validation
    y_pred_val = dtc.predict(x_val)
    print("Confusion Matrics for Validation:")
    print(confusion_matrix(y_val, y_pred_val))
    print("classification_report for Validation:")
    print(classification_report(y_val, y_pred_val))


    #this is for Testing
    y_pred_test = dtc.predict(x_test)
    print("Confusion Matrics for TEST:")
    print(confusion_matrix(y_test, y_pred_test))
    print("classification_report of TEST:")
    print(classification_report(y_test, y_pred_test))

    features = pd.DataFrame(dtc.feature_importances_, index=X.columns)
    print(features.head(6))

    # Visualization of the Decision Tree
    #text_representation = tree.export_text(dtc)
    #print(text_representation)
    # with open("decistion_tree.log", "w") as fout:
    #     fout.write(text_representation)

    # fig = plt.figure(figsize=(25, 20))
    # _ = tree.plot_tree(dtc, feature_names=X.columns, filled=True)
    #plt.show()

Decision_Tress()



