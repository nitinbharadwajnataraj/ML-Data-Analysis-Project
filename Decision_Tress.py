import pandas as pd
from sklearn.preprocessing import LabelEncoder
from main import df_fitting_and_evaluation
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def prepare_DT_df():
    df, style_df = df_fitting_and_evaluation()
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    dtc = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.03)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    print("Confusion Matrics:")
    print(confusion_matrix(y_test, y_pred))
    print("classification_report:")
    print(classification_report(y_test, y_pred))
    features = pd.DataFrame(dtc.feature_importances_, index=X.columns)
    print(features.head(6))


Decision_Tress()



