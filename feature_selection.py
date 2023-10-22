import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import model_selection, metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

feature_importances = []


def get_removal_order(X, y):
    model = ExtraTreesClassifier()
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importances = list(zip(importances, df.columns))
    feature_importances.sort()
    for feature_importance in feature_importances:
        print(feature_importance)
    feature_importances.reverse()
    feature_removal_order = []
    for _ in range(len(feature_importances)):
        feature_removal_order.append(feature_importances.pop()[1])
    return feature_removal_order


if __name__ == "__main__":
    df = pd.read_csv("data_5K.csv")
    df = df.drop(["Patient Id"], axis=1)
    X = df.iloc[:, :-1]
    y = df["Level"]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    removal_order = get_removal_order(X, y)
    count = 1
    for column_name in removal_order[:-5]:
        df = df.drop([column_name], axis=1)

        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns
        )
        X_train_scaled
        X_test_scaled = pd.DataFrame(
            scaler.fit_transform(X_test), columns=X_test.columns
        )
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, prediction)
        print(f"dropped {count} value")
        print(accuracy)
        count += 1
