from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

def test_train_split(df2):
    y = df2['action']  # Lables
    X = df2.drop('action', axis=1)
    X = X.to_numpy()  # Array of features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    return X_train, X_test, y_train, y_test


def scaling_values(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def MLP_fit(df2, train_df):
    y_train = train_df['action']  # Labels
    X_train = train_df.drop('action', axis=1)
    X_test = df2.drop(['Close', 'roi'], axis=1)

    # X_train, X_test, y_train, y_test = test_train_split(df2)
    # X_train, X_test = scaling_values(X_train, X_test)

    MLP = MLPClassifier(hidden_layer_sizes= (20, 10, 8, 6, 5), max_iter=2000)
    MLP.fit(X_train, y_train.values.ravel())

    predictions = MLP.predict(X_test)
    df2['predictions'] = predictions

    # print(confusion_matrix(y_test, predictions))
    # print(classification_report(y_test, predictions))

    return df2