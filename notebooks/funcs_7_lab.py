import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import optuna
from functools import partial
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
import os
from joblib import dump, load
from math import sqrt
import warnings

warnings.filterwarnings("ignore")


def class_kfold_metrics(model, X, y, fitted=False):
    oversampler = SMOTE()
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    accuracy = []
    precision = []
    recall = []
    f1 = []

    for train_index, test_index in kf.split(X_resampled):
        X_train, X_test = X_resampled[train_index], X_resampled[test_index]
        y_train, y_test = y_resampled[train_index], y_resampled[test_index]

        if not fitted:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred, average='macro'))
        recall.append(recall_score(y_test, y_pred, average='macro'))
        f1.append(f1_score(y_test, y_pred, average='macro'))

    accuracy = "{:.2f}".format(np.mean(accuracy))
    precision = "{:.2f}".format(np.mean(precision))
    recall = "{:.2f}".format(np.mean(recall))
    f1 = "{:.2f}".format(np.mean(f1))

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')

    return [f1, accuracy, precision, recall], model

def reg_kfold_metrics(model, X, y, fitted=False):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    mae = []
    mse = []
    rmse = []
    mape = []
    r2 = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if not fitted:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae.append(mean_absolute_error(y_test, y_pred))
        mse.append(mean_squared_error(y_test, y_pred))
        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        mape.append(sqrt(mean_absolute_percentage_error(y_test, y_pred)))
        r2.append(model.score(X_test, y_test))

    mae = "{:.4f}".format(np.mean(mae))
    mse = "{:.2e}".format(np.mean(mse))
    rmse = "{:.4f}".format(np.mean(rmse))
    mape = "{:.4f}".format(np.mean(mape))
    r2 = "{:.4f}".format(np.mean(r2))

    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape}')
    print(f'R^2: {r2}\n')

    return [mae, mse, rmse, mape, r2], model

