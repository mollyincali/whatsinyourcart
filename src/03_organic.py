''' 
code to predict if a user has ordered an organic items
'''
import numpy as np
import pandas as pd

from graphing import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score

def decision_t(X_test, y_test, X_train, y_train):
    ''' decision tree  code '''
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_predict = dt.predict(X_test)
    dt_score = dt.score(X_test, y_test)
    dt_f1 = f1_score(y_test, dt_predict)
    print(f'Decision Tree Accuracy: {dt_score:.5}')
    print(f'Decision Tree F1 Score: {dt_f1:.5}\n')
    return dt, dt_score, dt_f1, dt_predict

def random_f(X_test, y_test, X_train, y_train):
    ''' random forest code '''
    rf = RandomForestClassifier(n_estimators=100, max_depth=10)
    rf.fit(X_train, np.ravel(y_train))
    rf_predict = rf.predict(X_test)
    rf_score = rf.score(X_test, y_test)
    rf_f1 = f1_score(y_test, rf_predict)
    print(f'Random Forest Score: {rf_score:.5}')
    print(f'Random Forest F1 Score: {rf_f1:.5}\n')
    return rf, rf_score, rf_f1, rf_predict

def gradient_b(X_test, y_test, X_train, y_train):
    ''' gradient boost code '''
    gb = GradientBoostingClassifier(learning_rate = 0.01, n_estimators=100, max_depth = 5)
    gb.fit(X_train, np.ravel(y_train))
    gb_predict = gb.predict(X_test)
    gb_score = gb.score(X_test, y_test)
    gb_f1 = f1_score(y_test, gb_predict)
    print(f'Gradient Boost Mean Accuracy: {gb_score:.5}')
    print(f'Gradient Boost F1 Score: {gb_f1:.5}')
    return gb, gb_score, gb_f1, gb_predict

if __name__ == '__main__':
    #---    Upload csv
    X_train = pd.read_csv("../old/organicxtrain.csv").drop('Unnamed: 0', axis = 1)
    X_test = pd.read_csv("../old/organicxtest.csv").drop('Unnamed: 0', axis = 1) 
    y_train = pd.read_csv("../old/organicytrain.csv").drop('Unnamed: 0', axis = 1) 
    y_test = pd.read_csv("../old/organicytest.csv").drop('Unnamed: 0', axis = 1) 

    # decision tree
    dt, dt_score, dt_f1, dt_predict = decision_t(X_test, y_test, X_train, y_train)

    # random forest
    rf, rf_score, rf_f1, rf_predict = random_f(X_test, y_test, X_train, y_train)

    # gradient boosting
    gb, gb_score, gb_f1, gb_predict = gradient_b(X_test, y_test, X_train, y_train)


