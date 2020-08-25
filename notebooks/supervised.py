import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

#---    Model Codes
def gridsearch_with_output(estimator, parameter_grid, X_train, y_train):
    '''
        Parameters: estimator: the type of model (e.g. RandomForestRegressor())
                    paramter_grid: dictionary defining the gridsearch parameters
                    X_train: 2d numpy array
                    y_train: 1d numpy array
        Returns:  best parameters and model fit with those parameters
    '''
    model_gridsearch = GridSearchCV(estimator,
                                    parameter_grid,
                                    n_jobs=-1,
                                    verbose=True,
                                    scoring='neg_mean_squared_error')
    model_gridsearch.fit(X_train, y_train)
    best_params = model_gridsearch.best_params_ 
    model_best = model_gridsearch.best_estimator_
    print("\nResult of gridsearch:")
    print("{0:<20s} | {1:<8s} | {2}".format("Parameter", "Optimal", "Gridsearch values"))
    print("-" * 55)
    for param, vals in parameter_grid.items():
        print("{0:<20s} | {1:<8s} | {2}".format(str(param), 
                                                str(best_params[param]),
                                                str(vals)))
    return best_params, model_best

if __name__ == '__main__':
    #---    Upload test / train csv
    X_test = pd.read_csv("../X_test.csv")
    y_test = pd.read_csv("../y_test.csv")
    X_train = pd.read_csv("../X_train.csv")
    y_train = pd.read_csv("../y_train.csv")

    #---    Get dummies on product_id, will need to merge back to OG X_train
    pd.get_dummies(X_train['product_id'])

    #---    Decision Tree Basic Model
    # clf = DecisionTreeClassifier(random_state = 3)
    # clf.fit(X_train, y_train)
    # clf.predict(X_test)
    # dt_score = clf.score(X_test, y_test)
    # print(f'Decision Tree Score {dt_score:.5}')

    #---    Decision Tree GridSearch
    # dt_boosting_grid = {'min_samples_split': [4, 8],
    #                 'max_features': [3, 4],
    #                 'random_state': [3]} 

    # dt_best_params, dt_best_model = gridsearch_with_output(DecisionTreeClassifier(), dt_boosting_grid, X_train, y_train)

    #---    Random Forest
    # rf = RandomForestClassifier(n_estimators=50)
    # rf.fit(X_train, y_train)
    # y_predict = rf.predict(X_test)
    # rf_score = rf.score(X_test, y_test)
    # print(f'Random Forest Score {rf_score:.5}')

    #---    Random Forest GridSearch
    # random_forest_grid = {'max_depth': [4, 8, 12],
    #                   'max_features': [3, 4],
    #                   'min_samples_split': [4, 8],
    #                   'n_estimators': [80, 100],
    #                   'random_state': [3]}

    # rf_best_params, rf_best_model = gridsearch_with_output(RandomForestClassifier(), 
    #                                                    random_forest_grid, 
    #                                                    X_train, y_train)