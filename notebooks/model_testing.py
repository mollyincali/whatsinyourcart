# from helper_functions import *
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#---    Get DataFrame Codes
def get_csv(path):
    df = pd.read_csv(path)
    return df

def test_df(orders_df, prior_df):
    '''
    input: 
       takes in the "orders" df to identifying 75,000 specific test users
       takes in the "prior" df to connect those test users to their itemized order

    returns: 
        full_test: full merged df of 75,000 users
        test_df: df of 75,000 userse with collinear/duplicate columns removed
        test_target: target column
    '''
    order_test = orders_df[orders_df['eval_set'] == "test"].copy()
    order_test['prev'] = order_test['order_number'] - 1
    order_test = order_test[['user_id', 'prev']]
    merged_test = order_test.merge(orders_df, how='inner', left_on=["user_id", "prev"], right_on=["user_id", "order_number"])
    
    full_test = pd.merge(merged_test, prior_df, on="order_id") 
    test_df = full_test[['order_dow', 'order_hour_of_day', 'days_since_prior_order', 'product_id',
       'add_to_cart_order', 'reordered']]
    test_target = test_df.pop('reordered')
    
    return full_test, test_df, test_target

def train_df(orders_df, train_df):
    '''
    input: 
        takes in the "train" df to identify my training users
        takes in "orders" df to connect my train users to their most recent itemized orders

    returns: 
        full_train: full merged df of 121k+ users
        train_df: df of 121k+ userse with collinear/duplicate columns removed
        train_target: target column
    '''
    full_train = pd.merge(train_df, orders_df, how='left')
    train_df = full_train[['order_dow', 'order_hour_of_day', 'days_since_prior_order', 'product_id',
       'add_to_cart_order', 'reordered']]
    train_target = train_df.pop('reordered')
    
    return full_train, train_df, train_target 

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
    #create all data frames
    order_train = get_csv('../../instacart_data/order_products__train.csv')
    order_prior = get_csv('../../instacart_data/order_products__prior.csv')
    orders = get_csv('../../instacart_data/orders.csv')

    #split into train and test
    full_test, X_test, y_test = test_df(orders, order_prior)
    full_train, X_train, y_train = train_df(orders, order_train)

    #---    Decision Tree Basic Model
    # clf = DecisionTreeClassifier(random_state = 3)
    # clf.fit(X_train, y_train)
    # clf.predict(X_test)
    # dt_score = clf.score(X_test, y_test)
    # print(f'Decision Tree Score {dt_score:.5}')

    #---    Decision Tree GridSearch
    # dt_boosting_grid = {'min_samples_split': [5, 10],
    #                 'max_features': ['auto', 'log2', 'sqrt'],
    #                 'random_state': [3]} 

    # dt_best_params, dt_best_model = gridsearch_with_output(DecisionTreeClassifier(), dt_boosting_grid, X_train, y_train)

    #---    'Best' DT score = 0.55963
    # clf = DecisionTreeClassifier(min_samples_split=10, max_features='auto', random_state = 3)
    # clf.fit(X_train, y_train)
    # clf.predict(X_test)
    # dt_score = clf.score(X_test, y_test)
    # print(f'Decision Tree Score {dt_score:.5}')

    #---    Random Forest
    # rf = RandomForestClassifier(n_estimators=50)
    # rf.fit(X_train, y_train)
    # y_predict = rf.predict(X_test)
    # rf_score = rf.score(X_test, y_test)
    # print(f'Random Forest Score {rf_score:.5}')
    
    #DO THIS! 
    # .confusion_matrix(y_true, y_pred)

    #---    Random Forest GridSearch
    random_forest_grid = {'max_depth': [3, 5, 7, None],
                      'max_features': ['sqrt', 'log2', None],
                      'min_samples_split': [2, 4, 8],
                      'n_estimators': [20, 40, 80],
                      'random_state': [3]}

    rf_best_params, rf_best_model = gridsearch_with_output(RandomForestClassifier(), 
                                                       random_forest_grid, 
                                                       X_train, y_train)