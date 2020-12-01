'''
will the user reorder a particular item?
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from graphing import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

def setup(order_train, orders, order_prior):
    ''' sets up our data frames '''
    order_train = order_train.merge(orders, on='order_id')
    
    #order_test has no target, so need to pull previous order from that user
    #then get all that information from order_prior
    order_test = orders[orders['eval_set'] == 'test'].copy()
    order_test['order_number'] = order_test['order_number'] - 1
    order_test = order_test[['user_id','order_number']]
    order_test = pd.merge(order_test, orders,  how='inner')
    return order_train, order_test

def new_col(df):
    ''' used to get percent new column '''
    df['reorder_sum'] = df.groupby(['order_id'])['reordered'].transform(lambda x: x.sum())
    df['num_in_cart'] = df.groupby(['order_id'])['add_to_cart_order'].transform(lambda x: x.max())
    df['percent_new'] = 1 - (df['reorder_sum'] / df['num_in_cart'])
    return df

def get_perc_new(df):
    ''' used to get average percent new items from each user '''
    new_df = df.groupby('order_id').agg({'add_to_cart_order':'max','reordered':'sum'}).reset_index()
    new_df['perc_new'] = 1 - (new_df['reordered'] / new_df['add_to_cart_order'])
    new_df = pd.merge(new_df, orders,  how='inner', on = 'order_id')
    new_df = new_df.groupby('user_id').agg({'perc_new':'mean','add_to_cart_order':'mean'})
    new_df = new_df.rename(columns={'perc_new':'user_avg_perc_new', 'add_to_cart_order':'avg_cart_order'}).reset_index() 
    return new_df

def combine_user_product(df):
    ''' creates user_product column '''
    df['user_product'] = (df.apply(lambda row: '%s_%s' %(str(row.user_id),
                                   str(row.product_id)),axis=1))
    return df

def return_newdf(df):
    ''' organizes our df for train and test split '''
    final_df = df.groupby('user_product').agg({'add_to_cart_order':'max', 
                'order_dow':'max', 'order_hour_of_day':'max', 
                'days_since_prior_order':'max', 'user_avg_perc_new':'max',
                'avg_cart_order':'max', 'reordered':'max', 'add_to_cart_order':'max'}).reset_index()
    final_target = final_df.pop('reordered')
    return final_df, final_target

def decision_t(X_test, y_test, X_train, y_train):
    ''' decision tree  code '''
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_predict = dt.predict(X_test)
    dt_score = dt.score(X_test, y_test)
    dt_f1 = f1_score(y_test, dt_predict)
    print(f'Decision Tree Accuracy: {dt_score:.5}')
    print(f'Decision Tree F1 Score: {dt_f1:.5}\n')
    return dt, dt_score, dt_f1

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
    return gb, gb_score, gb_f1

def log_regression(model, num_folds):
    kfold = KFold(n_splits=num_folds)

    acc = []
    f1 = []

    for train_index, test_index in kfold.split(X_train):
        model = LogisticRegression(random_state=3)
        model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        y_predict = model.predict(X_train.iloc[test_index])
        y_true = y_train.iloc[test_index]
        acc.append(accuracy_score(y_true, y_predict))
        f1.append(f1_score(y_true, y_predict))

    print("Accuracy:", np.average(acc))
    print("F1 Score:", np.average(f1))
    return acc, f1

if __name__ == '__main__':
    #read in csv
    order_train = pd.read_csv('../../instacart_data/order_products__train.csv')
    orders = pd.read_csv("../../instacart_data/orders.csv")
    order_prior = pd.read_csv("../../instacart_data/order_products__prior.csv")
    
    # get df
    order_train, order_test = setup(order_train, orders, order_prior)

    # getting avg % new for each user
    user_avg = get_perc_new(order_prior)
    
    # histogram graph
    hist_perc_new(user_avg['user_avg_perc_new'])
    hist_avg_cart(user_avg['avg_cart_order'])

    # merge
    order_train = pd.merge(order_train, user_avg,  how='inner', on = 'user_id')
    order_test = pd.merge(order_test, user_avg,  how='inner', on = 'user_id') 
    order_test = pd.merge(order_test, order_prior, how='inner', on= 'order_id')

    # add user_product col
    train = combine_user_product(order_train)
    test = combine_user_product(order_test)

    # get final dataframe
    X_train, y_train = return_newdf(train)
    X_test, y_test = return_newdf(test)

    # logistic regression
    clf = LogisticRegression(random_state=3)
    lracc, lrf1 = log_regression(clf, 10)

    # decision tree
    dt, dt_score, dt_f1 = decision_t(X_test, y_test, X_train, y_train)

    # random forest
    rf, rf_score, rf_f1, rf_predict = random_f(X_test, y_test, X_train, y_train)

    # gradient boosting
    gb, gb_score, gb_f1 = gradient_b(X_test, y_test, X_train, y_train)

    # graphing
    cm = confusion_matrix(y_test, rf_predict)
    make_heat(cm)