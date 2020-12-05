'''
code to predict if a user has ordered the most reordered item: bananas
product_id = 24852 banana
product_id = 13176 bag of organic bananas
'''
import numpy as np
import pandas as pd
from graphing import * 

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score, accuracy_score

def setup(order_train, orders, order_prior):
    ''' sets up our data frames '''
    order_train = order_train.merge(orders, on='order_id')
    
    #order_test has no target, so need to pull previous order from that user
    #then get all that information from order_prior
    order_test = orders[orders['eval_set'] == 'test'].copy()
    order_test['order_number'] = order_test['order_number'] - 1
    order_test = order_test[['user_id','order_number']]
    order_test = pd.merge(order_test, orders,  how='inner')
    order_test = pd.merge(order_test, order_prior, how='inner', on= 'order_id')
    return order_train, order_test

def make_target(df):
    ''' create the banana target '''
    df['banana1'] = np.where(df['product_id'] == 24852, 1, 0)
    df['banana2'] = np.where(df['product_id'] == 13176, 1, 0)
    df['banana'] = df['banana2'] + df['banana1']
    df.drop(['banana1', 'banana2'], axis = 1, inplace = True)
    return df

def model_df(df):
    ''' convert df into what will run through model '''
    new_df = df.groupby('order_id').agg({'order_dow':'max', 
                                        'order_hour_of_day':'max', 
                                        'days_since_prior_order':'max', 
                                        'add_to_cart_order':'max', 'banana':'max'}).reset_index()
    target = new_df.pop('banana')
    return new_df, target

def div_count_pos_neg(X, y):
    ''' code needed for oversample '''
    negatives, positives = y == 0, y == 1
    negative_count, positive_count = np.sum(negatives), np.sum(positives)
    X_positives, y_positives = X[positives], y[positives]
    X_negatives, y_negatives = X[negatives], y[negatives]
    return negative_count, positive_count, X_positives, \
           X_negatives, y_positives, y_negatives

def oversample(X, y, tp):
    """
    Randomly choose positive observations from X & y, with replacement
    to achieve the target proportion of positive to negative observations.
    """
    if 0.5 < np.mean(y):
        return X, y
    neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = div_count_pos_neg(X, y)
    positive_range = np.arange(pos_count)
    positive_size = (tp * neg_count) / (1 - tp)
    positive_idxs = np.random.choice(a=positive_range,
                                     size=int(positive_size),
                                     replace=True)
    X_positive_oversampled = X_pos[positive_idxs]
    y_positive_oversampled = y_pos[positive_idxs]
    X_oversampled = np.vstack((X_positive_oversampled, X_neg))
    y_oversampled = np.concatenate((y_positive_oversampled, y_neg))

    return X_oversampled, y_oversampled

def log_regression(num_folds):
    kfold = KFold(n_splits=num_folds)

    acc = []
    f1 = []

    for train_index, test_index in kfold.split(x_o):
        model = LogisticRegression(random_state=3)
        model.fit(x_o[train_index], y_o[train_index])
        y_predict = model.predict(x_o[test_index])
        y_true = y_o[test_index]
        acc.append(accuracy_score(y_true, y_predict))
        f1.append(f1_score(y_true, y_predict))

    print("Accuracy:", np.average(acc))
    print("F1 Score:", np.average(f1))
    return acc, f1

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
    # read in csv
    order_train = pd.read_csv('../../instacart_data/order_products__train.csv')
    orders = pd.read_csv("../../instacart_data/orders.csv")
    order_prior = pd.read_csv("../../instacart_data/order_products__prior.csv")

    # combine df
    order_train, order_test = setup(order_train, orders, order_prior)

    # make banana col
    order_train = make_target(order_train) 
    order_test = make_target(order_test) 

    # model df
    X_train, y_train = model_df(order_train) #26% have a banana
    X_test, y_test = model_df(order_test) #26% have a banana

    # oversample train
    x_o, y_o = oversample(X_train.values, y_train.values, 0.5)

    # logistic regression
    lracc, lrf1 = log_regression(10)

    # decision tree
    dt, dt_score, dt_f1, dt_predict = decision_t(X_test, y_test, x_o, y_o)

    # random forest
    rf, rf_score, rf_f1, rf_predict = random_f(X_test, y_test, x_o, y_o)

    # gradient boosting
    gb, gb_score, gb_f1, gb_predict = gradient_b(X_test, y_test, x_o, y_o)