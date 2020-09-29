'''
code to predict if a user has ordered the most reordered item: bananas
product_id = 24852 banana
product_id = 13176 bag of organic bananas
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score

def setup(order_train, orders, order_prior):
    ''' set up desired df '''
    order_train = order_train.merge(orders, on='order_id')
    order_test = orders[orders['eval_set'] == 'test'].copy()
    #order_test has no target, so need to pull previous order from that user
    #then get all that information from order_prior
    order_test['order_number'] = order_test['order_number'] - 1
    new_df = pd.merge(order_test, orders,  how='inner', 
                        left_on=['user_id','order_number'], 
                        right_on = ['user_id','order_number'])
    order_test = new_df[['user_id' , 'order_number', 'order_id_y', 
                        'eval_set_y','order_dow_y','order_hour_of_day_y',
                        'days_since_prior_order_y']].copy()
    order_test = order_test.rename(columns={'order_id_y':'order_id', 'order_dow_y':'order_dow',
                            'order_hour_of_day_y':'order_hour_of_day',
                            'days_since_prior_order_y':'days_since_prior_order'}) 
    order_test = order_test.merge(order_prior, how='inner', on='order_id')
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

def randomforest(X_test, y_test, X_train, y_train, trees):
    ''' fit, predict, and score randomforest model'''
    rf = RandomForestClassifier(n_estimators=trees)
    rf.fit(X_train, np.ravel(y_train))
    y_predict = rf.predict(X_test)
    rf_score = rf.score(X_test, y_test)
    f1 = f1_score(y_test, y_predict)
    print(f'Random Forest Score: {rf_score:.5}')
    print(f'Random Forest F1 Score: {f1:.5}\n')
    return rf, y_predict, rf_score, f1

def make_heat():
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    plt.show();

if __name__ == '__main__':
    #read in csv
    order_train = pd.read_csv('../../instacart_data/order_products__train.csv')
    orders = pd.read_csv("../../instacart_data/orders.csv")
    order_prior = pd.read_csv("../../instacart_data/order_products__prior.csv")

    #combine df
    order_train, order_test = setup(order_train, orders, order_prior)

    #make banana col
    order_train = make_target(order_train) 
    order_test = make_target(order_test) 

    #model df
    X_train, y_train = model_df(order_train) #26% have a banana
    X_test, y_test = model_df(order_test) #26% have a banana

    #oversample train
    x_o, y_o = oversample(X_train.values, y_train.values, 0.5)

    #basic random forest
    rf, y_predict, rf_score, f1 = randomforest(X_test, y_test, x_o, y_o, 50)

    #matrix
    cm = confusion_matrix(y_test, y_predict)

    #heatmap
    make_heat()

    #decision tree
    clf = DecisionTreeClassifier()
    clf.fit(x_o, y_o)
    y_predict = clf.predict(X_test)
    dt_score = clf.score(X_test, y_test)
    print(f'Decision Tree Score Best Param: {dt_score:.5}')
    print(f'Decision Tree F1 Score: {(f1_score(y_test, y_predict)):.5}\n')
    # Decision Tree Score Best Param: 0.66099
    # Decision Tree F1 Score: 0.33845

    #randomforest2
    rf = RandomForestClassifier(max_depth=4, max_features=3, min_samples_split=4, n_estimators=80)
    rf.fit(x_o, y_o)
    y_predict = rf.predict(X_test)
    rf_score = rf.score(X_test, y_test)
    print(f'Random Forest Score Best Param: {rf_score:.5}')
    print(f'Random Forest F1 Score Best Param: {(f1_score(y_test, y_predict)):.5}\n')
    # Random Forest Score Best Param: 0.62752
    # Random Forest F1 Score Best Param: 0.50198

    #gb
    gb = GradientBoostingClassifier(learning_rate = 0.1, n_estimators=100, max_depth = 5)
    gb.fit(x_o, y_o)
    y_predict = gb.predict(X_test)
    gb_score = gb.score(X_test, y_test)
    print(f'Gradient Boost Mean Accuracy: {gb_score:.5}')
    print(f'Gradient Boost F1 Score: {(f1_score(y_test, y_predict)):.5}')
    # Gradient Boost Mean Accuracy: 0.64355
    # Gradient Boost F1 Score: 0.50127