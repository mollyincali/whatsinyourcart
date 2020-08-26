import numpy as np
import pandas as pd

if __name__ == '__main__':
    #---    Upload test / train csv
    X_test = pd.read_csv("../X_test.csv")
    X_train = pd.read_csv("../X_train.csv")

    X_test.drop('Unnamed: 0', axis = 1, inplace = True)
    X_train.drop('Unnamed: 0', axis = 1, inplace = True)

    # looking for if order had B
    # products[products['product_name'] == 'Banana'] #24852    
    X_test['banana'] = np.where(X_test['product_id'] == 24852, 1, 0)
    X_test = X_test.groupby('user_id').agg({'order_dow':'max', 'order_hour_of_day':"max", 
                                    'days_since_prior_order':'max', 'add_to_cart_order':'max', 
                                    'banana':'max'}).reset_index()
    y_test = X_test.pop('banana') #1-10744, 0-64256

    X_train['banana'] = np.where(X_train['product_id'] == 24852, 1, 0)
    X_train = X_train.groupby('user_id').agg({'order_dow':'max', 'order_hour_of_day':"max", 
                                    'days_since_prior_order':'max', 'add_to_cart_order':'max', 
                                    'banana':'max'}).reset_index()
    y_train = X_train.pop('banana') #1-18726, 0-112483

    #---    Decision Tree Basic Model
    clf = DecisionTreeClassifier(random_state = 3)
    clf.fit(X_train, y_train)
    clf.predict(X_test)
    dt_score = clf.score(X_test, y_test)
    print(f'Decision Tree Score {dt_score:.5}')