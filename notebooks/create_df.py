import pandas as pd 
import numpy as np

def get_csv(path):
    df = pd.read_csv(path)
    return df

def test_df(orders_df, prior_df):
    '''
    input: 
       takes in the "orders" df to identifying those specific 75,000 test users
       takes in the "prior" df to connect those test users to their itemized order

    returns: 
        df of 75,000 users and their itemized last order
        Only 2 columns not included: 1 for a duplicate column and 1 the label or train or test
    '''
    order_test = orders[orders['eval_set'] == "test"].copy()
    order_test['prev'] = order_test['order_number'] - 1
    order_test = order_test[['user_id', 'prev']]

    merged_test = order_test.merge(orders, how='inner', left_on=["user_id", "prev"], right_on=["user_id","order_number"])
    merged_test[['user_id', 'prev', 'order_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]

    test = pd.merge(merged_test, prior_df, on="order_id") 
    test[['user_id', 'order_id', 'order_number', 'order_dow',
       'order_hour_of_day', 'days_since_prior_order', 'product_id',
       'add_to_cart_order', 'reordered']]

    test_target = test.pop('reordered')
    
    return test, test_target

def train_df(orders_df, train_df):
    '''
    input: 
        takes in the "train" df to identify my training users
        takes in "orders" df to connect my train users to their most recent itemized orders

    returns: 
        df of 131,209 users and their most recent itemized orders
        Only 2 columns not included: 1 for a duplicate column and 1 the label or train or test
    '''
    train = pd.merge(order_train, orders, how='left')
    train = train[['user_id', 'order_id', 'order_number', 'order_dow',
       'order_hour_of_day', 'days_since_prior_order', 'product_id',
       'add_to_cart_order', 'reordered']]
    train_target = train.pop('reordered')
    
    return train, train_target    

if __name__ == '__main__':
    #create all data frames
    aisle = get_csv('../../instacart_data/aisles.csv')
    depart = get_csv('../../instacart_data/departments.csv')
    products = get_csv('../../instacart_data/products.csv')
    order_train = get_csv('../../instacart_data/order_products__train.csv')
    order_prior = get_csv('../../instacart_data/order_products__prior.csv')
    orders = get_csv('../../instacart_data/orders.csv')

    full_test, test_target = test_df(orders, order_prior)
    full_train, train_target = train_df(orders, order_train)