import pandas as pd 
import numpy as np

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
        test_df: df of 75,000 users with collinear/duplicate columns removed
        test_target: target column
    '''
    order_test = orders_df[orders_df['eval_set'] == "test"].copy()
    order_test['prev'] = order_test['order_number'] - 1
    order_test = order_test[['user_id', 'prev']]
    merged_test = order_test.merge(orders_df, how='inner', left_on=["user_id", "prev"], right_on=["user_id", "order_number"])
    
    full_test = pd.merge(merged_test, prior_df, on="order_id") 
    test_df = full_test[['user_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'product_id',
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
        train_df: df of 121k+ users with collinear/duplicate columns removed
        train_target: target column
    '''
    full_train = pd.merge(train_df, orders_df, how='left')
    train_df = full_train[['user_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'product_id',
       'add_to_cart_order', 'reordered']]
    train_target = train_df.pop('reordered')
    
    return full_train, train_df, train_target 

def full_df(order_train, order_prior, products, aisle, depart, orders):
    """
    input: order_train, order_prior
    """
    df = pd.concat([order_train, order_prior], axis = 0)
    df = pd.merge(df, products, on='product_id')
    df = pd.merge(df, depart, on='department_id')
    df = pd.merge(df, aisle, on = "aisle_id")
    df = df[['order_id', 'add_to_cart_order', 
        'reordered', 'product_name', 'department', 'aisle']].copy()
    full = pd.merge(df, orders, on ='order_id')
    return full

if __name__ == '__main__':    
    #create all data frames
    order_train = get_csv('../../instacart_data/order_products__train.csv')
    order_prior = get_csv('../../instacart_data/order_products__prior.csv')
    orders = get_csv('../../instacart_data/orders.csv')
    aisle = get_csv('../../instacart_data/aisles.csv')
    depart = get_csv('../../instacart_data/departments.csv')
    products = get_csv('../../instacart_data/products.csv')
    
    #full df for graphing
    full = full_df(order_train, order_prior, products, aisle, depart, orders)
    
    #split into train and test
    full_test, X_test, y_test = test_df(orders, order_prior)
    full_train, X_train, y_train = train_df(orders, order_train)

    #save to csv for faster upload
    full_test.to_csv("../full_test.csv")
    X_test.to_csv("../X_test.csv")
    y_test.to_csv("../y_test.csv")
    full_train.to_csv("../full_train.csv")
    X_train.to_csv("../X_train.csv")
    y_train.to_csv("../y_train.csv")
    full.to_csv('../full.csv')