import pandas as pd 
import numpy as np

def get_csv(path):
    df = pd.read_csv(path)
    return df

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

if __name__ == "__main__":
    aisle = get_csv('../../instacart_data/aisles.csv')
    depart = get_csv('../../instacart_data/departments.csv')
    products = get_csv('../../instacart_data/products.csv')
    order_train = get_csv('../../instacart_data/order_products__train.csv')
    order_prior = get_csv('../../instacart_data/order_products__prior.csv')
    orders = get_csv('../../instacart_data/orders.csv')

    full = full_df(order_train, order_prior, products, aisle, depart, orders)