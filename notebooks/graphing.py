import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

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

    #checks to make sure below code is correct! YAY
    #part[(part['order_hour_of_day'] == 10) & (part['product_name'] == 'Bulgarian Yogurt')]

    #---    df to look at products by hour
    part = full[['product_name','order_hour_of_day','order_number']].copy()
    by_hour = part.groupby(['order_hour_of_day', 'product_name']).agg({'order_number':'size'}).reset_index()  
    order_num_array = np.array(by_hour.groupby('order_hour_of_day').agg({'order_number':'sum'}))

    # didn't work
    # total_num_orders = by_hour['order_number'].sum() 
    # by_hour['percent'] = (by_hour['order_number'] / total_num_orders).round(1)             

    #graphs by hour
    graph20 = by_hour[by_hour['order_hour_of_day'] == 20].sort_values(by='order_number', ascending = False)[:20].reset_index()
    graph10 = by_hour[by_hour['order_hour_of_day'] == 10].sort_values(by='order_number', ascending = False)[:20].reset_index()
    

    # #--- graphing
    # fig, ax = plt.subplots(figsize = (20, 10))
    # ax.plot(graph20['product_name'], graph20['percent'])
    # plt.show();                                                         