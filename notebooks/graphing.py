import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    full = pd.read_csv("../full.csv")

    #---    things to graph
    high_reord = full.groupby('product_name').agg({"reordered":"count"})\
                     .sort_values(by='reordered', ascending = False).reset_index()[:50]
    dow = full['order_dow'].value_counts()
    hours = full['order_hour_of_day'].value_counts().reset_index().sort_values(by="index")
    num_order = full['order_number'].value_counts()
    top_depart = full.groupby('department').agg({"product_name":"size"})\
                     .sort_values(by='product_name', ascending = False).reset_index()
    top_aisle = full.groupby('aisle').agg({"product_name":"size"})\
                     .sort_values(by='product_name', ascending = False).reset_index()[:25]
    bottom_aisle = full.groupby('aisle').agg({"product_name":"size"})\
                     .sort_values(by='product_name', ascending = False).reset_index()[84:]


    #---    df to look at products by hour
    part = full[['product_name','order_hour_of_day','order_number']].copy()
    by_prod = part.groupby(['product_name', 'order_hour_of_day']).agg({'order_number':'size'}).copy().reset_index()
                                                         