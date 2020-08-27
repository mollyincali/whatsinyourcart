import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

def make_graph(df, col_x, col_y, title, x_label, y_label):
    fig, ax = plt.subplots(figsize = (20, 10))
    ax = sns.barplot(df[col_x], df[col_y], palette = citrus)
    ax.set_ylabel(y_label, fontdict=fontaxis)
    ax.set_xlabel(x_label, fontdict=fontaxis)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.title(title, fontdict=fonttitle)
    plt.show()

if __name__ == "__main__":
    full = pd.read_csv("../full.csv")

    #---    making graphs special
    color1 = '#F1D78C'
    color2 = '#F6A811'
    color3 = '#F46708'
    color4 = '#EF727F'
    color5 = '#E84846'
    citrus = [color1, color2, color3, color4, color5]
    sns.palplot(sns.color_palette(citrus))

    fonttitle = {'fontname':'Helvetica', 'fontsize':30}
    fontaxis = {'fontname':'Helvetica', 'fontsize':20}

    #---    things to graph
    high_order = full.groupby('product_name').agg({"order_id":"count"})\
                     .sort_values(by='order_id', ascending = False).reset_index()[:15]
    high_reord = full.groupby('product_name').agg({"reordered":"count"})\
                     .sort_values(by='reordered', ascending = False).reset_index()[:15]
    dow = full['order_dow'].value_counts().reset_index()
    hours = full['order_hour_of_day'].value_counts().reset_index().sort_values(by="index")
    num_order = full['order_number'].value_counts()
    top_depart = full.groupby('department').agg({"product_name":"size"})\
                     .sort_values(by='product_name', ascending = False).reset_index()
    top_aisle = full.groupby('aisle').agg({"product_name":"size"})\
                     .sort_values(by='product_name', ascending = False).reset_index()[:20]
    bottom_aisle = full.groupby('aisle').agg({"product_name":"size"})\
                     .sort_values(by='product_name', ascending = False).reset_index()[84:]

    #---    plots
    make_graph(hours, 'index', 'order_hour_of_day', 'Orders by Hour', "Hours", 'Number of Orders')
    make_graph(dow, 'index', 'order_dow', 'Orders by Day of Week', "Day", 'Number of Orders', )
    make_graph(high_reord, 'reordered', 'product_name', 'Top 15 Highest Reordered Item', 'Number of Reorders', "Item")                

    #---    who orders 1 thing
    one = full.groupby('order_id').agg({'add_to_cart_order':'max'})
    one = pd.merge(one, full, on = 'order_id')
    one = one[one['add_to_cart_order_x'] == 1].copy()
    one = one[['order_id', 'add_to_cart_order_x', 
                'product_name', 'order_dow']] 

    grouped = one.groupby('product_name').agg({'order_dow':'mean', 'order_hour_of_day':'mean'})
    counts = one['product_name'].value_counts()
    counts = counts.reset_index()
    counts = counts.rename(columns={'product_name':'count', 'index':'product_name'})
    group = pd.merge(grouped, counts, on = 'product_name')

    top_one = group[group['count'] > 250]
    np.sum(group['count']) #163593

    ax = sns.scatterplot(top_one['order_hour_of_day'], top_one['count'])
    plt.show(); 