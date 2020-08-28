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

    #product 
    cool = full.groupby(['order_hour_of_day', 'aisle']).agg({'order_id':'count'})  

    cool20 = cool[cool['order_hour_of_day'] == 20].copy()
    cool10 = cool[cool['order_hour_of_day'] == 15].copy()
    cool15 = cool[cool['order_hour_of_day'] == 10].copy()

    cool20['perc'] = cool20['order_id'] / (np.sum(cool20['order_id']))
    cool10['perc'] = cool10['order_id'] / (np.sum(cool10['order_id']))
    cool15['perc'] = cool15['order_id'] / (np.sum(cool15['order_id']))

    cool10top = cool10.sort_values(by='perc', ascending = False)[:10]
    cool15top = cool15.sort_values(by='perc', ascending = False)[:10]
    cool20top = cool20.sort_values(by='perc', ascending = False)[:10]

    fig, ax = plt.subplots(figsize = (10, 5))
    plt.scatter(cool10top['aisle'], cool10top['perc'] * 100, alpha=0.5, color= '#F6A811', label = '10am', marker="*")
    plt.scatter(cool15top['aisle'], cool15top['perc'] * 100, alpha=0.5, color= '#F46708', label = '3pm')
    plt.scatter(cool20top['aisle'], cool20top['perc'] * 100, alpha=0.5, color= '#EF727F', label = '8pm')     
    plt.legend()
    plt.show();

    #---    ice cream
    ice = full[full['aisle'] == 'ice cream ice'] 
    top20 = ice['product_name'].value_counts()[:20]
    top20list = list(top20.index)
    top = ice[ice.product_name.isin(top20label)]

    graph = pd.read_csv('../ice.csv')
    label = list(graph['product_name'])

    fig, ax = plt.subplots(figsize = (20,10))
    for row in np.arange(20):
        ice = label[row]
        ax.plot(np.arange(24), graph.iloc[row,1:], label = ice)
    ax.set_ylabel('Hours', fontdict=fontaxis)
    ax.set_xlabel('Number of Orders', fontdict=fontaxis)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.title('Time When the Top 20 Ice Creams are Ordered', fontdict=fonttitle)
    plt.legend()
    plt.show();

    colors = sns.palplot(sns.color_palette("husl", 20))
    plt.show()

    fig, ax = plt.subplots(figsize = (20,10))
    for row in np.arange(20):
        ice = label[row]
        ax = sns.lineplot(x = np.arange(24), y = graph.iloc[row,1:], label = ice)
    plt.legend()
    plt.show();