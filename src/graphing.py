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
    high_reord = full.groupby('product_name').agg({"reordered":"count"})\
                     .sort_values(by='reordered', ascending = False).reset_index()[:15]
    dow = full['order_dow'].value_counts().reset_index()
    days = ['Saturday','Sunday','Monday','Tuesday',"Wednesday",'Thursday',"Friday"] 
    dow['days'] = days  
    hours = full['order_hour_of_day'].value_counts().reset_index().sort_values(by="index")
    num_order = full['order_number'].value_counts()

    #---    plots
    make_graph(hours, 'index', 'order_hour_of_day', 'Orders by Hour', "Hours", 'Number of Orders')
    make_graph(dow, 'days', 'order_dow', 'Orders by Day of Week', None, 'Number of Orders')
    make_graph(high_reord, 'reordered', 'product_name', 'Top 15 Highest Reordered Item', 'Number of Reorders', "Item")                

    #---    ice cream
    graph = pd.read_csv('../icecream.csv')
    label = list(graph['product_name'])

    fig, ax = plt.subplots(figsize = (20,10))
    for row in np.arange(20):
        ice = label[row]
        ax.plot(np.arange(24), graph.iloc[row,1:], label = ice)
    ax.set_xlabel('Hours', fontdict=fontaxis)
    ax.set_ylabel('Number of Orders', fontdict=fontaxis)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.title('Time When the Top 20 Ice Creams are Ordered', fontdict=fonttitle)
    plt.legend()
    plt.show();

    #---    who orders 1 thing
    one = full.groupby('order_id').agg({'add_to_cart_order':'max',  
                        'product_name':'max',  
                        'order_hour_of_day':'max'}) 
    one = one[one['add_to_cart_order'] == 1].copy()
    top25 = one['product_name'].value_counts()[:25] 
