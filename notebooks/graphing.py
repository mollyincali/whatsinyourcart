import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

if __name__ == "__main__":
    full = pd.read_csv("../full.csv")

    color1 = '#F1D78C'
    color2 = '#F6A811'
    color3 = '#F46708'
    color4 = '#EF727F'
    color5 = '#E84846'
    citrus = [color1, color2, color3, color4, color5]
    sns.palplot(sns.color_palette(citrus))

    fonttitle = {'fontname':'Helvetica', 'fontsize':25}
    fontaxis = {'fontname':'Helvetica', 'fontsize':18}

    #---    things to graph
    high_reord = full.groupby('product_name').agg({"reordered":"count"})\
                     .sort_values(by='reordered', ascending = False).reset_index()[:50]
    dow = full['order_dow'].value_counts().reset_index()
    hours = full['order_hour_of_day'].value_counts().reset_index().sort_values(by="index")
    num_order = full['order_number'].value_counts()
    top_depart = full.groupby('department').agg({"product_name":"size"})\
                     .sort_values(by='product_name', ascending = False).reset_index()
    top_aisle = full.groupby('aisle').agg({"product_name":"size"})\
                     .sort_values(by='product_name', ascending = False).reset_index()[:25]
    bottom_aisle = full.groupby('aisle').agg({"product_name":"size"})\
                     .sort_values(by='product_name', ascending = False).reset_index()[84:]

    #---    plots
    fig, ax = plt.subplots(figsize = (20, 10))
    ax = sns.barplot(hours['index'], hours['order_hour_of_day'], palette = citrus)
    ax.set_ylabel('Number of Orders', fontdict=fontaxis)
    ax.set_xlabel("Hours", fontdict=fontaxis)
    plt.title('Orders by Hour', fontdict=fonttitle)
    plt.show()
    plt.savefig('hour.png');

    fig, ax = plt.subplots(figsize = (20, 10))
    ax = sns.barplot(dow['index'], dow['order_dow'], palette = citrus)
    ax.set_ylabel('Number of Orders', fontdict=fontaxis)
    ax.set_xlabel("Day", fontdict=fontaxis)
    plt.title('Orders by Day of Week', fontdict=fonttitle)
    plt.show()
    plt.savefig('dayofweek');

    fig, ax = plt.subplots(figsize = (20, 10))
    ax = sns.barplot(high_reord['product_name'], high_reord['reordered'], palette = citrus)
    plt.xticks(rotation=90)
    ax.set_ylabel('Number of Reorders', fontdict=fontaxis)
    ax.set_xlabel("Item", fontdict=fontaxis)
    plt.title('Highest Reordered Item', fontdict=fonttitle)
    plt.show()
    plt.savefig('highestreorder');

    #---    df to look at products by hour
    # part = full[['product_name','order_hour_of_day','order_number']].copy()
    # by_prod = part.groupby(['product_name', 'order_hour_of_day']).agg({'order_number':'size'}).copy().reset_index()
                                                         