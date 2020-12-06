'''
file for graph codes used throughout project
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

def color_theme():
    ''' set up color theme for visuals '''
    color1 = '#F1D78C'
    color2 = '#F6A811'
    color3 = '#F46708'
    color4 = '#EF727F'
    color5 = '#E84846'
    return [color1, color2, color3, color4, color5]
    # return ['#F1D78C','#F6A811','#F46708','#EF727F','#E84846']

def hist_col(col, title, x_title):
    ''' Graph average number of items in cart '''
    fonttitle = {'fontname':'Helvetica', 'fontsize':30}
    plt.hist(col, bins = 100, color = '#F1D78C')
    plt.title(f"{title}", fontdict=fonttitle)
    # plt.axvline(np.mean(col), color ='#F46708', marker = '.', label = f'Mean Cart Size = {np.mean(col)}')
    plt.legend()
    plt.xlabel(f'{x_title}')
    plt.ylabel('Number of Users')
    plt.show();

def score_f1(model, f1, mean_acc, title):
    ''' Graph Score and F1 and Model '''
    fonttitle = {'fontname':'Helvetica', 'fontsize':30}
    fig, ax = plt.subplots(figsize = (20, 10))
    ax.plot(model, f1, color= '#EF727F', marker='*', linewidth = 5, label = 'F1 Score')
    ax.plot(model, mean_acc, color='#F6A811', marker='*', linewidth = 5, label = 'Mean Accuracy Score')     
    # ax.set_ylim(ymin = 0.3, ymax = 0.9)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.legend()
    plt.xticks(rotation = 10)
    plt.title(f'{title} \n Mean Accuracy Score and F1 Score by Model', fontdict=fonttitle)
    plt.show();

def make_bar(df, col_x, col_y, title, x_label, y_label, colors):
    fonttitle = {'fontname':'Helvetica', 'fontsize':30}
    fig, ax = plt.subplots(figsize = (20, 10))
    ax = sns.barplot(df[col_x], df[col_y], palette = colors)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.title(title, fontdict=fonttitle)
    plt.show();

def get_one_item(order_prior, products):
    one = order_prior.groupby('order_id').agg({'add_to_cart_order':'max', 'product_id':'max'}).reset_index()
    one = one[one['add_to_cart_order'] == 1] 
    one = one.groupby('product_id').agg({'add_to_cart_order':'sum'}).reset_index().sort_values(by = 'add_to_cart_order', ascending = False)[:20]
    one = pd.merge(one, products, how= 'inner', on = 'product_id')
    return one

def make_heat(cm):
    ''' make heatmap '''
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm, annot=labels, fmt='', cmap='Reds')
    plt.show();

def setup_depatment(order_prior, products, orders):
    ''' set up department data frame to build department order visual '''
    df = pd.merge(order_prior, products, how = 'left', on = 'product_id')
    df.drop(['add_to_cart_order','reordered', 'aisle_id', 'product_name'], axis = 1, inplace = True)
    df1 = pd.merge(orders, df, how = 'left', on = 'order_id')
    df2 = df1.groupby(['department_id','order_hour_of_day'])['order_id'].count().reset_index()
    df2 = pd.merge(df2, depart, how = 'left', on = 'department_id')
    topdepart = df2.groupby('department').agg({"order_id":'sum'}).sort_values(by='order_id', ascending = False)[:10]
    df2 = df2[df2['department'].isin(list(topdepart.index))]
    df3 = pd.pivot_table(df2, values = 'order_id', index = 'order_hour_of_day', 
                    columns = 'department', aggfunc = 'sum', fill_value = 0, margins = True).reset_index()
    return df3[['order_hour_of_day','produce','dairy eggs',
                'snacks','beverages','frozen','pantry',
                'bakery','canned goods','deli','dry goods pasta']]

def department_orders(df3):
    ''' line plot graph of top 10 department hours '''
    fonttitle = {'fontname':'Helvetica', 'fontsize':30}
    num=0
    line_color = sns.color_palette("hls", 10)
    for column in df3.drop('order_hour_of_day', axis=1):
        num+=1
        plt.plot(df3['order_hour_of_day'], df3[column], marker='', linewidth=2, alpha=0.9, label=column, color = line_color[num - 1])
    plt.title('Number of Orders Per Hour by Department', fontdict=fonttitle)
    plt.xlabel('Hour')
    plt.ylabel('Number of Orders')
    plt.legend(loc=2)
    plt.show();

def top_word_bar(tuple_list, title):
    """ Graphs top words and bigrams
    Args:
        tuple_list ([tuple]): Tuple of top word or bigrams 
        title ([str]): What to title the graph
    """
    words = []
    num = []
    
    for i in tuple_list[::-1]:
        words.append(i[0])
        num.append(i[1])

    fig, ax = plt.subplots(figsize = (10, 5))
    ax.barh(words, np.array(num))
    ax.set_title(f'{title}', fontsize = 30)
    ax.set_yticklabels(words)
    ax.set_ylabel('Number of Occurrences', fontsize = 20)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    #-- SET UP COLOR THEME
    citrus = color_theme()

    #-- BRING IN DATA
    order_prior = pd.read_csv("../../instacart_data/order_products__prior.csv")
    products = pd.read_csv("../../instacart_data/products.csv")
    orders = pd.read_csv('../../instacart_data/orders.csv')
    depart = pd.read_csv('../../instacart_data/departments.csv')

    #-- GETTING ONE ITEM
    one = get_one_item(order_prior, products)
    make_bar(one, 'add_to_cart_order', 'product_name', 'Top Items For Single Orders', 
                'Number of Times Occurs as Only Item Ordered', 'Product Name', citrus)

    #-- DEPARTMENT BY HOUR
    department_df = setup_depatment(order_prior, products, orders)
    department_order()