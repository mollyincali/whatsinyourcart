'''
file of EDA and graphing
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

def hist_perc_new(col):
    ''' Graph % New Items '''
    plt.hist(col, bins = 60, color = '#F1D78C')
    plt.title("Each Users Average Percentage of NEW Items")
    plt.xlabel('Percentage of NEW Items')
    plt.ylabel('Number of Users')
    plt.show();

def hist_avg_cart(col):
    ''' Graph average number of items in cart '''
    plt.hist(col, bins = 100, color = '#F46708')
    plt.title("Each Users Average Cart Order")
    plt.xlabel('Average Cart Order')
    plt.ylabel('Number of Users')
    plt.show();

def score_f1(model, f1, mean_acc, title):
    ''' Graph Score and F1 and Model '''
    fonttitle = {'fontname':'Helvetica', 'fontsize':30}
    fig, ax = plt.subplots(figsize = (20, 10))
    ax.plot(model, f1, color= '#EF727F', marker='*', linewidth = 5, label = 'F1 Score')
    ax.plot(model, mean_acc, color='#F6A811', marker='*', linewidth = 5, label = 'Mean Accuracy Score')     
    ax.set_ylim(ymin = 0.3, ymax = 0.9)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.legend()
    plt.xticks(rotation = 10)
    plt.title(f'{title} \n Mean Accuracy Score and F1 Score by Model', fontdict=fonttitle)
    plt.show();

def make_bar(df, col_x, col_y, title, x_label, y_label):
    fonttitle = {'fontname':'Helvetica', 'fontsize':30}
    fig, ax = plt.subplots(figsize = (20, 10))
    ax = sns.barplot(df[col_x], df[col_y], palette = citrus)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    # ax.tick_params(axis='both', which='major', labelsize=18)
    plt.title(title, fontdict=fonttitle)
    plt.show();

def get_one_item():
    order_train = pd.read_csv('../../instacart_data/order_products__train.csv')
    order_prior = pd.read_csv("../../instacart_data/order_products__prior.csv")
    products = pd.read_csv("../../instacart_data/products.csv")
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

if __name__ == "__main__":
    color1 = '#F1D78C'
    color2 = '#F6A811'
    color3 = '#F46708'
    color4 = '#EF727F'
    color5 = '#E84846'
    citrus = [color1, color2, color3, color4, color5]

    one = get_one_item()
    make_bar(one, 'add_to_cart_order', 'product_name', 'Top Items For Single Orders', 
                'Number of Times Occurs as Only Item Ordered', 'Product Name')