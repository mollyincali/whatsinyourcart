'''
find common ordered pairs of items
'''
import numpy as np
import pandas as pd
import gensim

def setup_product_list(df):
    ''' takes in a dataframe groups by order_id and creates list of product names'''
    new_df = pd.merge(df, products, how = 'inner', on = 'product_id')
    new_df['product'] = new_df['product_name'].str.replace(' ', '_')
    order_product = new_df.groupby('order_id').apply(lambda x: x['product'].tolist())
    return order_product

if __name__ == '__main__':
    #read in csv
    order_train = pd.read_csv('../../instacart_data/order_products__train.csv')
    order_prior = pd.read_csv("../../instacart_data/order_products__prior.csv")
    products = pd.read_csv('../../instacart_data/products.csv') 

    train_product = setup_product_list(order_train)
    prior_product = setup_product_list(order_prior)

    products['name'] = products['product_name'].str.replace(" ", "_")
    products = products['name']

    #combine and find longest
    sentence = pd.concat([prior_product, train_product])
    longest = np.max(sentence.apply(len))
    
    #build model
    model = gensim.models.Word2Vec(sentence.values, size=100, window=longest, min_count=2, workers=4)


    #print out random product and top 5 recommendations
    sample_product = products[np.random.randint(49689)]
    print(f'Top 5 Recommendations for {sample_product}: ')
    for i in model.wv.most_similar(sample_product)[0:5]:
        print(i)