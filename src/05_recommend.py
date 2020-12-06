"""
get top purchased pairs of items
"""
import pandas as pd 
import numpy as np
from graphing import *

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import string 

def remove_punctuation(text): 
    ''' function to remove punctuation '''
    punct_to_remove = string.punctuation 
    return text.translate(str.maketrans('', '', punct_to_remove)) 

def clean_product_text(prod):
    ''' function to clean product name '''
    prod['lower'] = prod['product_name'].str.lower()
    prod['clean'] = prod['lower'].apply(lambda text: remove_punctuation(text))
    prod['clean2'] = 0

    for row in range(prod.shape[0]):
        prod.iloc[row, 6] = "".join([i for i in prod.iloc[row, 5] if not i.isdigit()]) 
    
    prod['name'] = prod['clean2'].str.replace(" ", "_")
    return prod 

def setup_product_list(df):
    ''' takes in a dataframe groups by order_id and creates list of product for that order'''
    new_df = pd.merge(df, prod, how = 'inner', on = 'product_id')
    order_product = new_df.groupby('order_id').apply(lambda x: ", ".join(x['name'])).reset_index()
    return order_product

def get_top_words(corpus, n, ngram = (1, 1)):
    ''' Use CountVectorizer to get top n words from tweet list
    Args:
        corpus [list]: List of tweets
        stop_words [list]: List of stop words to remove
        n [int]: Number of words to return 
    Returns:
        [list]: List of top n frequently used words
    '''
    stop_words = stopwords.words('english')
    vec = CountVectorizer(lowercase = True, strip_accents = 'ascii', stop_words = stop_words, ngram_range=ngram)
    bow = vec.fit_transform(corpus)
    sum_words = bow.sum(axis = 0)
    word_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    word_freq = sorted(word_freq, key = lambda x: x[1], reverse = True)
    print(f"Top {n} Words: {word_freq[:n]}")
    return word_freq

if __name__ == '__main__':
    #-- SET UP
    order_train = pd.read_csv('../../instacart_data/order_products__train.csv')
    order_prior = pd.read_csv("../../instacart_data/order_products__prior.csv")
    product = pd.read_csv('../../instacart_data/products.csv')
    
    #-- CLEAN UP PRODUCT NAME
    prod = clean_product_text(product)

    #-- CREATE NEW DATAFRAME FOR
    train = setup_product_list(order_train)
    prior = setup_product_list(order_prior)
    full = pd.concat([prior, train], axis = 0) 
    word_freq = get_top_words(full[0], 10, (2,2))