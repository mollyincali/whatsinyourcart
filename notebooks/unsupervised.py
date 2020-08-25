import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    #---    Upload test / train csv
    X_test = pd.read_csv("../X_test.csv").drop('Unnamed: 0', axis = 1, inplace=True)
    y_test = pd.read_csv("../y_test.csv").drop('Unnamed: 0', axis = 1, inplace=True)
    X_train = pd.read_csv("../X_train.csv").drop('Unnamed: 0', axis = 1, inplace=True)
    y_train = pd.read_csv("../y_train.csv").drop('Unnamed: 0', axis = 1, inplace=True)

    #---    Upload product to then merge with test
    products = get_csv('../../instacart_data/products.csv')
    X_train = pd.merge(X_train, products, on = 'product_id')

    #---    Tfidf 
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_train['product_name'])
    features = vectorizer.get_feature_names()
    #maybe stem these words?

    #---    kMean
    kmeans = KMeans() 
    kmeans.fit(X)

    #---    Printing
    top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
    print("\n3) top features (words) for each cluster:")
    for num, centroid in enumerate(top_centroids):
        print(f"{num}, {', '.join(features[i] for i in centroid)}")

    #---    Create pivot table
    # table = pd.pivot_table(X_train, index = ['user_id'], columns = ['product_id'], aggfunc = np.unique) 