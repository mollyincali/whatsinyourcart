import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn import metrics

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
    print("Top features (words) for each cluster:")
    for num, centroid in enumerate(top_centroids):
        print(f"{num}, {', '.join(features[i] for i in centroid)}")

    #---    Fewerer Features
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X1000 = vectorizer.fit_transform(X_train['product_name'])
    features = vectorizer.get_feature_names()
    kmeans = KMeans()
    kmeans.fit(X1000)

    labels = kmeans.labels_
    metrics.silhouette_score(X1000, labels, metric='euclidean')

    top_centroids1000 = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
    print("Top features for each cluster with 1000 max features:")
    for num, centroid in enumerate(top_centroids1000):
        print(f"{num}: {', '.join(features[i] for i in centroid)}")

"""
Results:
Top features for each cluster with 1000 max features:
0: yogurt, greek, total, plain, fat, strawberry, strained, lowfat, vanilla, nonfat
1: banana, original, butter, chocolate, chicken, free, bread, cream, chips, white
2: water, sparkling, spring, mineral, natural, grapefruit, coconut, lime, pure, lemon
3: red, pepper, bell, onion, organic, grapes, seedless, peppers, vine, tomato
4: cheese, cheddar, string, shredded, macaroni, sharp, cream, mozzarella, pizza, organic
5: large, lemon, eggs, grade, brown, organic, aa, extra, cage, free
6: milk, fat, organic, almond, reduced, unsweetened, free, vanilla, coconut, vitamin
7: organic, baby, avocado, strawberries, bananas, bag, spinach, carrots, apple, blueberries
"""








    #---    Create pivot table
    # table = pd.pivot_table(X_train, index = ['user_id'], columns = ['product_id'], aggfunc = np.unique) 