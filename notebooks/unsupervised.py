import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    #---    Upload test / train csv
    X_test = pd.read_csv("../X_test.csv")
    y_test = pd.read_csv("../y_test.csv")
    X_train = pd.read_csv("../X_train.csv")
    y_train = pd.read_csv("../y_train.csv")

    X_test.drop('Unnamed: 0', axis = 1, inplace = True)
    y_test.drop('Unnamed: 0', axis = 1, inplace = True)
    X_train.drop('Unnamed: 0', axis = 1, inplace = True)
    y_train.drop('Unnamed: 0', axis = 1, inplace = True)

    #---    merge aisle and products
    aisle = pd.read_csv('../../instacart_data/aisles.csv')
    products = pd.read_csv('../../instacart_data/products.csv')
    p_a = pd.merge(aisle, products, on = 'aisle_id')
    p_a.drop('department_id', axis = 1, inplace = True)
    full = pd.merge(X_train, p_a, on = 'product_id')    

    #---    PCA Work
    pca_work = pd.crosstab(full['user_id'], full['aisle'])  #shape (131,209 by 134)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    pca_scaled = scaler.fit_transform(pca_work) 

    pca = PCA(n_components=6)
    pca.fit(pca_scaled)
    X_pca = pca.transform(pca_scaled) #from 134 features to 6

    print("\nData after PCA into 6 components")
    print("PC1\tPC2\tPC3\tPC4\tPC5\tPC6")
    for i in range(6):
        print(f'{X_pca[i,0]:0.1f}, \t{X_pca[i,1]:0.1f}, \t{X_pca[i,2]:0.1f}, \t{X_pca[i,3]:0.1f}, \t{X_pca[i,4]:0.1f}, \t{X_pca[i,5]:0.1f}')

    #ugly graph
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(X_pca[:, 0], X_pca[:, 1],
           cmap=plt.cm.Set1, edgecolor='k', s=40)
    plt.show();

    #explain variance only 10%
    np.sum(pca.explained_variance_ratio_)

    #this graph shows something different?
    total_variance = np.sum(pca.explained_variance_)
    cum_variance = np.cumsum(pca.explained_variance_)
    prop_var_expl = cum_variance/total_variance

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(prop_var_expl, color='red', linewidth=2, label='Explained variance')
    ax.axhline(0.9, label='90% goal', linestyle='--', color="black", linewidth=1)
    ax.set_ylabel('cumulative prop. of explained variance')
    ax.set_xlabel('number of principal components')
    ax.legend()
    plt.show();