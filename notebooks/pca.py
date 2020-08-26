import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
    ax.scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='k', s=40)
    ax.set_title("First two PCA directions")
    ax.set_xlabel("1st eigenvector (PC1)")
    ax.set_ylabel("2nd eigenvector (PC2)")
    plt.show();

    #this graph shows something different?
    # total_variance = np.sum(pca.explained_variance_)
    # cum_variance = np.cumsum(pca.explained_variance_)
    # prop_var_expl = cum_variance/total_variance

    #looking for ideal PCA number
    ratio = []
    for num in range(6, 134):
        pca = PCA(n_components=num)
        pca.fit(pca_scaled)
        X_pca = pca.transform(pca_scaled)
        ratio.append([num, np.sum(pca.explained_variance_ratio_)])
    
    #make df
    ratio_df = pd.DataFrame(ratio)

    #graph variance
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(ratio_df[1], color='#F46708', linewidth=3, label='Explained Variance')
    ax.axhline(0.9, label='90% goal', linestyle='--', linewidth='2', color="#F1D78C")
    ax.axvline(111, label='Target # PCA = 117', linestyle='--', linewidth='2', color="#E84846")
    ax.set_ylabel('Explained Variance Ratio', fontdict=fontaxis)
    ax.set_xlabel('Number of Principal Components', fontdict=fontaxis)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend()
    plt.show();

    