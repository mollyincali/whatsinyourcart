import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

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

    #--- looking for ideal PCA number
    ratio = []
    for num in range(6, 134):
        pca = PCA(n_components=num)
        pca.fit(pca_scaled)
        X_pca = pca.transform(pca_scaled)
        ratio.append([num, np.sum(pca.explained_variance_ratio_)])
    
    #make df
    ratio_df = pd.DataFrame(ratio)

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

    #---    graph variance
    fig, ax = plt.subplots(figsize=(20,10))
    ax.plot(ratio_df[1], color='#F46708', linewidth=3, label='Explained Variance')
    ax.axhline(0.9, label='90% goal', linestyle='--', linewidth='2', color="#F1D78C")
    ax.axvline(110, label='Target # PCA = 117', linestyle='--', linewidth='2', color="#E84846")
    ax.set_ylabel('Explained Variance Ratio', fontdict=fontaxis)
    ax.set_xlabel('Number of Principal Components', fontdict=fontaxis)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.title('Principal Component Analysis', fontdict=fonttitle)
    ax.legend()
    plt.show();

    