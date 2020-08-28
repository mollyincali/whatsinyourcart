#TFIDF stuff on product name not helpful...
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
    table = pd.pivot_table(X_train, index = ['user_id'], columns = ['product_id'], aggfunc = np.unique) 

    #---    by product testby_prod
    by_prod.groupby('product_name').agg({"order_number":"max"})
    
    list_sum1 = pd.Series(list_sumby_prod).reset_index()
    merge = pd.merge(grouped, list_sum1, on = 'index')
    merge['perc'] = merge['order_number'] / merge[0]

    list_sum = []
    for prod in by_prod['product_name'].unique():
        list_sum.append(np.sumd(by_prod[by_prod['product_name'] == prod]['order_number']))

    #---    df to look at products by hour
    baked = by_prod[by_prod['product_name'] == 'Half Baked Frozen Yogurt'].copy()   
    total_num_orders = by_hour['order_number'].sum() 
    by_hour['percent'] = (by_hour['order_number'] / total_num_orders).round(1)             
    by_hour = part.groupby(['order_hour_of_day', 'product_name']).agg({'order_number':'size'}).reset_index()  
    order_num_array = np.array(by_hour.groupby('order_hour_of_day').agg({'order_number':'sum'}))
    graphs by hour

    test20 = pd.merge(graph20, merge, on = 'product_name')
    test20 = test20[['product_name', 'order_hour_of_day', 'perc']].sort_values(by='perc', ascending = False)[:50]

    test10 = pd.merge(graph10, merge, on = 'product_name')
    test10 = test10[['product_name', 'order_hour_of_day', 'perc']].sort_values(by='perc', ascending = False)[:50]

    part = full[['product_name','order_hour_of_day','order_number']].copy()
    by_prod = part.groupby(['product_name', 'order_hour_of_day']).agg({'order_number':'size'}).copy().reset_index()
    graph15 = by_prod[by_prod['order_hour_of_day'] == 15].sort_values(by='order_number', ascending = False).reset_index()
    graph10 = by_prod[by_prod['order_hour_of_day'] == 10].sort_values(by='order_number', ascending = False).reset_index()

    x = []
    y = []
    fig, ax = plt.subplots(figsize = (20, 10))
    for idx in range(10):
        x = graph10.iloc[idx, 2], y = append(graph10.iloc[idx, 3]
    plt.show()
    
    #---    Old code for trees
    #---    Upload test / train csv
    X_test = pd.read_csv("../X_test.csv")
    y_test = pd.read_csv("../y_test.csv")
    X_train = pd.read_csv("../X_train.csv")
    y_train = pd.read_csv("../y_train.csv")

    X_test.drop('Unnamed: 0', axis = 1, inplace = True)
    y_test.drop('Unnamed: 0', axis = 1, inplace = True)
    X_train.drop('Unnamed: 0', axis = 1, inplace = True)
    y_train.drop('Unnamed: 0', axis = 1, inplace = True)

    #---    Get dummies on product_id, will need to merge back to OG X_train
    predict = X_train[['user_id', 'product_id']] 
    predict = pd.get_dummies(predict['product_id'])

    predict_test = X_test[['user_id', 'product_id']] 
    predict_test = pd.get_dummies(predict_test['product_id'])

    # cm = confusion_matrix(y_test, y_predict)
    # print(f'Confusion Matrix of Random Foreclearst Best Params: \n {cm}')
    # print(f'True Negatives: {round((cm[0,0] / np.sum(cm)), 4)}, \nFalse Negatives: {round((cm[1,0] / np.sum(cm)), 4)}, \nTrue Positives: {round((cm[1,1] / np.sum(cm)), 4)}, \nFalse Positives: {round((cm[0,1] / np.sum(cm)), 4)}')

    # full_train = full_train[['product_id', 'add_to_cart_order', 'user_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]
    # organic = products['product_name'].str.contains('Organic')

    '''
    #---    Gradient Boost
    gradient_boosting_grid = {'learning_rate': [0.05, 0.1],
                          'max_depth': [3, 5],
                          'min_samples_leaf': [3, 5],
                          'n_estimators': [100],
                          'random_state': [3]}
    gdbr_best_params, gdbr_best_model = gridsearch_with_output(GradientBoostingClassifier(), 
                                                           gradient_boosting_grid, 
                                                           X_train, y_train)
'''

# f1=[0.3486948253722419,0.3381415603078138,0.1314208385880886,0.5022338996096836,0.23750991898580134]
# mean_acc = [0.6473099261860604,0.6777782142186052,0.7397073982736052,0.6031829275672187,0.7419144913335237]                                                                                                                                                                  
# model = ['Decision Tree','Random Forest Basic','Random Forest Best Param',
#          'Random Forest Class Weights','Gradient Boost']

''' banana old run
    # #dat from previous run
    # oldf1 = [0.3077210344505648, 0.3257932199300277, 0.20258872651356993, 0.5025858015984956]
    # oldmean_acc = [0.7034, 0.7019466666666667, 0.74536, 0.6332266666666667]
    # oldmodel = ['Decision Tree','Random Forest Basic','Random Forest Best Param', 'Random Forest Class Weights'   ]
    # #---    Graph Score and F1 and Model
    # fig, ax = plt.subplots(figsize = (20, 10))
    # ax= sns.lineplot(x = model, y = oldf1, color= '#EF727F', dashes=[(1, 1), (5, 10)], linewidth = 5, label = 'F1 Score')
    # ax = sns.lineplot(x = model, y = oldmean_acc, color='#F6A811', linewidth = 5, label = 'Accuracy Score')    
    # ax.set_ylim(ymin = 0.1, ymax = 0.9)
    # ax.tick_params(axis='both', which='major', labelsize=18)
    # plt.xticks(rotation = 10)
    # plt.title('Did you order Bananas? \n Mean Accuracy Score and F1 Score by Model', fontdict=fonttitle)
    # plt.show();