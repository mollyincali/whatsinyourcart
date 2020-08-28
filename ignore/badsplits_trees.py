import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score

#---    Model Codes
def gridsearch_with_output(estimator, parameter_grid, X_train, y_train):
    '''
        Parameters: estimator: the type of model (e.g. RandomForestRegressor())
                    paramter_grid: dictionary defining the gridsearch parameters
                    X_train: 2d numpy array
                    y_train: 1d numpy array
        Returns:  best parameters and model fit with those parameters
    '''
    model_gridsearch = GridSearchCV(estimator,
                                    parameter_grid,
                                    n_jobs=-1,
                                    verbose=True,
                                    scoring='neg_mean_squared_error')
    model_gridsearch.fit(X_train, y_train)
    best_params = model_gridsearch.best_params_ 
    model_best = model_gridsearch.best_estimator_
    print("\nResult of gridsearch:")
    print("{0:<20s} | {1:<8s} | {2}".format("Parameter", "Optimal", "Gridsearch values"))
    print("-" * 55)
    for param, vals in parameter_grid.items():
        print("{0:<20s} | {1:<8s} | {2}".format(str(param), 
                                                str(best_params[param]),
                                                str(vals)))
    return best_params, model_best

if __name__ == '__main__':
    #---    Upload test / train csv
    X_test = pd.read_csv("../X_test.csv")
    X_train = pd.read_csv("../X_train.csv")

    # looking for if order had B
    # products[products['product_name'] == 'Banana'] #24852    
    X_test['banana1'] = np.where(X_test['product_id'] == 24852, 1, 0)
    X_test['banana2'] = np.where(X_test['product_id'] == 13176, 1, 0)
    X_test['banana'] = X_test['banana2'] + X_test['banana1']
    X_test = X_test.groupby('user_id').agg({'order_dow':'max', 'order_hour_of_day':"max", 
                                    'days_since_prior_order':'max', 'add_to_cart_order':'max', 
                                    'banana':'max'}).reset_index()
    X_test.drop('user_id', axis = 1, inplace = True)
    y_test = X_test.pop('banana') #1-10744, 0-64256

    X_train['banana1'] = np.where(X_train['product_id'] == 24852, 1, 0)
    X_train['banana2'] = np.where(X_train['product_id'] == 13176, 1, 0)
    X_train['banana'] = X_train['banana2'] + X_train['banana1']
    X_train = X_train.groupby('user_id').agg({'order_dow':'max', 'order_hour_of_day':"max", 
                                    'days_since_prior_order':'max', 'add_to_cart_order':'max', 
                                    'banana':'max'}).reset_index()
    X_train.drop('user_id', axis = 1, inplace = True)
    y_train = X_train.pop('banana') #1-18726, 0-112483

    #---    Empty lists to be appended
    mean_acc = []
    f1 = []
    model = []

    #---    Decision Tree Basic Model
    clf = DecisionTreeClassifier(random_state = 3)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    dt_score = clf.score(X_test, y_test)
    print(f'Decision Tree Score Best Param: {dt_score:.5}')
    print(f'Decision Tree F1 Score: {(f1_score(y_test, y_predict)):.5}\n')
    mean_acc.append(dt_score)
    f1.append((f1_score(y_test, y_predict)))
    model.append('Decision Tree')

    #---    Random Forest Basic Model 
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)
    rf_score = rf.score(X_test, y_test)
    print(f'Random Forest Score: {rf_score:.5}')
    print(f'Random Forest F1 Score: {(f1_score(y_test, y_predict)):.5}\n')
    mean_acc.append(rf_score)
    f1.append((f1_score(y_test, y_predict)))
    model.append('Random Forest Basic')

    #---    Random Forest GridSearch
#     random_forest_grid = {'max_depth': [4, 8, 12],
#                       'max_features': [3, 4],
#                       'min_samples_split': [4, 8],
#                       'n_estimators': [80, 100],
#                       'random_state': [3]}

#     rf_best_params, rf_best_model = gridsearch_with_output(RandomForestClassifier(), 
#                                                        random_forest_grid, 
#                                                        X_train, y_train)

    #---    Random Forest BEST Model 
    rf = RandomForestClassifier(max_depth=4, max_features=3, min_samples_split=4, n_estimators=80, random_state=3)
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)
    rf_score = rf.score(X_test, y_test)
    print(f'Random Forest Score Best Param: {rf_score:.5}')
    print(f'Random Forest F1 Score Best Param: {(f1_score(y_test, y_predict)):.5}\n')
    mean_acc.append(rf_score)
    f1.append((f1_score(y_test, y_predict)))
    model.append('Random Forest Best Param')

    #---    Balance Work
    n1 = np.sum(y_train)
    n2 = len(y_train) - n1
    n_samples = n1 + n2
    w1 = n_samples / (2 * n1)
    w2 = n_samples / (2 * n2)
    print(f"w1: {w1:0.2f}, w2: {w2:0.2f}")

    #---    Random Forest Trying to Balance with Class Weights
    rf = RandomForestClassifier(max_depth=4, max_features=3, min_samples_split=4, bootstrap=True,
                    n_estimators=80, random_state=3, class_weight={1: w1, 0: w2})
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)
    rf_score = rf.score(X_test, y_test)
    print(f'Random Forest Mean Accuracy Weight + Bootstrap: {rf_score:.5}')
    print(f'Random Forest F1 Score Weight + Bootstrap: {(f1_score(y_test, y_predict)):.5}')
    mean_acc.append(rf_score)
    f1.append((f1_score(y_test, y_predict)))
    model.append('Random Forest Class Weights')

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

    #---    Graph Score and F1 and Model
    fig, ax = plt.subplots(figsize = (20, 10))
    ax = sns.lineplot(x = model, y = f1, color= '#F6A811', marker='*', linewidth = 5, label = 'F1 Score')
    ax = sns.lineplot(x = model, y = mean_acc, color='#EF727F', marker='*', linewidth = 5, label = 'Mean Accuracy Score')
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.xticks(rotation = 10)
    plt.title('Mean Accuracy Score and F1 Score', fontdict=fonttitle)
    plt.show();
