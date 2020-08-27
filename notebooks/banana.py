import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
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

def balance_work(y_train):
    n1 = np.sum(y_train)
    n2 = len(y_train) - n1
    n_samples = n1 + n2
    w1 = n_samples / (2 * n1)
    w2 = n_samples / (2 * n2)
    return w1, w2

if __name__ == '__main__':
    #---    Upload full csv
    full = pd.read_csv("../full.csv")

    full['banana1'] = np.where(full['product_name'] == 'Banana', 1, 0)
    full['banana2'] = np.where(full['product_name'] == 'Bag of Organic Bananas', 1, 0)
    full['banana'] = full['banana2'] + full['banana1']

    #grouped by user 120,725 ban / 206,209 orders 58% of people have ordered a banana
    # by_user = full.groupby('user_id').agg({'order_dow':'max', 'order_hour_of_day':"max", 
    #                                 'days_since_prior_order':'max', 'add_to_cart_order':'max', 
    #                                 'banana':'max'}).reset_index()

    #grouped by orderid 885,017 / 3,346,083 orders 25% of orders have a banana
    by_order = full.groupby('order_id').agg({'order_dow':'max', 'order_hour_of_day':"max", 
                                    'days_since_prior_order':'max', 'add_to_cart_order':'max', 
                                    'banana':'max'}).reset_index()

    #drops NaN in "Days Since Prior Order" which represents first time user
    by_order.dropna(inplace = True)
    y = by_order.pop('banana')
    X_train, X_test, y_train, y_test = train_test_split(by_order, y, test_size=0.3, random_state=3, stratify = y)

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

    #---    Get class weights
    w1, w2 = balance_work(y_train)

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

    gb = GradientBoostingClassifier(learning_rate = 0.1, n_estimators=100, max_depth = 5)
    gb.fit(X_train, y_train)
    y_predict = gb.predict(X_test)
    gb_score = gb.score(X_test, y_test)
    print(f'Gradient Boost Mean Accuracy: {gb_score:.5}')
    print(f'Gradient Boost F1 Score: {(f1_score(y_test, y_predict)):.5}')
    mean_acc.append(gb_score)
    f1.append((f1_score(y_test, y_predict)))
    model.append('Gradient Boost')

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

    #---    With more data
    model = ['Decision Tree','Random Forest Basic','Random Forest Best Param',
         'Random Forest Class Weights','Gradient Boost']
    fig, ax = plt.subplots(figsize = (20, 10))
    ax = sns.lineplot(x = model, y = f1, color= '#EF727F', marker='*', linewidth = 5, label = 'More Data F1 Score')
    ax = sns.lineplot(x = model, y = mean_acc, color='#F6A811', marker='*', linewidth = 5, label = 'More Data Mean Accuracy Score')    
    ax.set_ylim(ymin = 0.1, ymax = 0.9)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.xticks(rotation = 10)
    plt.title('Did you order Bananas? \n Additional Data Mean Accuracy Score and F1 Score by Model', fontdict=fonttitle)
    plt.show();


# f1=[0.3486948253722419,0.3381415603078138,0.1314208385880886,0.5022338996096836,0.23750991898580134]
# mean_acc = [0.6473099261860604,0.6777782142186052,0.7397073982736052,0.6031829275672187,0.7419144913335237]                                                                                                                                                                  
# model = ['Decision Tree','Random Forest Basic','Random Forest Best Param',
#          'Random Forest Class Weights','Gradient Boost']

'''
    #dat from previous run
    oldf1 = [0.3077210344505648, 0.3257932199300277, 0.20258872651356993, 0.5025858015984956]
    oldmean_acc = [0.7034, 0.7019466666666667, 0.74536, 0.6332266666666667]
    oldmodel = ['Decision Tree','Random Forest Basic','Random Forest Best Param', 'Random Forest Class Weights']
    #---    Graph Score and F1 and Model
    fig, ax = plt.subplots(figsize = (20, 10))
    ax= sns.lineplot(x = model, y = oldf1, color= '#EF727F', dashes=[(1, 1), (5, 10)], linewidth = 5, label = 'F1 Score')
    ax = sns.lineplot(x = model, y = oldmean_acc, color='#F6A811', linewidth = 5, label = 'Accuracy Score')    
    ax.set_ylim(ymin = 0.1, ymax = 0.9)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.xticks(rotation = 10)
    plt.title('Did you order Bananas? \n Mean Accuracy Score and F1 Score by Model', fontdict=fonttitle)
    plt.show();
'''