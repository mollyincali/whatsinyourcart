
def run_model(models, X_train, X_test, y_train, y_test):
    mean_acc = []
    f1 = []
    model_lst = []
    
    for model in models:
        rf = model.fit(X_train, y_train)
        y_predict = rf.predict(X_test)
        rf_score = rf.score(X_test, y_test)
        mean_acc.append(rf_score)
        f1.append((f1_score(y_test, y_predict)))
        model_lst.append(model)

    return mean_acc, f1, model

def balance_work(y_train):
    n1 = np.sum(y_train)
    n2 = len(y_train) - n1
    n_samples = n1 + n2
    w1 = n_samples / (2 * n1)
    w2 = n_samples / (2 * n2)
    return w1, w2

if __name__ == "__main__"
    w1, w2 = balance_work(y_train)
    
    mean_acc, f1, model = run_model(models, X_train, X_test, y_train, y_test)

    models = ['DecisionTreeClassifier(random_state = 3)',
        'RandomForestClassifier(n_estimators=50)',
        'RandomForestClassifier(max_depth=4, max_features=3, min_samples_split=4, n_estimators=80, random_state=3)',
        'RandomForestClassifier(max_depth=4, max_features=3, min_samples_split=4, bootstrap=True, n_estimators=80, random_state=3, class_weight={1: w1, 0: w2})'
