import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD

if __name__ == "__main__":
    #---    Upload full csv
    full = pd.read_csv("../full.csv")

    full['banana1'] = np.where(full['product_name'] == 'Banana', 1, 0)
    full['banana2'] = np.where(full['product_name'] == 'Bag of Organic Bananas', 1, 0)
    full['banana'] = full['banana2'] + full['banana1']

    by_order = full.groupby('order_id').agg({'order_dow':'max', 'order_hour_of_day':"max", 
                                    'days_since_prior_order':'max', 'add_to_cart_order':'max', 
                                    'banana':'max'}).reset_index()

    #drops NaN in "Days Since Prior Order" which represents first time user
    by_order.dropna(inplace = True)
    y = by_order.pop('banana')

    X_train, X_test, y_train, y_test = train_test_split(by_order, y, test_size=0.3, random_state=3, stratify = y)