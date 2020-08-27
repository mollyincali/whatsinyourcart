'''
Start tensorflow docker container: docker exec -it tensorflow /bin/bash
pip install -U scikit-learn  
pip install pandas
ipython
get into correct folder run below

'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD

if __name__ == "__main__":
    #---    Upload full csv
    full = pd.read_csv("whatsinyourcart/full.csv")

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

    #--- Neural Network run1
    model = keras.models.Sequential([
        keras.layers.Dense(128, activation='swish', input_shape=[5]), 
        keras.layers.Dense(1)
        ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.2)

'''
Run1 with 1 layer relu
Epoch 1/3
loss: 793.7966 - accuracy: 0.6140 - val_loss: 270.4421 - val_accuracy: 0.2913
Epoch 2/3
loss: 181.4234 - accuracy: 0.6274 - val_loss: 57.9650 - val_accuracy: 0.7322
Epoch 3/3
loss: 21.2258 - accuracy: 0.6700 - val_loss: 0.5802 - val_accuracy: 0.7339
'''

    #--- Neural Network run2
    model = keras.models.Sequential([
        keras.layers.Dense(128, activation='swish', input_shape=[5]), 
        keras.layers.Dense(1)
        ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.2)
'''
Epoch 1/3
loss: 873.9270 - accuracy: 0.6138 - val_loss: 136.7671 - val_accuracy: 0.7283
Epoch 2/3
loss: 167.2178 - accuracy: 0.6281 - val_loss: 38.5526 - val_accuracy: 0.3124
Epoch 3/3
loss: 4.1501 - accuracy: 0.7076 - val_loss: 0.5795 - val_accuracy: 0.7339
'''

    #--- Neural Network run3
    model = keras.models.Sequential([
        keras.layers.Dense(128, activation='swish', input_shape=[5]),
        keras.layers.Dense(64, activation='swish', input_shape=[5]), 
        keras.layers.Dense(1)
        ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
'''
Epoch 1/3
loss: 179.9152 - accuracy: 0.6912 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 2/3
loss: 0.7337 - accuracy: 0.7334 - val_loss: 0.5795 - val_accuracy: 0.7339
Epoch 3/3
loss: 0.9619 - accuracy: 0.7334 - val_loss: 0.5793 - val_accuracy: 0.7339
'''

    #--- Neural Network run4
    model = keras.models.Sequential([
        keras.layers.Dense(128, activation='swish', input_shape=[5]),
        keras.layers.Dense(64, activation='swish', input_shape=[5]), 
        keras.layers.Dense(1)
        ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
'''
Epoch 1/5
loss: 307.1750 - accuracy: 0.6727 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 2/5
loss: 0.7176 - accuracy: 0.7332 - val_loss: 0.5793 - val_accuracy: 0.7339
Epoch 3/5
loss: 0.8530 - accuracy: 0.7333 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 4/5
loss: 0.8805 - accuracy: 0.7333 - val_loss: 0.5793 - val_accuracy: 0.7339
Epoch 5/5
loss: 0.9131 - accuracy: 0.7334 - val_loss: 0.5793 - val_accuracy: 0.7339
'''

    #--- Neural Network run5
    model = keras.models.Sequential([
        keras.layers.Dense(128, activation='swish', input_shape=[5]),
        keras.layers.Dense(128, activation='swish', input_shape=[5]),
        keras.layers.Dense(64, activation='swish', input_shape=[5]), 
        keras.layers.Dense(1)
        ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

'''
Epoch 1/5
loss: 54.7439 - accuracy: 0.7221 - val_loss: 0.5796 - val_accuracy: 0.7339
Epoch 2/5
loss: 1.2292 - accuracy: 0.7332 - val_loss: 0.5795 - val_accuracy: 0.7339
Epoch 3/5
loss: 0.8140 - accuracy: 0.7334 - val_loss: 0.5793 - val_accuracy: 0.7339
Epoch 4/5
loss: 0.6572 - accuracy: 0.7335 - val_loss: 0.5796 - val_accuracy: 0.7339
Epoch 5/5
loss: 0.7685 - accuracy: 0.7335 - val_loss: 0.5794 - val_accuracy: 0.7339
'''

    #--- Neural Network run6
    model = keras.models.Sequential([
        keras.layers.Dense(256, activation='swish', input_shape=[5]),
        keras.layers.Dense(128, activation='swish', input_shape=[5]),
        keras.layers.Dense(64, activation='swish', input_shape=[5]),
        keras.layers.Dense(64, activation='swish', input_shape=[5]), 
        keras.layers.Dense(1)
        ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
