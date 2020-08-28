import numpy as np
import pandas as pd
import tensorflow as tf
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
        keras.layers.Dense(128, activation='reLu', input_shape=[5]), 
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
        keras.layers.Dense(64, activation='swish'), 
        keras.layers.Dense(1)
        ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

'''
Epoch 1/5
loss: 298.1448 - accuracy: 0.6887 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 2/5
loss: 0.7434 - accuracy: 0.7334 - val_loss: 0.5793 - val_accuracy: 0.7339
Epoch 3/5
loss: 0.8325 - accuracy: 0.7334 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 4/5
loss: 0.7049 - accuracy: 0.7335 - val_loss: 0.5793 - val_accuracy: 0.7339
Epoch 5/5
loss: 0.6982 - accuracy: 0.7335 - val_loss: 0.5793 - val_accuracy: 0.7339
'''

    #--- Neural Network run4
    model = keras.models.Sequential([
        keras.layers.Dense(256, activation='swish', input_shape=[5]),
        keras.layers.Dense(128, activation='swish'),
        keras.layers.Dense(64, activation='swish'),
        keras.layers.Dense(64, activation='swish'), 
        keras.layers.Dense(1)
        ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
'''
Epoch 1/20
loss: 25.1232 - accuracy: 0.7265 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 2/20
loss: 1.2212 - accuracy: 0.7335 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 3/20
loss: 0.7131 - accuracy: 0.7335 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 4/20
loss: 0.8418 - accuracy: 0.7335 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 5/20
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5795 - val_accuracy: 0.7339
Epoch 6/20
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5796 - val_accuracy: 0.7339
Epoch 7/20
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5793 - val_accuracy: 0.7339
Epoch 8/20
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 9/20
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 10/20
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5793 - val_accuracy: 0.7339
Epoch 11/20
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5793 - val_accuracy: 0.7339
Epoch 12/20
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5793 - val_accuracy: 0.7339
Epoch 13/20
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5793 - val_accuracy: 0.7339
Epoch 14/20
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 15/20
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 16/20
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5793 - val_accuracy: 0.7339
Epoch 17/20
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5793 - val_accuracy: 0.7339
Epoch 18/20
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 19/20
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 20/20
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5794 - val_accuracy: 0.7339
'''

    #---    Neural Net5
    model = keras.models.Sequential([
        keras.layers.Dense(256, activation='swish', input_shape=[5]),
        keras.layers.Dense(256, activation='swish'),
        keras.layers.Dense(128, activation='swish'),
        keras.layers.Dense(64, activation='swish'),
        keras.layers.Dense(64, activation='swish'), 
        keras.layers.Dense(1)
        ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
run4acc = [0.7262, 0.7335, 0.7336, 0.7336, 0.7336]
'''
Epoch 1/5
loss: 21.1934 - accuracy: 0.7262 - val_loss: 0.5795 - val_accuracy: 0.7339
Epoch 2/5
loss: 1.0010 - accuracy: 0.7335 - val_loss: 0.5793 - val_accuracy: 0.7339
Epoch 3/5
loss: 0.5798 - accuracy: 0.7336 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 4/5
loss: 0.5798 - accuracy: 0.7336 - val_loss: 0.5794 - val_accuracy: 0.7339
Epoch 5/5
loss: 0.5797 - accuracy: 0.7336 - val_loss: 0.5793 - val_accuracy: 0.7339
'''

    #ugly graphing data from NN
    # fig, axs = plt.subplots(1, 4, figsize = (5,5), sharey = True)
    # acc = [[0.6140, 0.6274, 0.6700], [0.6138, 0.6281, 0.7076],
    #         [0.6887, 0.7334, 0.7334, 0.7335, 0.7335], [0.7262, 0.7335, 0.7336, 0.7336, 0.7336]]
    # for i, ax in enumerate(axs.flatten()):
    #     ax.plot(acc[i])
    # plt.show();

    fig, axs = plt.subplots(1, 2, figsize = (10,10), sharey = True)
    acc = [[0.6140, 0.6274, 0.6700], [0.6138, 0.6281, 0.7076],
            [0.6887, 0.7334, 0.7334, 0.7335, 0.7335], [0.7262, 0.7335, 0.7336, 0.7336, 0.7336]]
    for i, ax in enumerate(axs.flatten()):
        ax.scatter(np.arange(3), acc[i], alpha=0.8)
    plt.show();

    fig, ax = plt.subplots(1, 2, figsize = (10,10), sharey = True)
    x = range(3)
    ax[0,0] = sns.scatterplot([x, run1acc, alpha=0.8)
    ax[0,0] = sns.scatterplot(x, run2acc, alpha=0.8)
    ax[0,0] = sns.scatterplot(x, run3acc, alpha=0.8)
    ax[0,0] = sns.scatterplot(x, run4acc, alpha=0.8)
    plt.show();

run1acc = [0.6140, 0.6274, 0.6700]
run2acc = [0.6138, 0.6281, 0.7076]
run3acc = [0.6887, 0.7334, 0.7334, 0.7335, 0.7335]
run4acc = [0.7262, 0.7335, 0.7336, 0.7336, 0.7336]