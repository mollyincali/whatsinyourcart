import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD

def nn_plot(acc, loss, testacc, testloss, details):
    fig, ax = plt.subplots(1, 2, figsize = (10,5))
    ax[0].plot(range(1,len(acc) + 1), acc, alpha=0.8, label = 'train', color = '#F6A811', linewidth = 4)
    ax[0].plot(range(1,len(acc) + 1), testacc, alpha=0.8, label = 'validation', color = '#EF727F', linewidth = 4)
    ax[0].set_title('Accuracy')
    ax[1].plot(range(1,len(acc) + 1), loss, alpha=0.8, label = 'train', color = '#F6A811', linewidth = 4)
    ax[1].plot(range(1,len(acc) + 1), testloss, alpha=0.8, label = 'validation', color = '#EF727F', linewidth = 4)                   
    ax[1].set_title('Loss')    
    fig.suptitle(f'Banana? Neural Net Model {len(acc)} Epoch, {details}', fontdict=fonttitle)    
    plt.legend()
    plt.show();

if __name__ == "__main__":
    #---    Import train, test
    X_train = pd.read_csv("../bananaxtrain.csv")
    X_test = pd.read_csv("../bananaxtest.csv")
    y_train = pd.read_csv("../bananaytrain.csv")
    y_test = pd.read_csv("../bananaytest.csv")

    #---    Neural Network run1
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

    #---    Neural Net4
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

    acc1 = [0.6140, 0.6274, 0.6700]
    loss1 =[793.7966, 181.4234, 21.2258]
    testloss1 =[270.4421, 57.9650, 0.5802]
    testacc1 = [0.2913, 0.7322, 0.7339]

    testloss2 = [136.7671, 38.5526, 0.5795]
    testacc2 = [0.7283, 0.3124, 0.7339]
    loss2 = [873.9270, 167.2178, 4.1501]
    acc2 = [0.6138, 0.6281, 0.7076]

    loss3 = [298.1448,0.7434,0.8325,0.7049,0.6982]
    acc3 = [0.6887, 0.7334, 0.7334, 0.7335, 0.7335]
    testloss3= [0.5794,0.5793,0.5794,0.5793,0.5793]
    testacc3 = [0.7339,0.7339,0.7339,0.7339,0.7339]

    acc4 = [0.7262, 0.7335, 0.7336, 0.7336, 0.7336]
    loss4 =[21.1934,1.0010,0.5798, 0.5798, 0.5797]
    testloss4 =[0.5795, 0.5793,0.5794,0.5794,0.5793]
    testacc4 = [0.7339,0.7339,0.7339,0.7339,0.7339]

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

    #---    Calling graph function
    nn_plot(acc1, loss1, testacc1, testloss1, '1 Layer, and ReLU')
    nn_plot(acc2, loss2, testacc2, testloss2, '1 Layer, and Swish')
    nn_plot(acc3, loss3, testacc3, testloss3, '2 Layers, and Swish')
    nn_plot(acc4, loss4, testacc4, testloss4, '5 Layers, and Swish')