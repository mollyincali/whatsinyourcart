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
    fig.suptitle(f'Is it Organic? Neural Net Model {len(acc)} Epoch, {details}', fontdict=fonttitle)    
    plt.legend()
    plt.show();

if __name__ == '__main__':
    #---    Upload full csv
    # X_train = pd.read_csv("../organicxtrain.csv")
    # X_test = pd.read_csv("../organicxtest.csv")
    # y_train = pd.read_csv("../organicytrain.csv")
    # y_test = pd.read_csv("../organicytest.csv")

    # #---    Neural Network run1
    # model = keras.models.Sequential([
    #     keras.layers.Dense(128, activation='relu', input_shape=[5]), 
    #     keras.layers.Dense(1)
    #     ])
    # model.compile(optimizer='adam',
    #           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #           metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.2)

    loss1 = [792.2125, 217.6456, 54.1215]
    acc1 = [0.6306, 0.6468, 0.6568]
    valloss1 = [513.0070, 595.2487, 1.6928]
    valacc1 = [0.7375, 0.7372, 0.6548]
'''
Epoch 1/3
loss: 792.2125 - accuracy: 0.6306 - val_loss: 513.0070 - val_accuracy: 0.7375
Epoch 2/3
loss: 217.6456 - accuracy: 0.6468 - val_loss: 595.2487 - val_accuracy: 0.7372
Epoch 3/3
loss: 54.1215 - accuracy: 0.6568 - val_loss: 1.6928 - val_accuracy: 0.6548
'''

    #--- Neural Network run2
    # model = keras.models.Sequential([
    #     keras.layers.Dense(128, activation='swish', input_shape=[5]), 
    #     keras.layers.Dense(1)
    #     ])
    # model.compile(optimizer='adam',
    #           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #           metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.2)

    loss2 = [757.4401, 164.3079, 37.7631]
    acc2 = [0.6303, 0.6487, 0.6581]
    valloss2 = [20.3793, 20.4697, 16.3025]
    valacc2 = [0.6766, 0.7429, 0.7377]
'''
Epoch 1/3
loss: 757.4401 - accuracy: 0.6303 - val_loss: 20.3793 - val_accuracy: 0.6766
Epoch 2/3
loss: 164.3079 - accuracy: 0.6487 - val_loss: 20.4697 - val_accuracy: 0.7429
Epoch 3/3
loss: 37.7631 - accuracy: 0.6581 - val_loss: 16.3025 - val_accuracy: 0.7377
'''

    #--- Neural Network run3
    # model = keras.models.Sequential([
    #     keras.layers.Dense(128, activation='swish', input_shape=[5]),
    #     keras.layers.Dense(64, activation='swish'), 
    #     keras.layers.Dense(1)
    #     ])
    # model.compile(optimizer='adam',
    #           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #           metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

    loss3 = [247.2045, 0.6649, 0.6303, 0.6677, 0.6695]
    acc3 = [0.6969, 0.7378, 0.7377, 0.7378, 0.7377]
    valloss3 = [0.5764, 0.5763, 0.5764, 0.5763, 0.5763]
    valacc3 = [0.7369, 0.7369, 0.7369, 0.7369, 0.7369]

'''
Epoch 1/5
loss: 247.2045 - accuracy: 0.6969 - val_loss: 0.5764 - val_accuracy: 0.7369
Epoch 2/5
loss: 0.6649 - accuracy: 0.7378 - val_loss: 0.5763 - val_accuracy: 0.7369
Epoch 3/5
loss: 0.6303 - accuracy: 0.7377 - val_loss: 0.5764 - val_accuracy: 0.7369
Epoch 4/5
loss: 0.6677 - accuracy: 0.7378 - val_loss: 0.5763 - val_accuracy: 0.7369
Epoch 5/5
loss: 0.6695 - accuracy: 0.7377 - val_loss: 0.5763 - val_accuracy: 0.7369
'''

    #---    Neural Net4
    # model = keras.models.Sequential([
    #     keras.layers.Dense(256, activation='swish', input_shape=[5]),
    #     keras.layers.Dense(256, activation='swish'),
    #     keras.layers.Dense(128, activation='swish'),
    #     keras.layers.Dense(64, activation='swish'),
    #     keras.layers.Dense(64, activation='swish'), 
    #     keras.layers.Dense(1)
    #     ])
    # model.compile(optimizer='adam',
    #           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #           metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

    loss4 = [12.4360, 0.5857, 0.6313, 0.5754, 0.5754]
    acc4 = [0.7293,0.7278, 0.7377, 0.7378, .7378]
    valloss4 = [0.5768, 0.5763, 0.5764, 0.5764, 0.5763]
    valacc4 = [0.7369, 0.7369, 0.7369, 0.7369, 0.7369]
'''
Epoch 1/5
loss: 12.4360 - accuracy: 0.7293 - val_loss: 0.5768 - val_accuracy: 0.7369
Epoch 2/5
loss: 0.5857 - accuracy: 0.7378 - val_loss: 0.5763 - val_accuracy: 0.7369
Epoch 3/5
0.6313 - accuracy: 0.7377 - val_loss: 0.5764 - val_accuracy: 0.7369
Epoch 4/5
loss: 0.5754 - accuracy: 0.7378 - val_loss: 0.5764 - val_accuracy: 0.7369
Epoch 5/5
loss: 0.5754 - accuracy: 0.7378 - val_loss: 0.5763 - val_accuracy: 0.7369
'''

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
    nn_plot(acc1, loss1, valacc1, valloss1, '1 Layer, and ReLU')
    nn_plot(acc2, loss2, valacc2, valloss2, '1 Layer, and Swish')
    nn_plot(acc3, loss3, valacc3, valloss3, '2 Layers, and Swish')
    nn_plot(acc4, loss4, valacc4, valloss4, '5 Layers, and Swish')
