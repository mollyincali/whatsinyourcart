import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD

def nn_1layer(epoch, activate):
    '''
    input:  epoch and activation
    return: List of loss, validation loss, accuracy and validation accuracy 
    '''
    model = keras.models.Sequential([
        keras.layers.Dense(128, activation=activate, input_shape=[5]), 
        keras.layers.Dense(1)
        ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=32, validation_split=0.2)
    return history.history['loss'], history.history['val_loss'], history.history['accuracy'], history.history['val_accuracy']

def nn_2layers(epoch, activate):
    '''
    input:  epoch and activation
    return: List of loss, validation loss, accuracy and validation accuracy 
    '''
    model = keras.models.Sequential([
        keras.layers.Dense(128, activation=acivate, input_shape=[5]),
        keras.layers.Dense(64, activation=acivate), 
        keras.layers.Dense(1)
        ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=32, validation_split=0.2)
    return history.history['loss'], history.history['val_loss'], history.history['accuracy'], history.history['val_accuracy']

def nn_5layers(epoch, activate):
    '''
    input:  epoch and activation
    return: List of loss, validation loss, accuracy and validation accuracy 
    '''
    model = keras.models.Sequential([
            keras.layers.Dense(256, activation=activate, input_shape=[5]),
            keras.layers.Dense(256, activation=activate),
            keras.layers.Dense(128, activation=activate),
            keras.layers.Dense(64, activation=activate),
            keras.layers.Dense(64, activation=activate), 
            keras.layers.Dense(1)
            ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=32, validation_split=0.2)
    return history.history['loss'], history.history['val_loss'], history.history['accuracy'], history.history['val_accuracy']

def get_csv(path1, path2, path3, path4):
    '''
    input:  Path to X_train, X_test, y_train, y_test
    return: df of our train, test, split
    '''
    X_train = pd.read_csv(path1)
    X_test = pd.read_csv(path2)
    y_train = pd.read_csv(path3)
    y_test = pd.read_csv(path4)

    X_train.drop(['Unnamed: 0'], axis = 1, inplace = True)
    X_test.drop(['Unnamed: 0'], axis = 1, inplace = True)
    y_train.drop(['Unnamed: 0'], axis = 1, inplace = True)
    y_test.drop(['Unnamed: 0'], axis = 1, inplace = True)
    return X_train, X_test, y_train, y_test

def nn_plot(acc, loss, testacc, testloss, details, item):
    '''
    input:  list of test accuracy, test loss, validation accuracy, validation loss, 
            and details about Neural Net to be included in title, and item
    return: graph of test accuracy, test loss, validation accuracy, validation loss
    '''
    fonttitle = {'fontname':'Helvetica', 'fontsize':30}
    fig, ax = plt.subplots(1, 2, figsize = (10,5))
    ax[0].plot(range(1,len(acc) + 1), acc, alpha=0.8, label = 'train', color = '#F6A811', linewidth = 4)
    ax[0].plot(range(1,len(acc) + 1), testacc, alpha=0.8, label = 'validation', color = '#EF727F', linewidth = 4)
    ax[0].set_title('Accuracy')
    ax[1].plot(range(1,len(acc) + 1), loss, alpha=0.8, label = 'train', color = '#F6A811', linewidth = 4)
    ax[1].plot(range(1,len(acc) + 1), testloss, alpha=0.8, label = 'validation', color = '#EF727F', linewidth = 4)                   
    ax[1].set_title('Loss')    
    fig.suptitle(f'Is it {item}? Neural Net Model {len(acc)} Epoch, {details}', fontdict=fonttitle)    
    plt.legend()
    plt.show();

if __name__ == '__main__':
    #---    Train and Test for Organic
    X_train, X_test, y_train, y_test = get_csv("../organicxtrain.csv", "../organicxtest.csv", "../organicytrain.csv", "../organicytest.csv")

    #---    Run neural nets & graph for Organic
    loss1, val_loss1, acc1, val_acc1 = nn_1layer(3, 'relu')
    nn_plot(acc1, loss1, val_acc1, val_loss1, '1 Layer, and ReLU', 'Organic')

    loss2, val_loss2, acc2, val_acc2 = nn_1layer(3, 'swish')
    nn_plot(acc2, loss2, val_acc2, val_loss2, '1 Layer, and Swish', 'Organic')

    loss3, val_loss3, acc3, val_acc3 = nn_2layer(5, 'swish')
    nn_plot(acc3, loss3, valacc3, valloss3, '2 Layers, and Swish', 'Organic')

    loss4, val_loss4, acc4, val_acc4 = nn_5layers(5, 'swish') 
    nn_plot(acc4, loss4, valacc4, valloss4, '5 Layers, and Swish', 'Organic')


    #---    Train and Test for Banana
    X_train, X_test, y_train, y_test = get_csv("../bananaxtrain.csv", "../bananaxtest.csv", "../bananaytrain.csv", "../bananaytest.csv")


    #---    Run neural nets & graph for Banana
    ban_loss1, ban_val_loss1, ban_acc1, ban_val_acc1 = nn_1layer(3, 'relu')
    nn_plot(ban_acc1, ban_loss1, ban_val_acc1, ban_acc1, '1 Layer, and ReLU', 'Banana')

    b_loss2, b_val_loss2, b_acc2, b_val_acc2 = nn_1layer(3, 'swish')
    nn_plot(b_acc2, b_loss2, b_val_acc2, b_val_loss2, '1 Layer, and Swish', 'Banana')

    b_loss3, b_val_loss3, b_acc3, b_val_acc3 = nn_2layer(5, 'swish')
    nn_plot(b_acc3, b_loss3, b_valacc3, b_valloss3, '2 Layers, and Swish', 'Banana')

    b_loss4, b_val_loss4, b_acc4, b_val_acc4 = nn_5layers(5, 'swish') 
    nn_plot(b_acc4, b_loss4, b_valacc4, b_valloss4, '5 Layers, and Swish', 'Banana')
