import argparse
import sys
import time
import numpy as np

from dataset import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras import backend as K
import tensorflow as tf
tf.python.control_flow_ops = tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def run(dataset,neurons=128,activation='relu',init_mode = 'uniform',optimizer='Nadam',epochs=20000):
    try:
        print("Loading dataset from %s ..." %(dataset))
        lstart = time.time()
        X, y, list_fams, no_imgs = Dataset().load(dataset)
        lend = time.time()
        print("\nLoading time: %.2fs" %(lend - lstart))
    except:
        print("ERROR: File %s not found!" %(dataset))
        sys.exit(0)

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = np_utils.to_categorical(y)
    
    seed = 1
    np.random.seed(seed)
    figure = plt.gcf()
    figure.set_size_inches(24, 18)
    
    nsamples = X.shape[0]
    nfeatures = X.shape[1]
    nclasses = len(np.unique(y))
    batch_size = nsamples
    
    numfolds = 2
    kfold = StratifiedKFold(n_splits=numfolds, shuffle=True, random_state=seed)
    trainfolds, testfolds  = kfold.split(X, y)
    train = trainfolds[0]
    test = testfolds[0]
    
    tstart = time.time()

    print("\nBuilding Convnet (MNIST) model ...")
    
    model = Sequential()
    
    model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(28, 28, 1), init=init_mode, activation=activation))
    model.add(Convolution2D(32, 5, 5, border_mode='valid', init=init_mode, activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(neurons, init=init_mode, activation=activation))
    model.add(Dropout(0.5))
    
    model.add(Dense(nclasses, init=init_mode, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    print("\nTraining Convnet (MNIST) ...")
    tensorboard = TensorBoard(log_dir="tensorboard-mnist", histogram_freq=1, write_graph=False)
    #history = model.fit(X[train], Y[train], validation_data=(X[test], Y[test]), nb_epoch=epochs, batch_size=batch_size, verbose=1, callbacks=[tensorboard])
    history = model.fit(X[train].reshape(len(train), 28, 28, 1).astype('float32')/255, Y[train], validation_data=(X[test].reshape(len(test), 28, 28, 1).astype('float32')/255, Y[test]), nb_epoch=epochs, batch_size=batch_size, verbose=1, callbacks=[tensorboard])

    print("Train Acc: %f" % (history.history['acc'][-1]))
    print("Test Acc: %f" % (history.history['val_acc'][-1]))

    plt.subplot(211)
    plt.plot(history.history['acc'], label="Train", linestyle = 'solid')
    plt.plot(history.history['val_acc'], label="Test", linestyle = 'dashed')
    
    plt.subplot(212)
    plt.plot(history.history['loss'], label="Train", linestyle = 'solid')
    plt.plot(history.history['val_loss'], label="Test", linestyle = 'dashed')
                        
    tend = time.time()
    print("\nTotal time: %.2fs" %(tend - tstart))
    
    ax = plt.subplot(211)
    plt.title('Layers Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0.0,1.0))
    plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    
    ax = plt.subplot(212)      
    plt.title('Layers Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)

    plt.savefig("MNIST-%depochs.png" %(epochs), dpi=300)
  
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('dataset', help='Dataset in pickle format (.pkl).')
    opt = parse.parse_args()
    run(opt.dataset)
