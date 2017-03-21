import argparse
import sys
import numpy
import time

from dataset import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import TensorBoard

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def run(dataset,neurons=160,activation='tanh',init_mode = 'uniform',optimizer='Adam',epochs=20000):
    try:
        print("Loading dataset from %s ..." %(dataset))
        X, y, list_fams, no_imgs = Dataset().load(dataset)
    except:
        print("ERROR: File %s not found!" %(dataset))
        sys.exit(0)

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = np_utils.to_categorical(y)
    
    seed = 1
    numpy.random.seed(seed)
    figure = plt.gcf()
    figure.set_size_inches(24, 18)
    
    nsamples = X.shape[0]
    nfeatures = X.shape[1]
    nclasses = len(numpy.unique(y))
    batch_size = nsamples
    
    numfolds = 2
    kfold = StratifiedKFold(n_splits=numfolds, shuffle=True, random_state=seed)
    trainfolds, testfolds  = kfold.split(X, y)
    train = trainfolds[0]
    test = testfolds[0]
    
    optimizer_list = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    colors = iter(cm.gist_rainbow(numpy.linspace(0, 1, len(optimizer_list))))
    
    tstart = time.time()
    for optimizer in optimizer_list:
        start = time.time()
        print("\nOptimizer: %s" %(optimizer))
        
        model = Sequential()
        model.add(Dense(neurons, input_dim=nfeatures, init=init_mode, activation=activation))
        #model.add(Dropout(0.5))
        model.add(Dense(neurons/2, init=init_mode, activation=activation))
        #model.add(Dropout(0.5))
        model.add(Dense(neurons/4, init=init_mode, activation=activation))
        #model.add(Dropout(0.5))
        model.add(Dense(neurons/8, init=init_mode, activation=activation))
        #model.add(Dropout(0.5))
        model.add(Dense(nclasses, init=init_mode, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        #tensorboard = TensorBoard(log_dir="tensorboard-%s" %(optimizer), histogram_freq=1, write_graph=False)
        #history = model.fit(X[train], Y[train], validation_data=(X[test], Y[test]), nb_epoch=epochs, batch_size=batch_size, verbose=1, callbacks=[tensorboard])
        history = model.fit(X[train], Y[train], validation_data=(X[test], Y[test]), nb_epoch=epochs, batch_size=batch_size, verbose=0)
        
        end = time.time()
        print("Train Acc: %f" % (history.history['acc'][-1]))
        print("Test Acc: %f" % (history.history['val_acc'][-1]))
        print("Time: %.2fs" %(end - start))

        color=next(colors)
        plt.subplot(211)
        plt.plot(history.history['acc'], label="Train "+optimizer, color=color, linestyle = 'solid')
        plt.plot(history.history['val_acc'], label="Test "+optimizer, color=color, linestyle = 'dashed')
        
        plt.subplot(212)
        plt.plot(history.history['loss'], label="Train "+optimizer, color=color, linestyle = 'solid')
        plt.plot(history.history['val_loss'], label="Test "+optimizer, color=color, linestyle = 'dashed')
                           
    tend = time.time()
    print("\nTotal time: %.2fs" %(tend - tstart))
    
    ax = plt.subplot(211)
    plt.title('Optimizer Accuracy')
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
    plt.title('Optimizer Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)

    plt.savefig("Optimizers-4layers-%dneurons-%s-%depochs.png" %(neurons,activation,epochs), dpi=300)
  
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('dataset', help='Dataset in pickle format (.pkl).')
    opt = parse.parse_args()
    run(opt.dataset)
