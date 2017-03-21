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

def run(dataset,neurons=160,activation='tanh',init_mode = 'uniform',optimizer='Adam',epochs=50000):
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
    
    trainacc = []
    testacc = []
    
    numfolds = 5
    fold = 1
    folds_list = numpy.arange(1,numfolds+1)
    kfold = StratifiedKFold(n_splits=numfolds, shuffle=True, random_state=seed)
    colors = iter(cm.gist_rainbow(numpy.linspace(0, 1, numfolds)))
    
    tstart = time.time()
    for train, test in kfold.split(X, y):
        start = time.time()
        print("\nFold %d ..." %(fold))
        
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
        
        tensorboard = TensorBoard(log_dir="tensorboard-tanh", histogram_freq=1, write_graph=True)
        history = model.fit(X[train], Y[train], validation_data=(X[test], Y[test]), nb_epoch=epochs, batch_size=batch_size, verbose=1, callbacks=[tensorboard])
        #history = model.fit(X[train], Y[train], validation_data=(X[test], Y[test]), nb_epoch=epochs, batch_size=batch_size, verbose=0)
        
        end = time.time()
        print("Train Acc: %f" % (history.history['acc'][-1]))
        print("Test Acc: %f" % (history.history['val_acc'][-1]))
        print("Time: %.2fs" %(end - start))

        color=next(colors)
        plt.subplot(221)
        plt.plot(history.history['acc'], label="Train "+str(fold), color=color, linestyle = 'solid')
        plt.plot(history.history['val_acc'], label="Test "+str(fold), color=color, linestyle = 'dashed')
        
        plt.subplot(222)
        plt.plot(history.history['loss'], label="Train "+str(fold), color=color, linestyle = 'solid')
        plt.plot(history.history['val_loss'], label="Test "+str(fold), color=color, linestyle = 'dashed')
        
        trainacc.append(history.history['acc'][-1])
        testacc.append(history.history['val_acc'][-1])
                    
        fold += 1
    tend = time.time()
    print("\nTotal time: %.2fs" %(tend - tstart))
    print("Train Accuracy: %f (+/- %f)" % (numpy.mean(trainacc), numpy.std(trainacc)))
    print("Test Accuracy: %f (+/- %f)" % (numpy.mean(testacc), numpy.std(testacc)))
    
    ax = plt.subplot(221)
    plt.title('Train Folds Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0.0,1.0))
    plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    
    ax = plt.subplot(222)      
    plt.title('Train Folds Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)

    barwidth = 0.35
    ax = plt.subplot(223)
    rects1 = plt.bar(folds_list, trainacc, 
                 barwidth, alpha=0.4, color='b',
                 label='Train')
    rects2 = plt.bar(folds_list+barwidth, testacc, 
                 barwidth, alpha=0.4, color='r',
                 label='Test')
    plt.title('Train/Test Folds Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Fold')
    plt.xticks(folds_list+barwidth, folds_list)
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0.0,1.0))
    plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
 
    ax = plt.subplot(224)      
    rects1 = plt.bar(1, numpy.mean(trainacc), 
                 barwidth, alpha=0.4, color='b',
                 yerr=numpy.std(trainacc),
                 label='Train')
    rects2 = plt.bar(1+barwidth, numpy.mean(testacc), 
                 barwidth, alpha=0.4, color='r',
                 yerr=numpy.std(testacc),
                 label='Test')
    plt.tick_params(axis='x',          # changes apply to the x-axis
                 which='both',      # both major and minor ticks are affected
                 bottom='off',      # ticks along the bottom edge are off
                 top='off',         # ticks along the top edge are off
                 labelbottom='off') # labels along the bottom edge are off
    plt.title('Train/Test Mean Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Train/Test')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0.0,1.0))
    plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)

    plt.savefig("Grid-tensorboard-%depochs.png" %(epochs), dpi=300)
    # plt.legend(optimizer_list, bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gcf().transFigure)
    # plt.show()
  
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('dataset', help='Dataset in pickle format (.pkl).')
    opt = parse.parse_args()
    run(opt.dataset)
