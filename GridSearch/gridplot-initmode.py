import argparse
import sys
import numpy
import time
from dataset import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams

def run(dataset):
    # load dataset
    try:
        print("Loading dataset from %s ..." %(dataset))
        X, y, list_fams, no_imgs = Dataset().load(dataset)
    except:
        print("ERROR: File %s not found!" %(dataset))
        sys.exit(0)

    # Create one-hot-encode version of y  
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = np_utils.to_categorical(y)
    
    seed = 1
    numpy.random.seed(seed)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 12)
    
    nsamples = X.shape[0]
    nfeatures = X.shape[1]
    nclasses = len(numpy.unique(y))
    
    epochs=10000  # about 60 s (1 min) for each 1000 epochs
    activation = 'relu'
    optimizer = 'Adam'
    init_mode_list = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    tstart = time.time()
    for init_mode in init_mode_list:
        print("\nInit Mode: %s" %(init_mode))
        start = time.time()
        model = Sequential()
        model.add(Dense(160, input_dim=nfeatures, init=init_mode, activation=activation))
        model.add(Dense(nclasses, init=init_mode, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        history = model.fit(X, Y, nb_epoch=epochs, batch_size=nsamples, verbose=0)
        end = time.time()
        print("Time: %.2fs" %(end - start))
        plt.subplot(211)
        plt.plot(history.history['acc'], label=init_mode, linewidth=2.0)
        plt.subplot(212)
        plt.plot(history.history['loss'], label=init_mode, linewidth=2.0)
    tend = time.time()
    print("Total time: %.2fs" %(tend - tstart))
    
    rcParams['legend.loc'] = 'best'
    plt.subplot(211)
    plt.title('Init Mode Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(init_mode_list)
    plt.grid(True)
    
    plt.subplot(212)      
    plt.title('Init Mode Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(init_mode_list)
    plt.grid(True)
 
    plt.savefig("Grid-InitMode-%depochs.png" %(epochs), dpi=100)
    # plt.legend(optimizer_list, bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gcf().transFigure)
    # plt.show()
  
if __name__ == '__main__':
    parse = argparse.ArgumentParser('gridplot-initmode.py', description='Multi Layer Perceptron classifier.')
    parse.add_argument('dataset', help='Dataset in pickle format (.pkl).')
    opt = parse.parse_args()
    run(opt.dataset)
