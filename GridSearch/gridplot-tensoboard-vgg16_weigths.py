import argparse
import sys
import time
import numpy as np

from dataset import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import ZeroPadding2D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras import backend as K
import tensorflow as tf
tf.python.control_flow_ops = tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def run(dataset,epochs=100):
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
    batch_size = 10
    
    numfolds = 2
    kfold = StratifiedKFold(n_splits=numfolds, shuffle=True, random_state=seed)
    trainfolds, testfolds  = kfold.split(X, y)
    train = trainfolds[0]
    test = testfolds[0]
    
    tstart = time.time()

    print("\nBuilding VGG16 model ...")
    
    # Determine proper input shape
    input_shape = (224, 224, 3)
        
    model = Sequential()
    
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5')

    model.layers.pop()

    model.add(Dense(nclasses, activation='softmax'))

    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=['accuracy'])
    model.compile(optimizer='Nadam', loss='categorical_crossentropy',  metrics=['accuracy'])
    
    w, h = X[train].shape
    Xtrain = np.empty((w, input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32)
    for channel in range(input_shape[2]):
        Xtrain[:, :, :, channel] =  X[train].reshape(w, input_shape[0], input_shape[1]).astype('float32')/255
    Ytrain = Y[train]

    w, h = X[test].shape
    Xtest = np.empty((w, input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32)
    for channel in range(input_shape[2]):
        Xtest[:, :, :, channel] =  X[test].reshape(w, input_shape[0], input_shape[1]).astype('float32')/255
    Ytest = Y[test]

    history = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), nb_epoch=epochs, batch_size=batch_size, verbose=1)
    
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

    plt.savefig("VGG16_weigths-%depochs.png" %(epochs), dpi=300)
  
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('dataset', help='Dataset in pickle format (.pkl).')
    opt = parse.parse_args()
    run(opt.dataset)
