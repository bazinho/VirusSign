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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def run(dataset,epochs=10):
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
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 224, 224)
    else:
        input_shape = (224, 224, 1)
        
    img_input = Input(shape=input_shape)
    
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(nclasses, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    
    #tensorboard = TensorBoard(log_dir="tensorboard-vgg16", histogram_freq=1, write_graph=False)
    #history = model.fit(X[train], Y[train], validation_data=(X[test], Y[test]), nb_epoch=epochs, batch_size=batch_size, verbose=1, callbacks=[tensorboard])
    #history = model.fit(X[train].reshape(len(train), input_shape[0], input_shape[1], input_shape[2]).astype('float32')/255, Y[train], validation_data=(X[test].reshape(len(test), input_shape[0], input_shape[1], input_shape[2]).astype('float32')/255, Y[test]), nb_epoch=epochs, batch_size=batch_size, verbose=1, callbacks=[tensorboard])
    history = model.fit(X[train].reshape(len(train), input_shape[0], input_shape[1], input_shape[2]).astype('float32')/255, Y[train], validation_data=(X[test].reshape(len(test), input_shape[0], input_shape[1], input_shape[2]).astype('float32')/255, Y[test]), nb_epoch=epochs, batch_size=batch_size, verbose=1)
    
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

    plt.savefig("VGG16-%depochs.png" %(epochs), dpi=300)
  
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('dataset', help='Dataset in pickle format (.pkl).')
    opt = parse.parse_args()
    run(opt.dataset)
