import argparse
import sys
import time
import cPickle

from dataset import Dataset
import numpy as np
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.data_utils import get_file
from keras import backend as K
from tsne import bh_sne

import tensorflow as tf
tf.python.control_flow_ops = tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def VGG16(include_top=True):
    '''Instantiate the VGG16 architecture,
    loading weights pre-trained on ImageNet.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
    # Returns
        A Keras model instance.
    '''
   
    # Set proper input shape
    input_shape = (224, 224, 3)
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

    x = Flatten(name='flatten')(x)
    
    if include_top:
        # Classification block
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    # load weights
    if include_top:
        weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    else:
        weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model.load_weights(weights_path)
    return model

def savetsne(X, y, classes, filename, pca_d=None):
    print("Running t-SNE ...")
    vis_data = bh_sne(np.float64(X), d=2, pca_d=pca_d, perplexity=30., theta=0.5, random_state=1)
    print("t-SNE plot saved in: %s ..." %(filename))
    figure = plt.gcf()
    figure.set_size_inches(24, 18)
    plt.scatter(vis_data[:, 0], vis_data[:, 1], c=y, cmap=plt.cm.get_cmap("Set1", len(classes)))
    plt.clim(-0.5, len(classes)-0.5)
    cbar = plt.colorbar(ticks=range(len(classes)))
    cbar.ax.set_yticklabels(classes)                     
    plt.savefig(filename, dpi=300)

def run(dataset):
    try:
        print("Loading dataset from %s ..." %(dataset))
        X, y, classes, no_imgs = Dataset().load(dataset)
    except:
        print("ERROR: Dataset %s can't be loaded!" %(dataset))
        sys.exit(0)

    print("\nBuilding VGG16 model ...")
    model = VGG16(include_top=False)
    input_shape = (224, 224, 3)
    w, h = X.shape
    Xprocessed = np.empty((w, input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32)
    for channel in range(input_shape[2]):
        Xprocessed[:, :, :, channel] =  X.reshape(w, input_shape[0], input_shape[1]).astype('float32')/255
    vgg16features = model.predict(Xprocessed)
    
    print("\nSaving VGG16 extracted features in file vgg16features.pkl ...")
    vggdataset = (vgg16features, y, classes, no_imgs)
    f = open('vgg16features.pkl', 'wb')
    cPickle.dump(vggdataset, f)
    f.close()

    print("\nBuilding t-SNE plot for VGG16 features ...")
    savetsne(vgg16features, y, classes, "VGG16-Features-TSNE.png")
    
    print("\nBuilding t-SNE plot for VGG16 features with pca_d=50 ...")
    savetsne(vgg16features, y, classes, "VGG16-Features-TSNE_pca50.png", pca_d=50)

    print("\nBuilding t-SNE plot for raw input ...")
    savetsne(X, y, classes, "VGG16-Input-TSNE.png")
  
    print("\nBuilding t-SNE plot for raw input with pca_d=50 ...")
    savetsne(X, y, classes, "VGG16-Input-TSNE_pca50.png", pca_d=50)
  
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('dataset', help='Dataset in pickle format (.pkl).')
    opt = parse.parse_args()
    run(opt.dataset)
