import argparse
import sys
import numpy
import time
from dataset import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    figure.set_size_inches(24, 18)
    
    nsamples = X.shape[0]
    nfeatures = X.shape[1]
    nclasses = len(numpy.unique(y))

    epochs=20000 # about 60 s (1 min) for each 1000 epochs
    optimizer = 'Adam'
    init_mode = 'uniform'
    activation = 'tanh'
    neurons = 160
    
    finaltrainacc = []
    finaltestacc = []
    
    numfolds = 10
    fold = 1
    folds_list = numpy.arange(1,numfolds+1)
    kfold = StratifiedKFold(n_splits=numfolds, shuffle=True, random_state=seed)
    colors = iter(cm.gist_rainbow(numpy.linspace(0, 1, numfolds)))
    
    tstart = time.time()
    for train, test in kfold.split(X, y):
        start = time.time()
        
        model = Sequential()
        model.add(Dense(neurons, input_dim=nfeatures, init=init_mode, activation=activation))
        model.add(Dropout(0.5))
        model.add(Dense(neurons/2, input_dim=neurons, init=init_mode, activation=activation))
        model.add(Dropout(0.5))
        model.add(Dense(neurons/4, input_dim=neurons/2, init=init_mode, activation=activation))
        model.add(Dropout(0.5))
        model.add(Dense(neurons/8, input_dim=neurons/4, init=init_mode, activation=activation))
        model.add(Dropout(0.5))
        model.add(Dense(nclasses, init=init_mode, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        print("\nFold %d ..." %(fold))
        trainacc = []
        testacc = []
        trainloss = []
        testloss = []
        for epoch in xrange(epochs):
            history = model.fit(X[train], Y[train], nb_epoch=1, batch_size=nsamples, verbose=0)
            trainacc.append(history.history['acc'][-1])
            trainloss.append(history.history['loss'][-1])
            
            scores = model.evaluate(X[test], Y[test], verbose=0)
            testacc.append(scores[1])
            testloss.append(scores[0])
                    
        end = time.time()
        print("Train Acc: %f" % (trainacc[-1]))
        print("Test Acc: %f" % (testacc[-1]))
        print("Time: %.2fs" %(end - start))

        color=next(colors)
        plt.subplot(221)
        plt.plot(trainacc, label="Train "+str(fold), color=color, linestyle = 'solid')
        plt.plot(testacc, label="Test "+str(fold), color=color, linestyle = 'dashed')
        
        plt.subplot(222)
        plt.plot(trainloss, label="Train "+str(fold), color=color, linestyle = 'solid')
        plt.plot(testloss, label="Test "+str(fold), color=color, linestyle = 'dashed')
        
        finaltrainacc.append(trainacc[-1])
        finaltestacc.append(testacc[-1])
                    
        fold += 1
    tend = time.time()
    print("\nTotal time: %.2fs" %(tend - tstart))
    print("Train Accuracy: %f (+/- %f)" % (numpy.mean(finaltrainacc), numpy.std(finaltrainacc)))
    print("Test Accuracy: %f (+/- %f)" % (numpy.mean(finaltestacc), numpy.std(finaltestacc)))
    
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
    rects1 = plt.bar(folds_list, finaltrainacc, 
                 barwidth, alpha=0.4, color='b',
                 label='Train')
    rects2 = plt.bar(folds_list+barwidth, finaltestacc, 
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
    rects1 = plt.bar(1, numpy.mean(finaltrainacc), 
                 barwidth, alpha=0.4, color='b',
                 yerr=numpy.std(finaltrainacc),
                 label='Train')
    rects2 = plt.bar(1+barwidth, numpy.mean(finaltestacc), 
                 barwidth, alpha=0.4, color='r',
                 yerr=numpy.std(finaltestacc),
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

    plt.savefig("Grid-kfold2-4layers-tanh-dropout2-%depochs.png" %(epochs), dpi=300)
    # plt.legend(optimizer_list, bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gcf().transFigure)
    # plt.show()
  
if __name__ == '__main__':
    parse = argparse.ArgumentParser('gridplot-kfold2-4layers-tanh-dropout.py', description='Multi Layer Perceptron classifier.')
    parse.add_argument('dataset', help='Dataset in pickle format (.pkl).')
    opt = parse.parse_args()
    run(opt.dataset)
