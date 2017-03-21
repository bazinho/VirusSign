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

    epochs=10000 # about 60 s (1 min) for each 1000 epochs
    optimizer = 'Adam'
    init_mode = 'uniform'
    activation = 'relu'
    neurons = 160
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    trainscores = []
    testscores = []
    fold = 1
    folds_list = numpy.arange(1,11)
    tstart = time.time()
    for train, test in kfold.split(X, y):
        start = time.time()
        
        model = Sequential()
        model.add(Dense(neurons, input_dim=nfeatures, init=init_mode, activation=activation))
        model.add(Dense(nclasses, init=init_mode, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        print("\nTraining fold %d ..." %(fold))
        history = model.fit(X[train], Y[train], nb_epoch=epochs, batch_size=nsamples, verbose=0)
	print("Max Accuracy: %f" % (numpy.max(history.history['acc'])))
	print("Min Loss: %f" % (numpy.min(history.history['loss'])))
	print("Accuracy: %f" % (history.history['acc'][-1]))
	print("Loss: %f" % (history.history['loss'][-1]))
        trainscores.append(history.history['acc'][-1])

        plt.subplot(221)
        plt.plot(history.history['acc'], label=fold, linewidth=2.0)
        plt.subplot(222)
        plt.plot(history.history['loss'], label=fold, linewidth=2.0)
        
        print("Testing fold %d ..." %(fold))
        scores = model.evaluate(X[test], Y[test], verbose=0)
	print("Accuracy: %f" % (scores[1]))
	print("Loss: %f" % (scores[0]))
        testscores.append(scores[1])
        
        end = time.time()
        print("Time: %.2fs" %(end - start))
        
        fold += 1
    tend = time.time()
    print("\nTotal time: %.2fs" %(tend - tstart))
    print("Train Accuracy: %f (+/- %f)" % (numpy.mean(trainscores), numpy.std(trainscores)))
    print("Test Accuracy: %f (+/- %f)" % (numpy.mean(testscores), numpy.std(testscores)))
    
    rcParams['legend.loc'] = 'best'
    plt.subplot(221)
    plt.title('Train Folds Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0.0,1.0))
    plt.legend()
    plt.grid(True)
    
    plt.subplot(222)      
    plt.title('Train Folds Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    barwidth = 0.35
    plt.subplot(223)
    rects1 = plt.bar(folds_list, trainscores, 
                 barwidth, alpha=0.4, color='b',
                 label='Train')
    rects2 = plt.bar(folds_list+barwidth, testscores, 
                 barwidth, alpha=0.4, color='r',
                 label='Test')
    plt.title('Train/Test Folds Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Fold')
    plt.xticks(folds_list+barwidth, folds_list)
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0.0,1.0))
    plt.legend()
    plt.grid(True)
 
    plt.subplot(224)      
    rects1 = plt.bar(1, numpy.mean(trainscores), 
                 barwidth, alpha=0.4, color='b',
                 yerr=numpy.std(trainscores),
                 label='Train')
    rects2 = plt.bar(1+barwidth, numpy.mean(testscores), 
                 barwidth, alpha=0.4, color='r',
                 yerr=numpy.std(testscores),
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
    plt.grid(True)

    plt.savefig("Grid-kfold-%depochs.png" %(epochs), dpi=100)
    # plt.legend(optimizer_list, bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gcf().transFigure)
    # plt.show()
  
if __name__ == '__main__':
    parse = argparse.ArgumentParser('gridplot-kfold.py', description='Multi Layer Perceptron classifier.')
    parse.add_argument('dataset', help='Dataset in pickle format (.pkl).')
    opt = parse.parse_args()
    run(opt.dataset)
