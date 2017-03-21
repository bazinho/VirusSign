# Use scikit-learn to grid search the weight initialization
import argparse
import sys
import numpy
from dataset import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


def create_model(nfeatures, nclasses, init_mode='uniform', optimizer='adam'):
    model = Sequential()
    model.add(Dense(160, input_dim=nfeatures, init=init_mode, activation='relu'))
    model.add(Dense(nclasses, init=init_mode, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

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
    
    # create model
    model = KerasClassifier(build_fn=create_model, nfeatures=X.shape[1], nclasses=len(numpy.unique(y)), nb_epoch=100, verbose=1)
    
    seed = 1
    numpy.random.seed(seed)
    #kfold = KFold(n_folds=10, shuffle=True, random_state=seed)
    #results = cross_val_score(estimator, X, Y, cv=kfold)
    #print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    # define the grid search parameters
    #init_mode = ['uniform', 'lecun_uniform', 'normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    #param_grid = dict(init_mode=init_mode)
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(X, Y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.grid_scores_:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
  
if __name__ == '__main__':
    parse = argparse.ArgumentParser('grid-weight.py', description='Multi Layer Perceptron classifier with rectified linear units.')
    parse.add_argument('dataset', help='Dataset in pickle format (.pkl).')
    opt = parse.parse_args()
    run(opt.dataset)
