import numpy as np
import pickle
import os



classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)
def load_pickle(f):
    return  pickle.load(f, encoding='latin1')

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return x_train, y_train, Xte, Yte
def get_CIFAR10_data(cl0 = 0, cl1 = 6, num_training=10000, num_test=2000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'data/'
    x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir)
    mask = (y_train==cl0) | (y_train==cl1)
    x_train = x_train[mask].astype("float32")
    y_train = y_train[mask]
    y_train[y_train==cl0] = -1
    y_train[y_train==cl1] = 1
    y_train=y_train.astype(np.double)
    indexes = np.arange(x_train.shape[0])
    np.random.shuffle(indexes)
    x_train = x_train[indexes]
    y_train = y_train[indexes]
    
    mask = (y_test==cl0) | (y_test==cl1)
    x_test = x_test[mask].astype("float32")
    y_test = y_test[mask]
    y_test[y_test==cl0] = -1
    y_test[y_test==cl1] = 1
    y_test=y_test.astype(np.double)

    x_train /= 255
    x_test /= 255

    return x_train[:num_training], y_train[:num_training], x_test[:num_test], y_test[:num_test] 

