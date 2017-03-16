import numpy as np
import os
from sklearn import utils as skutils

def make_generator(X, batch_size):
    
    def get_epoch():
        # n_batches = len(X) / batch_size
        for i in range(0, len(X), batch_size):
            yield (X[i:i+batch_size],)
    return get_epoch

def reshape(X):
    return X.reshape(-1, 3, 32, 32)

def _load_batch_cifar10(batch_path, batch_name):
    '''
    load a batch in the CIFAR-10 format
    '''
    path = os.path.join(batch_path, batch_name)
    batch = np.load(path)
    data = batch['data']
    labels = batch['labels']
    return data, labels


def load(batch_size, dtype='int32', data_path="/data/lisatmp3/almahaia/data/processed/cifar10/cifar-10-batches-py"):
    # train
    trX = []
    trY = []
    for k in xrange(5):
        x, t = _load_batch_cifar10(data_path, 'data_batch_{}'.format(k + 1))
        trX.append(x)
        trY.append(t)

    trX = np.concatenate(trX)
    trY = np.concatenate(trY)

    trX, trY = skutils.shuffle(trX, trY,
                                       random_state=np.random.RandomState(12345))
    vaX, vaY = trX[:5000], trY[:5000]
    trX, trY = trX[5000:], trY[5000:]
    
    # test
    teX, teY = _load_batch_cifar10(data_path, 'test_batch')

    trX, vaX, teX = [reshape(X).astype(dtype) for X in trX, vaX, teX]

    # return trX, vaX, teX, trY, vaY, teY
    return make_generator(trX, batch_size), make_generator(vaX, batch_size),\
        make_generator(teX, batch_size)

