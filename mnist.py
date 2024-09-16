import os.path
import pickle
import gzip
from urllib.parse import urljoin
import urllib.request as req
import numpy as np

def load():
    url = 'https://github.com/mnielsen/rmnist/raw/master/data/'
    filename = 'mnist.pkl.gz'
    diretorio_download = '.'
    arquivo = os.path.join(diretorio_download, filename)

    if not os.path.isfile(arquivo):
        url = urljoin(url, filename)
        with req.urlopen(url) as d, open(arquivo, "wb") as opfile:
            data = d.read()
            opfile.write(data)

    f = gzip.open(arquivo, 'rb')
    tr_d, _, te_d = pickle.load(f, encoding="latin1")
    f.close()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training = zip(training_inputs, training_results)

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test = zip(test_inputs, te_d[1])

    return training, test

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e