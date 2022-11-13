import os

from sklearn import datasets
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import metrics
import vp

def load_dataset(dataset_name):
    data_dir = os.path.join('data', dataset_name)
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))

    return X, y

#db_name = 'Iris'
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target

#print(X.shape)

X, y = load_dataset('bank')

print(X)

# MA_1 tipo de dado
ma1 = 'tipo'
# MA_2 numero de linhas
ma2 = metrics.metric_dc_num_samples(X)
# MA_3 numero de dimensões
ma3 = metrics.metric_dc_num_features(X)
# MA_4 taxa dimensionalidade intrinseca
ma4 = metrics.metric_dc_intrinsic_dim(X)
# MA_5 razão de disoersão
ma5 = metrics.metric_dc_sparsity_ratio(X)
# MA_6 porcentagem de outiliers
ma6 = 1
# MA_7 correlação media absluta entre atributos continuos
ma7 = 1
# MA_8 Assimetria média de atributos continuos
ma8 = 1
# MA_9 Curtose media dos atributos continuos
ma9 = 1

print('tipo: ' + ma1, ' Linhas: ' + str(ma2), 'dimensoes: ' + str(ma3), 'Dim. intrinseca: ' + str(ma4), 'Razao dispers.: ' + str(ma5))
print('outiliers: ' + str(ma6), 'correlacao: ' + str(ma7),'Assimetria: '+ str(ma8),'curtose: '+ str(ma9))

#print(X, y)

#mm = MinMaxScaler()

#X = mm.fit_transform(X)

