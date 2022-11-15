# -*- coding: utf-8 -*-
 
import argparse
import gzip
import os
import shutil
import tarfile
import tempfile
import zipfile
from glob import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import wget
from keras import datasets as kdatasets
from keras import applications
from scipy.io import arff
#from skimage import io, transform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import metrics

def download_file(urls, base_dir, name):
    dir_name = os.path.join(base_dir, name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

        for url in urls:
            wget.download(url, out=dir_name)


def save_dataset(name, X, y):
    n_samples = metrics.metric_dc_num_samples(X)
    n_features = metrics.metric_dc_num_features(X)
    n_classes = metrics.metric_dc_num_classes(y)
    balanced = metrics.metric_dc_dataset_is_balanced(y)
    dim_int = metrics.metric_dc_intrinsic_dim(X)

    print(name, n_samples, n_features, n_classes, balanced, dim_int, X.shape)

    for l in np.unique(y):
        print('-->', l, np.count_nonzero(y == l))

    dir_name = os.path.join(base_dir, name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.astype('float32'))

    np.save(os.path.join(dir_name, 'X.npy'), X)
    np.save(os.path.join(dir_name, 'y.npy'), y)

    np.savetxt(os.path.join(dir_name, 'X.csv.gz'), X, delimiter=',')
    np.savetxt(os.path.join(dir_name, 'y.csv.gz'), y, delimiter=',')


def remove_all_datasets(base_dir):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

# Banco de dados bank
def process_bank():
    bank = zipfile.ZipFile('data/bank/bank-additional.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    bank.extractall(tmp_dir.name)

    df = pd.read_csv(os.path.join(
        tmp_dir.name, 'bank-additional', 'bank-additional-full.csv'), sep=';')

    y = np.array(df['y'] == 'yes').astype('uint8')
    X = np.array(pd.get_dummies(df.drop('y', axis=1)))

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.05, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('bank', X, y)

def process_cnae9():
    df = pd.read_csv('data/cnae9/CNAE-9.data', header=None)
    y = np.array(df[0])
    X = np.array(df.drop(0, axis=1))
    save_dataset('cnae9', X, y)


# código
if __name__ == '__main__':
    base_dir = './data'

    datasets = dict()

    datasets['bank'] = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip']
    datasets['cnae9'] = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/00233/CNAE-9.data']
    

    parser = argparse.ArgumentParser(
        description='Projection Survey Dataset Downloader')

    parser.add_argument('-d', action='store_true', help='delete all datasets')
    parser.add_argument('-s', action='store_true',
                        help='skip download, assume files are in place')
    args, unknown = parser.parse_known_args()

    if args.d:
        print('Removing all datasets')
        remove_all_datasets(base_dir)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    if not args.s:
        print('Downloading all datasets')
        for name, url in datasets.items():
            print('')
            print(name)
            download_file(url, base_dir, name)

    print('')
    print('Processing all datasets')

    for func in sorted([f for f in dir() if f[:8] == 'process_']):
        print(str(func))
        globals()[func]()

