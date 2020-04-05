
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.datasets import cifar10
from keras.engine import training
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average
from keras.losses import categorical_crossentropy
from keras.models import Model, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.python.framework.ops import Tensor
from typing import Tuple, List
import glob
import numpy as np
import os
import argparse
import numpy as np
from scipy import sparse
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from scipy.spatial import distance
from sklearn.linear_model import RidgeClassifier
import sklearn.metrics
import scipy.io as sio

import pgraph as Reader
import train_pGCN as Train

#example ensemble of 3 pGCNs. Can be extended to 

train_pGCN1_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'train_pGCN1_weights.hdf5')
train_pGCN2_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'train_pGCN2_pretrained_weights.hdf5')
train_pGCN3_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'train_pGCN3_pretrained_weights.hdf5')

train_pGCN1_model = train_pGCN1(model_input)
train_pGCN2_model = train_pGCN2(model_input)
train_pGCN3_model = train_pGCN3(model_input)

train_pGCN1_model.load_weights(train_pGCN1_WEIGHT_FILE)
train_pGCN2_model.load_weights(train_pGCN2_WEIGHT_FILE)
train_pGCN3_model.load_weights(train_pGCN3_WEIGHT_FILE)

models = [train_pGCN1_model, train_pGCN2_model, train_pGCN3_model]



def ensemble(models: List [training.Model], model_input: Tensor) -> training.Model:
    
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    
    model = Model(model_input, y, name='ensemble')
    
    return model