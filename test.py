from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
#import scikeras
#from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics
import numpy as np
from sklearn.metrics import r2_score
import tensorflow.keras
import elfi
import pygtc
import pickle
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf

print(tf.__version__)

import numpy as np
import emcee
import pygtc
import pickle
import matplotlib.pyplot as plt
import time
import tensorflow
from multiprocessing import Pool
from multiprocessing import cpu_count
import os
import sys
import shutil
import random
from glob import glob
import time

import numpy as np
import matplotlib.pyplot as plt

# Generate random data for demonstration
np.random.seed(1)
y_pred = np.random.randn(100)
y_val = y_pred + np.random.randn(100) * 0.1

# Calculate normalized absolute difference
y = np.abs(y_pred - y_val) / np.max(np.abs(y_pred - y_val))

# Plot the graph
plt.plot(y_val / np.max(y_val), y, 'o')
plt.xlabel('Normalized y_val')
plt.ylabel('Normalized Absolute Difference')
plt.title('Comparison between y_pred and y_val')
plt.show()
