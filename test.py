from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from VariationalDense import VariationalDense
#from VariationalConv2d import VariationalConv2d
from sklearn.utils import shuffle

def rw_schedule(epoch):
    if epoch <= 1:
        return 0
    else:
        return 0.0001 * (epoch - 1)

class VariationalLeNet(tf.keras.Model):
    def __init__(self, n_class=10):
        super().__init__()
        self.n_class = n_class

        #self.conv1 = VariationalConv2d((5,5,1,6), stride=1, padding='VALID')
        #self.pooling1 = tf.keras.layers.MaxPooling2D(padding='SAME')
        #self.conv2 = VariationalConv2d((5,5,6,16), stride=1, padding='VALID')
        #self.pooling2 = tf.keras.layers.MaxPooling2D(padding='SAME')
        
        self.d0 = Dense(3)
        self.d1 = Dense(100, kernel_initializer='uniform', activation='relu')
        self.d2 = Dense(100, kernel_initializer='uniform', activation='relu')
        self.d3 = Dense(100, kernel_initializer='uniform', activation='relu')
        self.d4 = Dense(100, kernel_initializer='uniform', activation='relu')
        self.d5 = Dense(8,kernel_initializer='uniform')

        #self.flat = tf.keras.layers.Flatten()
        self.fc1 = VariationalDense(120)
        self.fc2 = VariationalDense(84)
        self.fc3 = VariationalDense(10)

        self.hidden_layer = [self.fc1, self.fc2, self.fc3]

    @tf.function
    def call(self, x, sparse=False):
        x = self.d0(x)
        #x = tf.nn.relu(x)
        #x = self.pooling1(x)
        x = self.d1(x)
        x = self.d2(x)
        #x = tf.nn.relu(x)
        #x = self.pooling2(x)
        #x = self.flat(x)
        x = self.fc1(x, sparse)
        x = self.d3(x)
        #x = tf.nn.relu(x)
        x = self.fc2(x, sparse)
        x = self.d4(x)
        #x = tf.nn.relu(x)
        x = self.fc3(x, sparse)
        x = self.d5(x)

        return x

    def regularization(self):
        total_reg = 0
        for layer in self.hidden_layer:
            total_reg += layer.regularization

        return total_reg

    def count_sparsity(self):
        total_remain, total_param = 0, 0
        for layer in self.hidden_layer:
            a, b = layer.sparsity()
            total_remain += a
            total_param += b

        return 1 - (total_remain/total_param)



if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)
    tf.config.set_visible_devices([], 'GPU')
    @tf.function
    def compute_loss(label, pred, reg):
        return criterion(label, pred) + reg


    @tf.function
    def compute_loss2(label, pred):
        return criterion(label, pred)

    def train_step(x, t, epoch):
        with tf.GradientTape() as tape:
            preds = model(x)
            reg = rw_schedule(epoch) * model.regularization()
            loss = compute_loss(t, preds, reg)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)
        train_acc(t, preds)

        return preds

    @tf.function
    def test_step(x, t):
        preds = model(x, sparse=True)
        loss = compute_loss2(t, preds)
        test_loss(loss)
        test_acc(t, preds)

        return preds

model = VariationalLeNet()
input_shape=(3,)
model.build(input_shape)

opt = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-09,)
model.compile(loss='mean_square_error', optimizer=opt, metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
print(model.summary())

history = model.fit(X_train, y_train, batch_size=int(len(X_train)/3), epochs = epochs, shuffle=True, 
                    validation_data=(X_val, y_val), use_multiprocessing=True, callbacks=[es])


