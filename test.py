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
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

from VariationalDense import VariationalDense
from VariationalConv2d import VariationalConv2d
from sklearn.utils import shuffle

#=======================================================================
def create_samples():
    # Define the means and standard deviations for the two Gaussian distributions
    mean1_range = np.arange(0, 10001, 200)
    mean2_range = np.arange(0, 10001, 200)
    std_dev_range = np.arange(10, 1001, 50)
    run = mean1_range.shape[0]*mean2_range.shape[0]*std_dev_range.shape[0]
    sample_size = 500
    count = 0

    # Initialize empty arrays to store the samples and parameters
    samples = np.empty((run, 4), dtype=np.float64)
    para = np.empty((run, 3), dtype=np.float64)

    # Generate samples from each distribution
    for mean1 in mean1_range:
        for mean2 in mean2_range:
            for std_dev in std_dev_range:
                # Generate 250 samples from the first Gaussian distribution
                dist1_samples = np.random.normal(mean1, std_dev, size=int(sample_size/2))
                # Generate 250 samples from the second Gaussian distribution
                dist2_samples = np.random.normal(mean2, std_dev, size=int(sample_size/2))
                # Concatenate the samples from both distributions
                dist_samples = np.concatenate([dist1_samples, dist2_samples])
                # Calculate the moments
                mean = np.mean(dist_samples)
                variance = np.var(dist_samples)
                skewness = np.mean((dist_samples - mean) ** 3) / np.power(np.var(dist_samples), 3/2)
                kurtosis = np.mean((dist_samples - mean) ** 4) / np.power(np.var(dist_samples), 2) - 3
                # Append the samples to the main array
                samples[count] = np.array([mean,variance,skewness,kurtosis])
                para[count] = np.array([mean1, mean2,std_dev])
                count += 1
    
    return samples, para

samples, para = create_samples()

def rw_schedule(epoch):
    if epoch <= 1:
        return 0
    else:
        return 0.0001 * (epoch - 1)


class VariationalLeNet(tf.keras.Model):
    def __init__(self, n_class=4):
        super().__init__()
        self.n_class = n_class

        #self.conv1 = VariationalConv2d((5,5,1,6), stride=1, padding='VALID')
        #self.pooling1 = tf.keras.layers.MaxPooling2D(padding='SAME')
        #self.conv2 = VariationalConv2d((5,5,6,16), stride=1, padding='VALID')
        #self.pooling2 = tf.keras.layers.MaxPooling2D(padding='SAME')
        #self.in1 = Input(shape=(3,))
        self.d1 = tf.keras.layers.Dense(100, kernel_initializer='uniform', activation='relu')
        self.fc1 = VariationalDense(100)
        self.fc2 = VariationalDense(100)
        self.fc3 = VariationalDense(100)
        self.d2 = tf.keras.layers.Dense(n_class, kernel_initializer='uniform')

        #self.hidden_layer = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]
        self.hidden_layer = [self.fc1, self.fc2, self.fc3]

    @tf.function
    def call(self, x, sparse=False):
        #x = self.conv1(x, sparse)
        #x = tf.nn.relu(x)
        #x = self.pooling1(x)
        #x = self.conv2(x, sparse)
        #x = tf.nn.relu(x)
        #x = self.pooling2(x)
        #x = self.flat(x)
        #x = self.in1(x)
        x = self.d1(x)
        x = self.fc1(x, sparse)
        x = tf.nn.relu(x)
        x = self.fc2(x, sparse)
        x = tf.nn.relu(x)
        x = self.fc3(x, sparse)
        x = tf.nn.relu(x)

        return self.d2(x)

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
    def aleatoric_loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred,dtype=tf.float32)
        se = K.pow((y_true[:,:4]-y_pred[:,:4]),2)
        inv_std = K.exp(-y_pred[:,:4])
        mse = K.mean(K.batch_dot(inv_std,se))
        reg = K.mean(y_pred[:,:4])
        return 0.5*(mse + reg)

    @tf.function
    def compute_loss(label, pred, reg):
        return aleatoric_loss(label, pred) + reg


    @tf.function
    def compute_loss2(label, pred):
        return aleatoric_loss(label, pred)

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

    '''
    Load data
    '''
    mnist = tf.keras.datasets.mnist

    sample, para = create_samples()

    # split into train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(para, sample, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15)

    # scale and standardise
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    x_val = sc.transform(x_val)

    scy = StandardScaler()
    y_train = scy.fit_transform(y_train)
    y_test = scy.transform(y_test)
    y_val = scy.transform(y_val)

    #nr = np.zeros(len(y_train))
    #y_train = np.column_stack((y_train,nr, nr, nr, nr))
    #nr = np.zeros(len(y_val))
    #y_val = np.column_stack((y_val,nr,nr,nr,nr))

    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    #x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    #x_test = (x_test.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    #y_train = np.eye(10)[y_train].astype(np.float32)
    #y_test = np.eye(10)[y_test].astype(np.float32)

    '''
    Build model
    '''
    model = VariationalLeNet()
    criterion = tf.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()

    '''
    Train model
    '''
    epochs = 20
    batch_size = 100
    n_batches = x_train.shape[0] // batch_size

    train_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.MeanSquaredError()
    test_loss = tf.keras.metrics.Mean()
    test_acc = tf.keras.metrics.MeanSquaredError()

    for epoch in range(epochs):

        _x_train, _y_train = shuffle(x_train, y_train, random_state=42)

        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            train_step(_x_train[start:end], _y_train[start:end], epoch)

        if epoch % 1 == 0 or epoch == epochs - 1:
            preds = test_step(x_test, y_test)
            print('Epoch: {}, Valid Cost: {:.3f}, Valid Acc: {:.3f}'.format(
                epoch+1,
                test_loss.result(),
                test_acc.result()
            ))
            print("Sparsity: ", model.count_sparsity())

y_pred =  scy.inverse_transform(model(x_val,sparse=True))
#y_pred = model.predict(X_val)[:,:4]
y_val =  scy.inverse_transform(y_val[:,:4])

for i in range(4):
    plt.figure(i+2)
    plt.plot(y_val[:,i],y_val[:,i],'r.')
    plt.plot(y_val[:,i],y_pred[:,i],'ko',alpha=0.4)
    plt.figure(7)
    y = abs(y_pred[:,i] - y_val[:,i])/np.max(abs(y_pred[:,i] - y_val[:,i]))
    plt.plot(y_val[:,i]/np.max(y_val[:,i]),y,'o')
plt.show()

print(r2_score(y_val, y_pred[:,:4]))

y_pred_test = scy.inverse_transform(model(x_test,sparse=True))
y_test = scy.inverse_transform(y_test)
    
for i in range(4):
    plt.figure(i+2)
    plt.plot(y_test[:,i],y_test[:,i],'r.')
    plt.plot(y_test[:,i],y_pred_test[:,i],'ko',alpha=0.4)
    plt.figure(7)
    y = abs(y_pred_test[:,i] - y_test[:,i])/np.max(abs(y_pred_test[:,i] - y_test[:,i]))
    plt.plot(y_test[:,i]/np.max(y_test[:,i]),y,'o')
plt.show()

print(r2_score(y_test, y_pred_test))
plt.show()


