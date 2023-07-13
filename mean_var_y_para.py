# mlp for bimodal distribution
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.spatial import distance
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Input,Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Define the means and standard deviations for the two Gaussian distributions
mean1_range = np.arange(10, 1001, 10)
mean2_range = np.arange(10, 1001, 10)
std_dev_range = np.arange(10, 101, 10)
run = mean1_range.shape[0]*mean2_range.shape[0]*std_dev_range.shape[0]
sample_size = 500
count = 0

# Initialize empty arrays to store the samples and parameters
samples = np.empty((run, sample_size), dtype=np.float64)
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
            # Append the samples to the main array
            samples[count] = dist_samples
            para[count] = np.array([mean1, mean2,std_dev])
            count += 1

print(samples[1])
print(para[0])

def aleatoric_loss(y_true, y_pred):
    se = K.pow((y_true-y_pred),2)
    inv_std = K.exp(-y_pred)
    mse = K.mean(K.batch_dot(inv_std,se))
    reg = K.mean(y_pred)
    return 0.5*(mse + reg)

X = samples
y = para
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
# determine the number of input and output features
n_features = X_train.shape[1]
input_shape = (n_features,) 
output_shape = (y_train.shape[1],)

# scale and standardise
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)

scy = StandardScaler()
y_train = scy.fit_transform(y_train)
y_test = scy.transform(y_test)
y_val = scy.transform(y_val)

# set parameters
neurons = 100
layers = 3
dropout_rate = 0.2
epochs = 500

## define model
model = Sequential()
model.add(Dense(100, activation='relu', kernel_initializer='uniform', input_shape=(n_features,))) # special for only one dimension
for i in range(layers):
    model.add(Dropout(rate = dropout_rate))
    model.add(Dense(neurons, kernel_initializer='uniform', activation='relu'))
model.add(Dense(output_shape[0],kernel_initializer='uniform'))
opt = tf.keras.optimizers.Adam(learning_rate=0.001,
                               beta_1=0.9,beta_2=0.999,epsilon=1e-09,)
model.compile(loss=aleatoric_loss, optimizer=opt, metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
print(model.summary())
history = model.fit(X_train, y_train, batch_size=int(len(X_train)/3), epochs = epochs, shuffle=True, 
                    validation_data=(X_val, y_val), use_multiprocessing=True, callbacks=[es])

train_mse = model.evaluate(X_train, y_train, verbose=0)
test_mse = model.evaluate(X_val, y_val, verbose=0)
# plot loss during training
plt.figure(1)
plt.title('Loss / Mean Squared Error')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

y_pred =  scy.inverse_transform(model.predict(X_val))
y_val =  scy.inverse_transform(y_val)
h = [0,1,2,3]

for i in range(2):
    plt.figure(i+2)
    plt.plot(y_val[:,i],y_val[:,i],'r.')
    plt.plot(y_val[:,i],y_pred[:,i],'ko',alpha=0.4)
    # normalised difference
    plt.figure(100)
    y = abs(y_pred[:,i] - y_val[:,i])/np.max(abs(y_pred[:,i] - y_val[:,i]))
    plt.plot(y_val[:,i]/np.max(y_val[:,i]),y,'o')
plt.show()

i = 3:
plt.figure(i+5)
plt.plot(y_val[:,i],y_val[:,i],'r.')
plt.plot(y_val[:,i],y_pred[:,i],'ko',alpha=0.4)
plt.figure(101)
y = abs(y_pred[:,i]/y_val[:,i])
plt.plot(y,'o')
plt.show()

#testing
y_pred = scy.inverse_transform(model.predict(X_test))
y_test = scy.inverse_transform(y_test)

# plots for mean
for i in range(2):
    #plt.figure(i+2)
    plt.plot(y_test[:,i],y_test[:,i],'r.')
    plt.plot(y_test[:,i],y_pred[:,i],'ko',alpha=0.4)
    plt.figure(7)
    y = abs(y_pred[:,i] - y_test[:,i])/np.max(abs(y_pred[:,i] - y_test[:,i]))
    plt.plot(y_test[:,i]/np.max(y_test[:,i]),y,'o')

print(r2_score(y_test, y_pred))
plt.show()

# plots for standard deviation
i = 3:
plt.figure(i+5)
plt.plot(y_test[:,i],y_test[:,i],'r.')
plt.plot(y_test[:,i],y_pred[:,i],'ko',alpha=0.4)
plt.figure(101)
y = abs(y_pred[:,i]/y_val[:,i])
plt.plot(y,'o')
plt.show()

# Plot the points
plt.scatter(y_pred[:, 0], y_pred[:, 1],marker='x')
plt.scatter(y_test[:, 0], y_test[:, 1], c='r', marker='s', label='-0.3')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot of N Points')
plt.show()

dst = []
for i in range(y_pred.shape[0]):
    dist = distance.euclidean(y_pred[i], y_test[i])
    dst.append(dist)

plt.boxplot(dst)
plt.show()

# only the internal areas
