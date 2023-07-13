# mlp for bimodal distribution
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# Define the means and standard deviations for the two Gaussian distributions
mean1_range = np.arange(10, 1001, 10)
mean2_range = np.arange(10, 1001, 10)
std_dev = 50
run = mean1_range.shape[0]*mean2_range.shape[0]
sample_size = 500

# Initialize empty arrays to store the samples and parameters
samples = np.empty((run, 4), dtype=np.float64)
para = np.empty((run, 2), dtype=np.float64)

count = 0
# Generate samples from each distribution
for mean1 in mean1_range:
    for mean2 in mean2_range:
        # Generate 250 samples from the first Gaussian distribution
        dist1_samples = np.random.normal(mean1, std_dev, size=250)
        # Generate 250 samples from the second Gaussian distribution
        dist2_samples = np.random.normal(mean2, std_dev, size=250)
        # Concatenate the samples from both distributions
        dist_samples = np.concatenate([dist1_samples, dist2_samples])
        # Calculate the moments
        mean = np.mean(dist_samples)
        variance = np.var(dist_samples)
        skewness = np.mean((dist_samples - mean) ** 3) / np.power(np.var(dist_samples), 3/2)
        kurtosis = np.mean((dist_samples - mean) ** 4) / np.power(np.var(dist_samples), 2) - 3
        # Append the samples to the main array
        samples[count] = np.array([mean,variance,skewness,kurtosis])
        para[count] = np.array([mean1,mean2])
        count += 1

print(samples[1])
print(para[0])

def aleatoric_loss(y_true, y_pred):
    se = K.pow((y_true[:,:4]-y_pred[:,:4]),2)
    inv_std = K.exp(-y_pred[:,:4])
    mse = K.mean(K.batch_dot(inv_std,se))
    reg = K.mean(y_pred[:,:4])
    return 0.5*(mse + reg)

X = para
y = samples
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
# determine the number of input and output features
n_features = X_train.shape[1]
input_shape = (n_features,) 
# output_shape = (y_train.shape[1],) see below

# scale and standardise
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)

scy = StandardScaler()
y_train = scy.fit_transform(y_train)
y_test = scy.transform(y_test)
y_val = scy.transform(y_val)

nr = np.zeros(len(y_train))
y_train = np.column_stack((y_train,nr, nr, nr, nr))
nr = np.zeros(len(y_val))
y_val = np.column_stack((y_val,nr,nr,nr,nr))
output_shape = (y_train.shape[1],)

# set parameters
neurons = 100
layers = 3
dropout_rate = 0.5
epochs = 500
#training = True

inputs = Input(shape=(2,))
hl = Dense(100, kernel_initializer='uniform', activation='relu')(inputs)
for i in range(layers):
    hl = Dense(neurons, kernel_initializer='uniform', activation='relu')(hl)
    hl = Dropout(rate = dropout_rate)(hl, training=True)
outputs = Dense(8, kernel_initializer='uniform')(hl)
model = Model(inputs, outputs)

## define model
#model = Sequential()
#model.add(Dense(100, activation='relu', kernel_initializer='uniform', input_shape=(n_features,))) # special for only one dimension
#for i in range(layers):
    #model.add(Dropout(rate = dropout_rate))
    #model.add(Dense(neurons, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(output_shape[0],kernel_initializer='uniform'))

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

y_pred =  scy.inverse_transform(model.predict(X_val)[:,:4])
y_val =  scy.inverse_transform(y_val[:,:4])
h = [0,1,2,3]

plt.plot(y_val[2], label='y_val[1]')
plt.plot(y_pred[2], label='y_pred[1]')
plt.legend()
plt.show()

for i in range(4):
    plt.figure(i+2)
    plt.plot(y_val[:,i],y_val[:,i],'r.')
    plt.plot(y_val[:,i],y_pred[:,i],'ko',alpha=0.4)
    plt.figure(7)
    y = abs(y_pred[:,i] - y_val[:,i])/np.max(abs(y_pred[:,i] - y_val[:,i]))
    plt.plot(y_val[:,i]/np.max(y_val[:,i]),y,'o')
plt.show()

print(r2_score(y_val, y_pred[:,:4]))

ext = "new"
model.save("emu_model_"+ext+".h5")
save_object(sc, "emu_sc_"+ext+".pkl")
save_object(scy, "emu_scy_"+ext+".pkl")

save_object(X_test, "X_test_"+ext+".pkl")
save_object(y_test, "y_test_"+ext+".pkl")

#testing
y_pred_test = scy.inverse_transform(model.predict(X_test)[:,:4])
y_test = scy.inverse_transform(y_test[:,:4])

for i in range(4):
    plt.figure(i+2)
    plt.plot(y_test[:,i],y_test[:,i],'r.')
    plt.plot(y_test[:,i],y_pred_test[:,i],'ko',alpha=0.4)
    plt.figure(7)
    y = abs(y_pred_test[:,i] - y_test[:,i])/np.max(abs(y_pred_test[:,i] - y_test[:,i]))
    plt.plot(y_test[:,i]/np.max(y_test[:,i]),y,'o')
plt.show()

print(r2_score(y_test, y_pred_test[:,:4]))
plt.show()



# plot code
no_run = 100

generate_prediction_distribution(10,10)
# Function to generate prediction distribution
def generate_prediction_distribution(theta_1, theta_2):
    theta_pred = np.empty((no_run, 4), dtype=np.float64)
    #for i in range(no_run):
    para_sim = np.array([[theta_1,theta_2]]*no_run)
        #para_sim = (np.expand_dims(para_sim,0))
    input_para = sc.fit_transform(para_sim)
    output_para = scy.inverse_transform(model.predict(input_para)[:,:4])
    theta_pred = output_para
        # Append the samples to the main array
        # theta_pred[i] = output_para
        # Return a probability distribution (e.g., numpy array) for the given parameters
    return theta_pred

def generate_simulation_distribution(theta_1, theta_2):
    theta_sim = np.empty((no_run, 4), dtype=np.float64)
    for i in range(no_run):
        dist1_samples = np.random.normal(theta_1, std_dev, size=250)
        # Generate 250 samples from the second Gaussian distribution
        dist2_samples = np.random.normal(theta_2, std_dev, size=250)
        # Concatenate the samples from both distributions
        dist_samples = np.concatenate([dist1_samples, dist2_samples])
        mean = np.mean(dist_samples)
        variance = np.var(dist_samples)
        skewness = np.mean((dist_samples - mean) ** 3) / np.power(np.var(dist_samples), 3/2)
        kurtosis = np.mean((dist_samples - mean) ** 4) / np.power(np.var(dist_samples), 2) - 3
        # Append the samples to the main array
        theta_sim[i] = np.array([mean,variance,skewness,kurtosis])
    # Return a probability distribution (e.g., numpy array) for the given parameters
    return theta_sim
    
# Define the sets of parameters [a, b]
# Define the means and standard deviations for the two Gaussian distributions
theta_1 = np.arange(50, 501, 10)
theta_2 = np.arange(50, 501, 10)
std_dev = 50
run = theta_1.shape[0]*theta_2.shape[0]
sample_size = 500

para_sim = np.empty((run, 2), dtype=np.float64)

count = 0
# Generate samples from each distribution
for mean1 in theta_1:
    for mean2 in theta_2:
        para_sim[count] = np.array([mean1,mean2])
        count += 1

# Initialize lists to store the results
mean_diff_std_arr = np.array([])
median_diff_M_sim_arr =np.array([])
std_ratio_arr = np.array([])
wasserstein_distances_arr = np.array([])

# Iterate over each parameter set
for params in para_sim:
    a, b = params

    # Generate the prediction and simulation distributions
    pred = generate_prediction_distribution(a, b)[:,0]
    sim = generate_simulation_distribution(a, b)[:,0]

    # Calculate the scores using different methods
    mean_diff_std = np.mean(pred - sim) / np.std(pred)
    wasserstein_dist = wasserstein_distance(pred, sim)
    median_diff_M_sim = np.median(pred - sim) / np.median(sim)
    std_ratio = np.std(pred) / np.std(sim)

    # Append the scores to the respective lists
    mean_diff_std_arr = np.append(mean_diff_std_arr,mean_diff_std)
    wasserstein_distances_arr = np.append(wasserstein_distances_arr,wasserstein_dist)
    median_diff_M_sim_arr = np.append(median_diff_M_sim_arr,median_diff_M_sim)
    std_ratio_arr = np.append(std_ratio_arr,std_ratio)

plt.boxplot(mean_diff_std_arr)
plt.show()

plt.boxplot(wasserstein_distances_arr)
plt.show()

plt.boxplot(median_diff_M_sim_arr)
plt.show()

plt.boxplot(std_ratio_arr)
plt.show()

# Print the results
print("KS Scores:", ks_scores)
print("Wasserstein Distances:", wasserstein_distances)
print("T-test p-values:", ttest_p_values)
print("Mann-Whitney U p-values:", mannwhitneyu_p_values)

# Plot the distributions for the first parameter set as an example
plt.plot(prediction_dist, label='Prediction Distribution')
plt.plot(simulation_dist, label='Simulation Distribution')
plt.legend()
plt.show()

theta_1 = 50
theta_2 = 200
std_dev = 50
no_run = 100
theta_sim = np.empty((no_run, 4), dtype=np.float64)
theta_pred = np.empty((no_run, 4), dtype=np.float64)

for i in range(no_run):
    dist1_samples = np.random.normal(theta_1, std_dev, size=250)
    # Generate 250 samples from the second Gaussian distribution
    dist2_samples = np.random.normal(theta_2, std_dev, size=250)
    # Concatenate the samples from both distributions
    dist_samples = np.concatenate([dist1_samples, dist2_samples])
    mean = np.mean(dist_samples)
    variance = np.var(dist_samples)
    skewness = np.mean((dist_samples - mean) ** 3) / np.power(np.var(dist_samples), 3/2)
    kurtosis = np.mean((dist_samples - mean) ** 4) / np.power(np.var(dist_samples), 2) - 3
    # Append the samples to the main array
    theta_sim[i] = np.array([mean,variance,skewness,kurtosis])

for i in range(no_run):
    para_sim = np.array([theta_1,theta_2])
    para_sim = (np.expand_dims(para_sim,0))
    input_para = sc.fit_transform(para_sim)
    output_para = scy.inverse_transform(model.predict(input_para)[:,:4])
    # Append the samples to the main array
    theta_pred[i] = output_para

para_sim = np.array([[theta_1, 50]]*no_run)

#para_sim = (np.expand_dims(para_sim,0))
input_para = sc.fit_transform(para_sim)
output_para = scy.inverse_transform(model.predict(input_para)[:,:4])

# try the first column
# Assuming pred and sim are NumPy arrays or lists
pred = theta_pred[:,0]
sim = theta_sim[:,0]
mean_diff_std = np.mean(pred - sim) / np.std(pred)

# Calculate the median difference over M_sim
median_diff_M_sim = np.median(pred - sim) / np.median(sim)

# Calculate the ratio of standard deviations (std_pred / std_sim)
std_ratio = np.std(pred) / np.std(sim)

# Calculate the Wasserstein distance
wasserstein_dist = wasserstein_distance(pred, sim)

print("Mean difference over standard deviation:", mean_diff_std)








dst_test_1 = []
dst_test_2 = []
dst_test_3 = []
dst_test_4 = []
for i in range(4):
    # Calculate the mean difference over standard deviation
    mean_diff_std = np.mean(pred_test[:,i] - sim_test[:,i]) / np.std(sim_test[:,i])
    dst_test_1.append(mean_diff_std)
    # Calculate the median difference over M_sim
    median_diff_M_sim = abs(np.median(pred_test[:,i] - sim_test[i])) / np.median(sim_test[:,i])
    dst_test_2.append(median_diff_M_sim)
    # Calculate the ratio of standard deviations (std_pred / std_sim)
    std_ratio = np.std(pred_test[i]) / np.std(sim_test[i])
    dst_test_3.append(std_ratio)
    # Calculate the Wasserstein distance
    wasserstein_dist = wasserstein_distance(pred_test[i], sim_test[i])
    dst_test_4.append(wasserstein_dist)

# Print the results
print("Mean difference over standard deviation:", dst_1)
plt.boxplot(dst_1)
plt.show()
plt.boxplot(dst_2)
plt.show()
plt.boxplot(dst_3)
plt.show()
print("Wasserstein distance:", dst_4)
plt.boxplot(dst_4)
plt.show()





# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
df = read_csv(path, header=None)
print(df.head())
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
print(X.shape)
print(y.shape)
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
print(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,))) # special for only one dimension
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# configure early stopping
es = EarlyStopping(monitor='val_loss', patience=5)
# summarize the model
model.summary()
plot_model(model, 'model.png', show_shapes=True)
# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))
# fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0,callbacks=[cp_callback])

latest = tf.train.latest_checkpoint(checkpoint_dir)
# Create a new model instance
model = create_model()
# Load the previously saved weights
model.load_weights(latest)

# save model to file
model.save('model.h5')
# load the model from file
model = load_model('model.h5')
# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
# make a prediction
row = [5.1,3.5,1.4,0.2]
yhat = model.predict([row])
print('Predicted: %s (class=%d)' % (yhat, np.argmax(yhat)))