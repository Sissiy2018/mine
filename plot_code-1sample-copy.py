# mlp for bimodal distribution
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

from MDN_func import mdnglob, save_object
from MDN_func import loss_func, MDN_Full, slice_parameter_vectors_full, mixgauss_full
from MDN_func import predicting
from MDN_func import loss_func_single

#=======================================================================
#=======================================================================
#plot code new

ext = "range103"
#mdn = tensorflow.keras.models.load_model("emu_MDN2_model_"+ext+".h5", compile=False)
# To load weights ...
no_mix, no_parameters, neurons, components, dim_out = mdnglob()
model = MDN_Full(neurons=neurons, ncomp=no_mix,dim=dim_out)
opt = tf.keras.optimizers.Adam(learning_rate=1e-5,beta_1=0.9,beta_2=0.999,epsilon=1e-09,) 
model.compile(loss=loss_func_single, optimizer=opt)
model.load_weights("emu_MDN2_model_"+ext+".h5")


opt = tf.keras.optimizers.Adam(learning_rate=1e-5,beta_1=0.9,beta_2=0.999,epsilon=1e-09,) 

    folder = "../GRID/"

    if "104" in ext:
        grid_name = "grid_0_dim4_100.h5"
    if "103" in ext:
        grid_name = "grid_0_dim4_100l.h5"
    if "105" in ext:
        grid_name = "grid_0_dim4_100h.h5"
    if "102" in ext:
        grid_name = "grid_0_dim4_100vl.h5"

    X, y, header = load_grid(folder + grid_name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)
    x_train = sc.transform(X_train)
    x_val = sc.transform(X_val)
    y_train = scy.transform(y_train)
    y_val = scy.transform(y_val)

    model = MDN_Full(neurons=neurons, ncomp=no_mix,dim=dim_out)
    model.compile(loss=loss_func_single, optimizer=opt)
    model.fit(x=x_train, y=y_train, epochs=1, validation_data=(x_val, y_val), batch_size=1, verbose=1)
    model.load_weights("emu_MDN2_model_"+ext+".h5")

with open("emu_MDN2_sc_"+ext+".pkl", 'rb') as run:
    sc = pickle.load(run)

with open("emu_MDN2_scy_"+ext+".pkl", 'rb') as run:
    scy = pickle.load(run)

def parameter_set():
    # Define the sets of parameters [a, b, c]
    # Define the means and standard deviations for the two Gaussian distributions
    theta_1 = np.arange(0, 1001, 50)
    theta_2 = np.arange(0, 1001, 50)
    std_dev = np.arange(10, 101, 20)
    run = theta_1.shape[0]*theta_2.shape[0]*std_dev.shape[0]
    #sample_size = 500
    
    para_sim = np.empty((run, 3))
    
    count = 0
    # Generate samples from each distribution
    for mean1 in theta_1:
        for mean2 in theta_2:
            for std in std_dev:
                para_sim[count] = np.array([mean1,mean2,std])
                count += 1
    
    return para_sim

para_sim = parameter_set()
no_mix, no_parameters, neurons, components, dim_out = mdnglob()



def generate_prediction_distribution_full(para_sim):
    #para_sim = np.repeat(para_sim, no_run, axis=0)
        #para_sim = (np.expand_dims(para_sim,0))
    input_para = sc.transform(para_sim)
    pred = predicting(input_para,mdn,no_mix,dim_out,scy)
    output_para = inv_trans(pred,scy)
    theta_pred = output_para
        # Append the samples to the main array
        # theta_pred[i] = output_para
        # Return a probability distribution (e.g., numpy array) for the given parameters
    return theta_pred

def generate_simulation_distribution_full(para_sim,no_run):
    
    sim = np.empty((0, 4), dtype=np.float64)
    count = 0
    
    for params in para_sim:
        theta_1, theta_2, std_dev= params
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
        sim = np.vstack((sim, theta_sim))
        count += 1
        print(count)
    return sim

# Generate the prediction and simulation distributions
pred = generate_prediction_distribution_full(para_sim)
sim = generate_simulation_distribution_full(para_sim,1000)

preda = pred[:,:,0]
preda_test = preda.reshape(2205000)

#=======================================================================
theta_1 = np.arange(0, 1001, 50)
theta_2 = np.arange(0, 1001, 50)
std_dev = np.arange(10, 101, 20)
run = theta_1.shape[0]*theta_2.shape[0]*std_dev.shape[0]

mean_diff_std_arr_full = np.empty((0, run), dtype=np.float32)
for col_idx in range(4):
    pred_col = pred[:,:, col_idx]
    pred_col = pred_col.reshape(run*1000)
    sim_col = sim[:, col_idx]
    
    mean_diff_std_arr = np.array([])

    for start_idx in range(0, len(pred_col), 1000):
        end_idx = start_idx + 1000
        pred_subset = pred_col[start_idx:end_idx]
        sim_subset= sim_col[start_idx:end_idx]

        # Perform operations on the subset of data
        mean_diff_std = (np.mean(pred_subset) - np.mean(sim_subset)) / np.std(sim_subset) ### check!!!
        mean_diff_std_arr = np.append(mean_diff_std_arr,mean_diff_std)

    mean_diff_std_arr_full = np.vstack((mean_diff_std_arr_full, mean_diff_std_arr))

#mean_diff_std_arr_full.shape
#Combine the arrays and label them
data = [mean_diff_std_arr_full[0], mean_diff_std_arr_full[1], mean_diff_std_arr_full[2],mean_diff_std_arr_full[3]]
labels = ['Mean', 'Variance', 'Skewness', 'Kurtosis']

# Plot the boxplot
plt.boxplot(data, labels=labels)
plt.ylim(-100, 100)
plt.ylabel('mean_diff_std')
plt.title('Boxplot')
plt.show()

plt.hist(sim[:,0][24500:24599])
plt.hist(pred[:,0][24500:24599])
plt.show()

sim[:,0][0:99]

#=======================================================================
median_diff_M_sim_arr_full = np.empty((0, run), dtype=np.float64)
for col_idx in range(4):
    pred_col = pred[:,:,col_idx]
    pred_col = pred_col.reshape(run*1000)
    sim_col = sim[:, col_idx]
    
    mean_diff_std_arr = np.array([])
    wasserstein_distances_arr = np.array([])
    median_diff_M_sim_arr = np.array([])
    std_ratio_arr = np.array([])

    for start_idx in range(0, len(pred_col), 1000):
        end_idx = start_idx + 1000
        pred_subset = pred_col[start_idx:end_idx]
        sim_subset= sim_col[start_idx:end_idx]

        median_diff_M_sim = np.median(pred_subset - sim_subset) / np.median(sim_subset)
        median_diff_M_sim_arr = np.append(median_diff_M_sim_arr,median_diff_M_sim)
    
    median_diff_M_sim_arr_full = np.vstack((median_diff_M_sim_arr_full, median_diff_M_sim_arr))

#Combine the arrays and label them
data = [median_diff_M_sim_arr_full[0], median_diff_M_sim_arr_full[1], median_diff_M_sim_arr_full[2],median_diff_M_sim_arr_full[3]]
labels = ['Mean', 'Variance', 'Skewness', 'Kurtosis']

# Plot the boxplot
plt.boxplot(data, labels=labels)
plt.ylim(-20, 20)
plt.ylabel('median_diff_M_sim')
plt.title('Boxplot')
plt.show()

plt.hist()

#=======================================================================
std_ratio_arr_full = np.empty((0, run), dtype=np.float64)
for col_idx in range(4):
    pred_col = pred[:,:,col_idx]
    pred_col = pred_col.reshape(run*1000)
    sim_col = sim[:, col_idx]
    
    mean_diff_std_arr = np.array([])
    wasserstein_distances_arr = np.array([])
    median_diff_M_sim_arr = np.array([])
    std_ratio_arr = np.array([])

    for start_idx in range(0, len(pred_col), 1000):
        end_idx = start_idx + 1000
        pred_subset = pred_col[start_idx:end_idx]
        sim_subset= sim_col[start_idx:end_idx]

        # Perform operations on the subset of data
        std_ratio = np.std(pred_subset) / np.std(sim_subset)
        std_ratio_arr = np.append(std_ratio_arr,std_ratio)
    
    std_ratio_arr_full = np.vstack((std_ratio_arr_full, std_ratio_arr))

#Combine the arrays and label them
data_M1 = [std_ratio_arr_full[0], std_ratio_arr_full[1], std_ratio_arr_full[2],std_ratio_arr_full[3]]
labels_M1 = ['Mean', 'Variance', 'Skewness', 'Kurtosis']

# Plot the boxplot
plt.boxplot(data_M1, labels=labels_M1)
plt.ylim(0, 300)
plt.ylabel('std_ratio')
plt.title('Boxplot')
plt.show()

wasserstein_distances_arr_full = np.empty((0, run), dtype=np.float64)
for col_idx in range(4):
    pred_col = pred[:,:,col_idx]
    pred_col = pred_col.reshape(run*1000)
    sim_col = sim[:, col_idx]
    
    mean_diff_std_arr = np.array([])
    wasserstein_distances_arr = np.array([])
    median_diff_M_sim_arr = np.array([])
    std_ratio_arr = np.array([])

    for start_idx in range(0, len(pred_col), 1000):
        end_idx = start_idx + 1000
        pred_subset = pred_col[start_idx:end_idx]
        sim_subset= sim_col[start_idx:end_idx]

        # Perform operations on the subset of data
        
        wasserstein_dist = wasserstein_distance(pred_subset, sim_subset)
        wasserstein_distances_arr = np.append(wasserstein_distances_arr,wasserstein_dist)

    wasserstein_distances_arr_full = np.vstack((wasserstein_distances_arr_full, wasserstein_distances_arr))

#Combine the arrays and label them
data_M1 = [wasserstein_distances_arr_full[0], wasserstein_distances_arr_full[1], wasserstein_distances_arr_full[2],wasserstein_distances_arr_full[3]]
labels_M1 = ['Mean', 'Variance', 'Skewness', 'Kurtosis']

# Plot the boxplot
plt.boxplot(data_M1, labels=labels_M1)
plt.ylabel('wasserstein_distances')
plt.ylim(0, 500)
plt.title('Boxplot')
plt.show()


for i in range(4):
    # Calculate the scores using different methods
    M1_mean_diff_std = np.mean(pred[:,i] - sim[:,i]) / np.std(pred[:,i])
    M1_wasserstein_dist = wasserstein_distance(pred[:,i], sim[:,i])
    M1_median_diff_M_sim = np.median(pred[:,i] - sim[:,i]) / np.median(sim[:,i])
    M1_std_ratio = np.std(pred[:,i]) / np.std(sim[:,i])
    
    # Combine the arrays and label them
    data_M1 = [mean_diff_std_arr, median_diff_M_sim_arr, std_ratio_arr, wasserstein_distances_arr]
    labels_M1 = ['Mean Difference Std', 'Median Difference M_sim', 'Std Ratio', 'Wasserstein Distances']
    
    # Plot the boxplot
    plt.boxplot(data_M1, labels=labels_M1)
    plt.ylabel('Value')
    plt.title('Boxplot')
    plt.show()