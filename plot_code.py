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

#=======================================================================
#=======================================================================
#plot code new

ext = "range105_method1_dummy_aloss"
model = tensorflow.keras.models.load_model("emu_model_"+ext+".h5", compile=False)

with open("emu_sc_"+ext+".pkl", 'rb') as run:
    sc = pickle.load(run)

with open("emu_scy_"+ext+".pkl", 'rb') as run:
    scy = pickle.load(run)

def parameter_set():
    # Define the sets of parameters [a, b, c]
    # Define the means and standard deviations for the two Gaussian distributions
    theta_1 = np.arange(0, 10001, 200)
    theta_2 = np.arange(0, 10001, 200)
    std_dev = np.arange(10, 1001, 50)
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

def generate_prediction_distribution_full(para_sim,no_run):
    para_sim = np.repeat(para_sim, no_run, axis=0)
        #para_sim = (np.expand_dims(para_sim,0))
    input_para = sc.transform(para_sim)
    output_para = scy.inverse_transform(model.predict(input_para)[:,:4])
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
pred = generate_prediction_distribution_full(para_sim,100)
sim = generate_simulation_distribution_full(para_sim,100)

#=======================================================================
# plot mean_diff_std for 4 moments
mean_diff_std_arr_full = np.empty((0, 52020), dtype=np.float64)
for col_idx in range(4):
    pred_col = pred[:, col_idx]
    sim_col = sim[:, col_idx]
    
    mean_diff_std_arr = np.array([])

    for start_idx in range(0, len(pred_col), 100):
        end_idx = start_idx + 100
        pred_subset = pred_col[start_idx:end_idx]
        sim_subset= sim_col[start_idx:end_idx]

        # Perform operations on the subset of data
        mean_diff_std = np.mean(pred_subset - sim_subset) / np.std(pred_subset)
        mean_diff_std_arr = np.append(mean_diff_std_arr,mean_diff_std)

    mean_diff_std_arr_full = np.vstack((mean_diff_std_arr_full, mean_diff_std_arr))

#Combine the arrays and label them
data = [mean_diff_std_arr_full[0], mean_diff_std_arr_full[1], mean_diff_std_arr_full[2],mean_diff_std_arr_full[3]]
labels = ['Mean', 'Variance', 'Skewness', 'Kurtosis']

# Plot the boxplot
plt.boxplot(data_M1, labels=labels_M1)
plt.ylabel('mean_diff_std')
plt.title('Boxplot')
plt.show()

#=======================================================================
median_diff_M_sim_arr_full = np.empty((0, 52020), dtype=np.float64)
for col_idx in range(4):
    pred_col = pred[:, col_idx]
    sim_col = sim[:, col_idx]
    
    mean_diff_std_arr = np.array([])
    wasserstein_distances_arr = np.array([])
    median_diff_M_sim_arr = np.array([])
    std_ratio_arr = np.array([])

    for start_idx in range(0, len(pred_col), 100):
        end_idx = start_idx + 100
        pred_subset = pred_col[start_idx:end_idx]
        sim_subset= sim_col[start_idx:end_idx]

        median_diff_M_sim = np.median(pred_subset - sim_subset) / np.median(sim_subset)
        median_diff_M_sim_arr = np.append(median_diff_M_sim_arr,median_diff_M_sim)
    
    median_diff_M_sim_arr_full = np.vstack((median_diff_M_sim_arr_full, median_diff_M_sim_arr))

#Combine the arrays and label them
data = [median_diff_M_sim_arr_full[0], median_diff_M_sim_arr_full[1], median_diff_M_sim_arr_full[2],mmedian_diff_M_sim_arr_full[3]]
labels = ['Mean', 'Variance', 'Skewness', 'Kurtosis']

# Plot the boxplot
plt.boxplot(data_M1, labels=labels_M1)
plt.ylabel('median_diff_M_sim')
plt.title('Boxplot')
plt.show()

#=======================================================================
std_ratio_arr_full = np.empty((0, 52020), dtype=np.float64)
for col_idx in range(4):
    pred_col = pred[:, col_idx]
    sim_col = sim[:, col_idx]
    
    mean_diff_std_arr = np.array([])
    wasserstein_distances_arr = np.array([])
    median_diff_M_sim_arr = np.array([])
    std_ratio_arr = np.array([])

    for start_idx in range(0, len(pred_col), 100):
        end_idx = start_idx + 100
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
plt.ylim(0, 100)
plt.ylabel('std_ratio')
plt.title('Boxplot')
plt.show()

wasserstein_distances_arr_full = np.empty((0, 52020), dtype=np.float64)
for col_idx in range(4):
    pred_col = pred[:, col_idx]
    sim_col = sim[:, col_idx]
    
    mean_diff_std_arr = np.array([])
    wasserstein_distances_arr = np.array([])
    median_diff_M_sim_arr = np.array([])
    std_ratio_arr = np.array([])

    for start_idx in range(0, len(pred_col), 100):
        end_idx = start_idx + 100
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
plt.ylim(-1000, 1000)
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