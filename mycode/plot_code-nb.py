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

ext = "range105_method1_nodummy_aloss"
model = tensorflow.keras.models.load_model("emu_model_"+ext+".h5", compile=False)

with open("emu_sc_"+ext+".pkl", 'rb') as run:
    sc = pickle.load(run)

with open("emu_scy_"+ext+".pkl", 'rb') as run:
    scy = pickle.load(run)

def parameter_set():
    # Define the sets of parameters [a, b, c]
    # Define the means and standard deviations for the two Gaussian distributions
    r_range = np.arange(5, 1001, 5)
    p_range = np.arange(0.05, 1.00, 0.05)

    run = r_range.shape[0]*p_range.shape[0]
    #sample_size = 500
    
    para_sim = np.empty((run, 2))
    
    count = 0
    # Generate samples from each distribution
    for r in r_range:
        for p in p_range:
            para_sim[count] = np.array([r,p])
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
        r,p = params
        theta_sim = np.empty((no_run, 4), dtype=np.float64)
        for i in range(no_run):
            # Generate 500 samples from the distribution
            dist_samples = np.random.negative_binomial(r, p, size=500)
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
r_range = np.arange(5, 1001, 5)
p_range = np.arange(0.05, 1.00, 0.05)
run = r_range.shape[0]*p_range.shape[0]

mean_diff_std_arr_full = np.empty((0, run), dtype=np.float64)
for col_idx in range(4):
    pred_col = pred[:, col_idx]
    sim_col = sim[:, col_idx]
    
    mean_diff_std_arr = np.array([])

    for start_idx in range(0, len(pred_col), 100):
        end_idx = start_idx + 100
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
plt.ylabel('mean_diff_std')
plt.ylim(-100,10)
plt.title('Boxplot')
plt.show()

#=======================================================================
median_diff_M_sim_arr_full = np.empty((0, run), dtype=np.float64)
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

        median_diff_M_sim = abs(np.median(pred_subset) - np.median(sim_subset)) / np.median(sim_subset)
        median_diff_M_sim_arr = np.append(median_diff_M_sim_arr,median_diff_M_sim)
    
    median_diff_M_sim_arr_full = np.vstack((median_diff_M_sim_arr_full, median_diff_M_sim_arr))

#Combine the arrays and label them
data = [median_diff_M_sim_arr_full[0], median_diff_M_sim_arr_full[1], median_diff_M_sim_arr_full[2],median_diff_M_sim_arr_full[3]]
labels = ['Mean', 'Variance', 'Skewness', 'Kurtosis']

# Plot the boxplot
plt.boxplot(data, labels=labels)
plt.ylabel('median_diff_M_sim')
#plt.ylim(-100, 100)
plt.title('Boxplot')
plt.show()

#=======================================================================
std_ratio_arr_full = np.empty((0, run), dtype=np.float64)
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
plt.ylim(-1, 1)
plt.ylabel('std_ratio')
plt.title('Boxplot')
plt.show()

wasserstein_distances_arr_full = np.empty((0, run), dtype=np.float64)
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
plt.ylim(-100, 8000)
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