import numpy as np
from scipy.stats import wasserstein_distance

# Assuming pred and sim are NumPy arrays or lists
pred, sim
# Calculate the mean difference over standard deviation
mean_diff_std = np.mean(pred - sim) / np.std(pred)

# Calculate the median difference over M_sim
median_diff_M_sim = np.median(pred - sim) / np.median(sim)

# Calculate the ratio of standard deviations (std_pred / std_sim)
std_ratio = np.std(pred) / np.std(sim)

# Calculate the Wasserstein distance
wasserstein_dist = wasserstein_distance(pred, sim)

# Print the results
print("Mean difference over standard deviation:", mean_diff_std)
print("Median difference over M_sim:", median_diff_M_sim)
print("Ratio of standard deviations (std_pred / std_sim):", std_ratio)
print("Wasserstein distance:", wasserstein_dist)
