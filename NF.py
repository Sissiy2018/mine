import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from torch import nn
from torch import optim

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

def bimodal_distribution(n_samples, mean1, cov1, mean2, cov2):
    # Generate data for mode 1
    mode1_samples = np.random.multivariate_normal(mean1, cov1, n_samples // 2)
    # Generate data for mode 2
    mode2_samples = np.random.multivariate_normal(mean2, cov2, n_samples // 2)

    # Combine the data from both modes
    data = np.vstack((mode1_samples, mode2_samples))

    return data

def generate_2D_bimodal():
    # Parameters for mode 1
    mean1 = [2, 3]
    cov1 = [[1, 0.5], [0.5, 1]]

    # Parameters for mode 2
    mean2 = [8, 6]
    cov2 = [[1, -0.7], [-0.7, 1]]

    # Number of samples to generate
    n_samples = 1000

    # Generate the bimodal distribution
    data = bimodal_distribution(n_samples, mean1, cov1, mean2, cov2)

    # Scatter plot to visualize the data
    #plt.scatter(data[:, 0], data[:, 1], alpha=0.6)
    #plt.title("2D Bimodal Distribution")
    #plt.xlabel("X")
    #plt.ylabel("Y")
    #plt.grid(True)
    #plt.show()
    return data

def create_samples():
    # Define the means and standard deviations for the two Gaussian distributions
    mean1 = 50
    mean2 = 300
    std_dev = 10
    run = 1000
    sample_size = 500
    samples = np.empty((run, 2), dtype=np.float64)

    for i in range(run):
        # Generate 250 samples from the first Gaussian distribution
        dist1_samples = np.random.normal(mean1, std_dev, size=int(sample_size//2))
        # Generate 250 samples from the second Gaussian distribution
        dist2_samples = np.random.normal(mean2, std_dev, size=int(sample_size//2))
        # Concatenate the samples from both distributions
        dist_samples = np.concatenate([dist1_samples, dist2_samples])
        # Draw one random sample from the 'samples' array
        random_sample = np.random.choice(dist_samples,2)
        samples[i] = random_sample
    
    return samples

samples = create_samples()
plt.hist(samples)
plt.show()

data = generate_2D_bimodal()
plt.scatter(data[:, 0],data[:, 1])
plt.show()

#x, y = datasets.make_moons(128, noise=.1)
#plt.scatter(x[:, 0], x[:, 1])
#plt.show()

num_layers = 5
base_dist = StandardNormal(shape=[2])

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=2))
    transforms.append(MaskedAffineAutoregressiveTransform(features=2, 
                                                          hidden_features=4))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters())

num_iter = 5000
for i in range(num_iter):
    #x, y = datasets.make_moons(128, noise=.1)
    x = create_samples()
    x = torch.tensor(x, dtype=torch.float32)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x).mean()
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 500 == 0:
        xline = torch.linspace(0, 12,100)
        yline = torch.linspace(0, 12,100)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

        plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
        plt.title('iteration {}'.format(i + 1))
        plt.show()

num_iter = 5000
for i in range(num_iter):
    #x, y = datasets.make_moons(128, noise=.1)
    x = create_samples()
    x = torch.tensor(x, dtype=torch.float32)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x).mean()
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 500 == 0:
        xline = torch.linspace(-100, 1000,100)
        #yline = torch.linspace(0, 12,100)
        #xgrid, ygrid = torch.meshgrid(xline, yline)
        #xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid = flow.log_prob(xline).exp().reshape(100)

        #plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
        plt.hist(zgrid.numpy())
        plt.title('iteration {}'.format(i + 1))
        plt.show()
