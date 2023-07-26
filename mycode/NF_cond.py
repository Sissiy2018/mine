import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from torch import nn
from torch import optim

from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation, RandomPermutation
from nflows.nn.nets import ResidualNet

from sklearn.preprocessing import StandardScaler

def draw_samples_from_mixture(num_samples, num_components):
    means = np.array([[0, 0],[50,50]])  # Random means for each Gaussian component
    cov = np.array([[1, 0.5], [0.5, 1]])  # Covariance matrix (assuming all components have the same covariance)

    samples = np.empty((num_samples, 2), dtype=np.float64)
    labels = np.empty((num_samples,1), dtype=np.float64)

    for i in range(num_samples):
        # Randomly choose a component from the mixture
        component = np.random.choice(num_components)

        # Generate a sample from the selected Gaussian component
        sample = np.random.multivariate_normal(means[component], cov)
        samples[i] = sample
        labels[i] = means[component][0]

    return samples, labels

# Example usage:
num_samples = 1000
num_components = 2

samples, labels = draw_samples_from_mixture(num_samples, num_components)
plt.scatter(samples[:, 0], samples[:, 1],c=labels)
plt.show()

x, y = datasets.make_moons(128, noise=.1)
plt.scatter(x[:, 0], x[:, 1], c=y)

num_layers = 5
base_dist = ConditionalDiagonalNormal(shape=[2], 
                                      context_encoder=nn.Linear(1, 4))

transforms = []
for _ in range(num_layers):
    #transforms.append(ReversePermutation(features=2))
    transforms.append(MaskedAffineAutoregressiveTransform(features=2, 
                                                          hidden_features=4, 
                                                          context_features=1,num_blocks=2,
                                                          use_residual_blocks=False,
                                                          random_mask=False,
                                                          activation=torch.tanh,
                                                          dropout_probability=0.0,use_batch_norm=True,))
    transforms.append(RandomPermutation(features=2))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters())

num_iter = 5000
for i in range(num_iter):
    x, y = draw_samples_from_mixture(num_samples, num_components)
    
    #sc = StandardScaler()
    #x = sc.fit_transform(x)

    #scy = StandardScaler()
    #y = scy.fit_transform(y)
    
    x = torch.tensor(x, dtype=torch.float32)
    #y = torch.tensor(y, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x, context=y).mean()
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 500 == 0:
        fig, ax = plt.subplots(1, 2)
        xline = torch.linspace(-20, 80,100)
        yline = torch.linspace(-20, 80,100)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
         
        with torch.no_grad():
            zgrid0 = flow.log_prob(xyinput, torch.full((10000, 1), 50, dtype=torch.float32)).exp().reshape(100, 100)
            zgrid1 = flow.log_prob(xyinput, torch.zeros(10000, 1)).exp().reshape(100, 100)

        ax[0].contourf(xgrid.numpy(), ygrid.numpy(), zgrid0.numpy())
        ax[1].contourf(xgrid.numpy(), ygrid.numpy(), zgrid1.numpy())
        plt.title('iteration {}'.format(i + 1))
        plt.show()

num_iter = 5000
for i in range(num_iter):
    x, y = draw_samples_from_mixture(num_samples, num_components)
    sc = StandardScaler()
    x = sc.fit_transform(x)

    scy = StandardScaler()
    y = scy.fit_transform(y)

    x = torch.tensor(x, dtype=torch.float32)
    #y = torch.tensor(y, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x, context=y).mean()
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 500 == 0:
        #fig, ax = plt.subplots(1, 2)
        xline = torch.linspace(-20, 60,100)
        yline = torch.linspace(-20, 60,100)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

        plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
        plt.title('iteration {}'.format(i + 1))
        plt.show()

## y reshape? activator? blocks? etc?log do?
