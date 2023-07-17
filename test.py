
import elfi

mu = elfi.Prior('uniform', -2, 4)
sigma = elfi.Prior('uniform', 1, 4)

import scipy.stats as ss
import numpy as np
    
def simulator(mu, sigma, batch_size=1, random_state=None):
    mu, sigma = np.atleast_1d(mu, sigma)
    return ss.norm.rvs(mu[:, None], sigma[:, None], size=(batch_size, 30), random_state=random_state)

a = simulator(3,5)

def mean(y):
    return np.mean(y, axis=1)
    
def var(y):
    return np.var(y, axis=1)


# Set the generating parameters that we will try to infer
mean0 = 1
std0 = 3
    
# Generate some data (using a fixed seed here)
np.random.seed(20170525) 
y0 = simulator(mean0, std0)
print(y0)

# Add the simulator node and observed data to the model
sim = elfi.Simulator(simulator, mu, sigma, observed=y0)
    
 # Add summary statistics to the model
S1 = elfi.Summary(mean, sim)
S2 = elfi.Summary(var, sim)
    
# Specify distance as euclidean between summary vectors (S1, S2) from simulated and
# observed data
d = elfi.Distance('euclidean', S1, S2)

If you have ``graphviz`` installed to your system, you can also
visualize the model:

.. code:: ipython3

    # Plot the complete model (requires graphviz)
elfi.draw(d)




.. image:: https://raw.githubusercontent.com/elfi-dev/notebooks/dev/figures/quickstart_files/quickstart_11_0.svg



.. Note:: The automatic naming of nodes may not work in all environments e.g. in interactive Python shells. You can alternatively provide a name argument for the nodes, e.g. ``S1 = elfi.Summary(mean, sim, name='S1')``.

We can try to infer the true generating parameters ``mean0`` and
``std0`` above with any of ELFI’s inference methods. Let’s use ABC
Rejection sampling and sample 1000 samples from the approximate
posterior using threshold value 0.5:

.. code:: ipython3

    rej = elfi.Rejection(d, batch_size=10000, seed=30052017)
    res = rej.sample(1000, threshold=.5)
    print(res)


.. parsed-literal::

    Method: Rejection
    Number of samples: 1000
    Number of simulations: 120000
    Threshold: 0.492
    Sample means: mu: 0.748, sigma: 3.1
    


Let’s plot also the marginal distributions for the parameters:

.. code:: ipython3

    import matplotlib.pyplot as plt
    res.plot_marginals()
    plt.show()



.. image:: https://raw.githubusercontent.com/elfi-dev/notebooks/dev/figures/quickstart_files/quickstart_16_0.png
