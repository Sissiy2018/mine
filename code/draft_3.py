#%matplotlib inline
import os
import tempfile

import scipy.stats as st

import pyabc

pyabc.settings.set_figure_params('pyabc')  # for beautified plots

def model(parameter):
    return {"data": parameter["mu"] + 0.5 * np.random.randn()}
prior = pyabc.Distribution(mu=pyabc.RV("uniform", 0, 5))
def distance(x, x0):
    return abs(x["data"] - x0["data"])
abc = pyabc.ABCSMC(model, prior, distance, population_size=1000)





# Define a gaussian model

sigma = 0.5


def model(parameters):
    # sample from a gaussian
    y = st.norm(parameters.x, sigma).rvs()
    # return the sample as dictionary
    return {"y": y}

# We define two models, but they are identical so far
models = [model, model]

# However, our models' priors are not the same.
# Their mean differs.
mu_x_1, mu_x_2 = 0, 1
parameter_priors = [
    pyabc.Distribution(x=pyabc.RV("norm", mu_x_1, sigma)),
    pyabc.Distribution(x=pyabc.RV("norm", mu_x_2, sigma)),
]