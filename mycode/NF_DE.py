"""
Title: Density estimation using Real NVP
Authors: [Mandolini Giorgio Maria](https://www.linkedin.com/in/giorgio-maria-mandolini-a2a1b71b4/), [Sanna Daniele](https://www.linkedin.com/in/daniele-sanna-338629bb/), [Zannini Quirini Giorgio](https://www.linkedin.com/in/giorgio-zannini-quirini-16ab181a0/)
Date created: 2020/08/10
Last modified: 2020/08/10
Description: Estimating the density distribution of the "double moon" dataset.
Accelerator: GPU
"""

"""
## Introduction

The aim of this work is to map a simple distribution - which is easy to sample
and whose density is simple to estimate - to a more complex one learned from the data.
This kind of generative model is also known as "normalizing flow".

In order to do this, the model is trained via the maximum
likelihood principle, using the "change of variable" formula.

We will use an affine coupling function. We create it such that its inverse, as well as
the determinant of the Jacobian, are easy to obtain (more details in the referenced paper).

**Requirements:**

* Tensorflow 2.9.1
* Tensorflow probability 0.17.0

**Reference:**

[Density estimation using Real NVP](https://arxiv.org/abs/1605.08803)
"""

"""
## Setup
# pip install --upgrade tensorflow
"""
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf
print(tf.__version__)
print(tfp.__version__)

"""
## Load the data
"""

def bimodal_density(x, mean1, mean2, std1, std2):
    return 0.3 * (np.exp(-0.5 * ((x - mean1) / std1) ** 2) / (std1 * np.sqrt(2 * np.pi))
                  + np.exp(-0.5 * ((x - mean2) / std2) ** 2) / (std2 * np.sqrt(2 * np.pi)))

def generate_bimodal_data(num_samples=1000, mean1=0, mean2=5, std1=1, std2=1):
    np.random.seed(42)

    # Generate x-axis values
    x_values = np.linspace(-10, 15, num_samples)

    # Calculate probability density at each x value
    density_values = bimodal_density(x_values, mean1, mean2, std1, std2)

    # Stack x and density values to create 2D dataset
    bimodal_data = np.column_stack((x_values, density_values))

    return bimodal_data

def generate_1D_bimodal():
    num_samples = 1000
    mean1, mean2 = 50, 300
    std1, std2 = 10, 10

    bimodal_data = generate_bimodal_data(num_samples, mean1, mean2, std1, std2)
    print("Bimodal dataset shape:", bimodal_data.shape)

    # Plot the bimodal distribution
    plt.figure(figsize=(10, 6))
    plt.plot(bimodal_data[:, 0], bimodal_data[:, 1], label='Bimodal Distribution')
    plt.xlabel('X')
    plt.ylabel('Density')
    plt.title('1D Bimodal Gaussian Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()
    return bimodal_data

"""

"""

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
    plt.scatter(data[:, 0], data[:, 1], alpha=0.6)
    plt.title("2D Bimodal Distribution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()
    return data


#data = make_moons(3000, noise=0.05)[0].astype("float32")
data = generate_2D_bimodal()
norm = layers.Normalization()
norm.adapt(data)
normalized_data = norm(data)

"""
## Affine coupling layer
"""

# Creating a custom layer with keras API.
output_dim = 256
reg = 0.01


def Coupling(input_shape):
    input = keras.layers.Input(shape=input_shape)

    t_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    t_layer_2 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_1)
    t_layer_3 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_2)
    t_layer_4 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_3)
    t_layer_5 = keras.layers.Dense(
        input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_4)

    s_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    s_layer_2 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_1)
    s_layer_3 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_2)
    s_layer_4 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_3)
    s_layer_5 = keras.layers.Dense(
        input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_4)

    return keras.Model(inputs=input, outputs=[s_layer_5, t_layer_5])


"""
## Real NVP
"""


class RealNVP(keras.Model):
    def __init__(self, num_coupling_layers):
        super().__init__()

        self.num_coupling_layers = num_coupling_layers

        # Distribution of the latent space.
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[0.0, 0.0], scale_diag=[1.0, 1.0]
        )
        self.masks = np.array(
            [[0, 1], [1, 0]] * (num_coupling_layers // 2), dtype="float32"
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.layers_list = [Coupling(2) for i in range(num_coupling_layers)]

    @property
    def metrics(self):
        """List of the model's metrics.

        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def call(self, x, training=True):
        log_det_inv = 0
        direction = 1
        if training:
            direction = -1
        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
                + x_masked
            )
            log_det_inv += gate * tf.reduce_sum(s, [1])

        return x, log_det_inv

    # Log likelihood of the normal distribution plus the log determinant of the jacobian.

    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -tf.reduce_mean(log_likelihood)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.log_loss(data)

        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}


"""
## Model training
"""

model = RealNVP(num_coupling_layers=6)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

history = model.fit(
    normalized_data, batch_size=256, epochs=300, verbose=2, validation_split=0.2
)

"""
## Performance evaluation
"""

plt.figure(figsize=(15, 10))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.legend(["train", "validation"], loc="upper right")
plt.ylabel("loss")
plt.xlabel("epoch")

# From data to latent space.
z, _ = model(normalized_data)

# From latent space to data.
samples = model.distribution.sample(3000)
x, _ = model.predict(samples)

f, axes = plt.subplots(2, 2)
f.set_size_inches(20, 15)

axes[0, 0].scatter(normalized_data[:, 0], normalized_data[:, 1], color="r")
axes[0, 0].set(title="Inference data space X", xlabel="x", ylabel="y")
axes[0, 1].scatter(z[:, 0], z[:, 1], color="r")
axes[0, 1].set(title="Inference latent space Z", xlabel="x", ylabel="y")
axes[0, 1].set_xlim([-3.5, 4])
axes[0, 1].set_ylim([-4, 4])
axes[1, 0].scatter(samples[:, 0], samples[:, 1], color="g")
axes[1, 0].set(title="Generated latent space Z", xlabel="x", ylabel="y")
axes[1, 1].scatter(x[:, 0], x[:, 1], color="g")
axes[1, 1].set(title="Generated data space X", label="x", ylabel="y")
axes[1, 1].set_xlim([-2, 2])
axes[1, 1].set_ylim([-2, 2])

plt.show()