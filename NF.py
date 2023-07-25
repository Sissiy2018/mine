import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Generate bimodal data for training
def generate_bimodal_data(num_samples=1000):
    np.random.seed(42)
    mode1 = np.random.normal(loc=0, scale=1, size=num_samples // 2)
    mode2 = np.random.normal(loc=5, scale=1, size=num_samples // 2)
    data = np.concatenate([mode1, mode2])
    return data

# Define the Normalizing Flow model
def build_normalizing_flow_model(input_dim, num_bijectors):
    inputs = Input(shape=(input_dim,))
    x = inputs

    bijectors = []
    for _ in range(num_bijectors):
        bijectors.append(tfp.bijectors.Affine(shift_and_log_scale_fn=tfp.bijectors.AffineScalar.params_fn))

    normalizing_flow = tfp.bijectors.Chain(bijectors=bijectors)
    distribution = tfp.distributions.TransformedDistribution(
        distribution=tfp.distributions.Normal(loc=0.0, scale=1.0),
        bijector=normalizing_flow
    )

    log_prob_layer = distribution.log_prob(inputs)
    model = Model(inputs, log_prob_layer)

    return model

def main():
    num_samples = 1000
    data = generate_bimodal_data(num_samples)
    input_dim = 1  # Assuming 1-dimensional data
    num_bijectors = 6  # Number of bijectors in the normalizing flow

    # Build and compile the normalizing flow model
    model = build_normalizing_flow_model(input_dim, num_bijectors)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=lambda _, log_prob: -log_prob)

    # Train the normalizing flow model
    model.fit(data, np.zeros((num_samples, 1)), epochs=100, batch_size=64)

    # Generate new samples from the learned bimodal distribution
    num_samples_to_generate = 1000
    z = np.random.normal(size=(num_samples_to_generate, input_dim))
    samples = model.predict(z)

    # Print the estimated density for a few example points
    example_points = np.array([-2, 0, 2, 4, 6])
    log_density = model.predict(example_points)
    density = np.exp(log_density)
    print("Estimated Density:")
    for point, dens in zip(example_points, density):
        print(f"Point: {point:.2f}, Density: {dens[0]:.5f}")

if __name__ == "__main__":
    main()
