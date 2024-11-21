# Variational Autoencoder (VAE)

This repository provides an implementation of a Variational Autoencoder (VAE) using PyTorch. VAEs are a class of deep generative models designed to learn latent representations of data and generate new samples. This README introduces the fundamental concepts of VAEs, details the implementation provided, and guides you through the setup using Poetry.

---

## Run

for inference run `poetry run inference` to sample or reconstruct images.
The images will appear in `/images`

for training run `poetry run train`

## Variational Autoencoder (VAE): An Overview

A Variational Autoencoder is a type of autoencoder that models the underlying data distribution through latent variables sampled from a probabilistic space. Unlike traditional autoencoders, VAEs are generative models capable of producing new data similar to the training set.

### Key Components:

1. **Encoder:**

   - Maps the input data to a latent space represented by a mean vector (`μ`) and a standard deviation vector (`σ`).
   - The latent space is regularized using a Kullback-Leibler (KL) divergence term to approximate a normal distribution.

2. **Reparameterization Trick:**

   - Allows backpropagation through the stochastic sampling process.
   - `z = μ + σ * ε`, where `ε` is sampled from a standard normal distribution.

3. **Decoder:**
   - Maps the latent variable `z` back to the data space, reconstructing the input data.

### Loss Function:

The VAE loss comprises two components:

- **Reconstruction Loss:** Measures how well the decoded output matches the input.
- **KL Divergence:** Regularizes the latent space to follow a standard normal distribution.

---

## Code Structure

The implementation includes:

- **Encoder:** Processes input data into latent representations (`μ`, `σ`).
- **Decoder:** Reconstructs data from latent variables.
- **Reparameterization Trick:** Enables sampling from a latent space while maintaining differentiability.

### Model Overview

```python
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        # Encoder and decoder architecture
```

- `encode(x)`: Encodes input into `μ` and `σ`.
- `decode(z)`: Decodes latent vector `z` back into the input space.
- `forward(x)`: Encodes, reparameterizes, and decodes.

---

## Setup Guide

Follow these steps to set up and run the project.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Dependencies Using Poetry

This project uses Poetry for dependency management. Ensure you have Poetry installed on your system. To install Poetry, follow the [official guide](https://python-poetry.org/docs/#installation).

```bash
poetry install
```

### 3. Activate the Virtual Environment

```bash
poetry shell
```

### 4. Run the Script

```bash
python vae.py
```

---

## Example Usage

The script processes a random batch of data through the VAE. Below is a quick demonstration:

```python
if __name__ == "__main__":
    input_dim = 28 * 28  # for MNIST data
    batch_size = 4
    x = torch.randn(batch_size, input_dim)  # Randomly generated data

    vae = VariationalAutoEncoder(input_dim=input_dim)
    x_reconstructed, mu, sigma = vae(x)

    print(f"Reconstructed Shape: {x_reconstructed.shape}")
    print(f"Latent Mean Shape: {mu.shape}")
    print(f"Latent Sigma Shape: {sigma.shape}")
```

### Expected Output:

```plaintext
Reconstructed Shape: torch.Size([4, 784])
Latent Mean Shape: torch.Size([4, 20])
Latent Sigma Shape: torch.Size([4, 20])
```

---

## Understanding the Architecture

### Encoder:

- Input Layer: Flattens the input image (`28x28 → 784`).
- Hidden Layer: Encodes features into a hidden representation.
- Outputs: Latent mean (`μ`) and standard deviation (`σ`).

### Decoder:

- Input: Sampled latent variable `z`.
- Hidden Layer: Maps latent variable back to the feature space.
- Output Layer: Reconstructs the original image.

---

## Contribution Guidelines

1. Fork the repository and create a branch for your feature.
2. Follow best practices for Python development.
3. Submit a pull request for review.

---

## References

- [Primary source](https://www.youtube.com/watch?v=VELQT1-hILo&t=1117s&ab_channel=AladdinPersson)
- [Kingma & Welling, 2014 - Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

# VAE
