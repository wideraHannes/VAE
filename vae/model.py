import torch
from torch import nn
import numpy as np

""" 
Input Dimension:
- Mnist: 28x28 -> 784


Input -> hidden dim -> mean,std -> Parameterisation trick -> Decoder -> Output img

"""


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        # encoder
        ## Input -> hidden dim -> mean,std -> Parameterisation trick
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(
            h_dim, z_dim
        )  # KL divergence Pushes this layer to learn Normal Dist
        self.hid_2sigma = nn.Linear(
            h_dim, z_dim
        )  # KL divergence Pushes this layer to learn Normal Dist

        # decoder
        ## -> Decoder -> Output img
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        # q_phi(z|x)
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        # p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))  # normalize for mnist

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return (
            x_reconstructed,
            mu,
            sigma,
        )  # need all 3 for the loss function KL divergence and Reconstruction loss


if __name__ == "__main__":
    input_dim = 28 * 28  # flattend images
    x = torch.randn(4, input_dim)  # batch_size x image_size
    vae = VariationalAutoEncoder(input_dim=input_dim)
    x_reconstructed, mu, sigma = vae(x)

    print(x_reconstructed.shape)
    print(sigma.shape)
    print(mu.shape)
