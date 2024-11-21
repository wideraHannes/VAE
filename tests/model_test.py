import pytest
import torch
from vae.models import VariationalAutoEncoder


@pytest.fixture
def vae():
    input_dim = 28 * 28  # flattened images
    return VariationalAutoEncoder(input_dim=input_dim)


def test_vae_output_shapes(vae):
    input_dim = 28 * 28
    x = torch.randn(4, input_dim)  # batch_size x image_size
    x_reconstructed, mu, sigma = vae(x)

    assert x_reconstructed.shape == (4, input_dim), "Reconstructed image shape mismatch"
    assert mu.shape == (4, vae.hid_2mu.out_features), "Mu shape mismatch"
    assert sigma.shape == (4, vae.hid_2sigma.out_features), "Sigma shape mismatch"


def test_vae_forward_pass(vae):
    input_dim = 28 * 28
    x = torch.randn(4, input_dim)  # batch_size x image_size
    x_reconstructed, mu, sigma = vae(x)

    assert isinstance(x_reconstructed, torch.Tensor), "Output is not a tensor"
    assert isinstance(mu, torch.Tensor), "Mu is not a tensor"
    assert isinstance(sigma, torch.Tensor), "Sigma is not a tensor"
    assert (
        x_reconstructed.min() >= 0 and x_reconstructed.max() <= 1
    ), "Reconstructed image values are not in range [0, 1]"


if __name__ == "__main__":
    pytest.main()
