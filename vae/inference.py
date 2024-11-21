import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from vae.model import VariationalAutoEncoder
from config.paths import IMAGES, STORAGE


class VAEInference:
    def __init__(self, model_path, input_dim, h_dim, z_dim, device):
        self.device = device
        self.model = VariationalAutoEncoder(input_dim, h_dim, z_dim).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.z_dim = z_dim

    def load_data(self, batch_size):
        test_dataset = datasets.MNIST(
            root="dataset/", train=False, transform=transforms.ToTensor(), download=True
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=True
        )
        return test_loader

    def inference(self, batch_size=10):
        test_loader = self.load_data(batch_size)
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(self.device).view(x.shape[0], -1)
                x_reconstructed, _, _ = self.model(x)
                self.plot_reconstructed_images(x, x_reconstructed)
                break

    def sample(self, num_samples=10):
        with torch.no_grad():
            z = torch.randn(num_samples, self.z_dim).to(self.device)
            samples = self.model.decode(z)
            self.plot_generated_images(samples)

    def plot_reconstructed_images(self, original, reconstructed):
        original = original.view(-1, 28, 28).cpu().numpy()
        reconstructed = reconstructed.view(-1, 28, 28).cpu().numpy()
        fig, axes = plt.subplots(2, len(original), figsize=(15, 3))
        for i in range(len(original)):
            axes[0, i].imshow(original[i], cmap="gray")
            axes[0, i].axis("off")
            axes[1, i].imshow(reconstructed[i], cmap="gray")
            axes[1, i].axis("off")
        plt.savefig(IMAGES / "reconstructed_images.png")
        plt.close()

    def plot_generated_images(self, samples):
        samples = samples.view(-1, 28, 28).cpu().numpy()
        fig, axes = plt.subplots(1, len(samples), figsize=(15, 3))
        for i in range(len(samples)):
            axes[i].imshow(samples[i], cmap="gray")
            axes[i].axis("off")
        plt.savefig(IMAGES / "generated_images.png")
        plt.close()


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = 784
    H_DIM = 200
    Z_DIM = 20
    MODEL_PATH = STORAGE / "trained_model.pth"

    inference = VAEInference(MODEL_PATH, INPUT_DIM, H_DIM, Z_DIM, DEVICE)
    inference.inference()
    inference.sample()
