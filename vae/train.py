import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from vae.model import VariationalAutoEncoder
from config.paths import IMAGES, STORAGE


class VAETrainer:
    def __init__(self, input_dim, h_dim, z_dim, batch_size, lr, num_epochs, device):
        self.device = device
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = VariationalAutoEncoder(input_dim, h_dim, z_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss(reduction="sum")
        self.train_loader, self.val_loader = self.load_data()

    def load_data(self):
        dataset = datasets.MNIST(
            root="dataset/", train=True, transform=transforms.ToTensor(), download=True
        )
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.batch_size, shuffle=False
        )
        return train_loader, val_loader

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        loop = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Training",
        )
        for i, (x, _) in loop:
            x = x.to(self.device).view(x.shape[0], self.input_dim)
            x_reconstructed, mu, sigma = self.model(x)
            loss = self.compute_loss(x, x_reconstructed, mu, sigma)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        return train_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            loop = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                desc="Validation",
            )
            for i, (x, _) in loop:
                x = x.to(self.device).view(x.shape[0], self.input_dim)
                x_reconstructed, mu, sigma = self.model(x)
                loss = self.compute_loss(x, x_reconstructed, mu, sigma)
                val_loss += loss.item()
                loop.set_postfix(loss=loss.item())
        return val_loss / len(self.val_loader)

    def compute_loss(self, x, x_reconstructed, mu, sigma):
        reconstruction_loss = self.loss_fn(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        return reconstruction_loss + kl_div

    def plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(
            range(1, self.num_epochs + 1),
            train_losses,
            label="Training Loss",
            marker="o",
        )
        plt.plot(
            range(1, self.num_epochs + 1),
            val_losses,
            label="Validation Loss",
            marker="o",
        )
        plt.title("Training and Validation Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(IMAGES / "train_val_loss_plot.png")
        plt.show()

    def save_model(self, path):
        print("saving model")
        torch.save(self.model.state_dict(), path)

    def train(self):
        train_losses = []
        val_losses = []
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )
        self.save_model(STORAGE / "trained_model.pth")
        self.plot_losses(train_losses, val_losses)


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = 784
    H_DIM = 200
    Z_DIM = 20
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    LR = 3e-4

    trainer = VAETrainer(INPUT_DIM, H_DIM, Z_DIM, BATCH_SIZE, LR, NUM_EPOCHS, DEVICE)
    trainer.train()


if __name__ == "__main__":
    main()
