import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn
from src.model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


# configuration
Devide = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR = 3e-4  # Karparthy constant

# Dataset load
