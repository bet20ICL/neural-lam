import os
import torch

print("tasks per node", os.environ.get("SLURM_NTASKS_PER_NODE"))
print("tasks", os.environ.get("SLURM_NTASKS"))

os.environ["SLURM_NTASKS_PER_NODE"] = str(torch.cuda.device_count())

print("tasks per node", os.environ.get("SLURM_NTASKS_PER_NODE"))

if torch.cuda.is_available():
    device_name = "cuda"
    torch.set_float32_matmul_precision(
        "high"
    )  # Allows using Tensor Cores on A100s
    print(f"Devices: {torch.cuda.device_count()}")
    print("===== CUDA ENABLED =====")
    print("Using deterministic algorithms:", torch.are_deterministic_algorithms_enabled())
else:
    device_name = "cpu"

from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Define a minimal neural network model
class SimpleNN(pl.LightningModule):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer_1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

# Prepare data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST('', train=True, download=True, transform=transform)
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=64, num_workers=4)

# Initialize the model, trainer, and start training
model = SimpleNN()

trainer = Trainer(
    max_epochs=5,
    deterministic=True,
    strategy="ddp",
    accelerator=device_name, 
    # devices=-1,  # Use all available GPUs
)

trainer.fit(model, train_loader)