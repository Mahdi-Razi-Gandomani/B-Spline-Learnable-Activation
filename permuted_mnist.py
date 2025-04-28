import torch
import numpy as np
import random
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Create permuted MNIST dataset
class PermutedMNIST(Dataset):
    def __init__(self, task_id, train=True, permute_seed=None):
        self.data = datasets.MNIST(root='./data', train=train, download=True)
        np.random.seed(permute_seed if permute_seed is not None else task_id)
        self.permutation = np.random.permutation(28 * 28)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = np.array(image).astype(np.float32).reshape(-1) / 255.0
        image = image[self.permutation]
        return torch.tensor(image), label

# Get dataloaders for each task
def get_permuted_mnist_tasks(num_tasks=10, batch_size=64):
    tasks = []
    for task_id in range(num_tasks):
        train_set = PermutedMNIST(task_id=task_id, train=True)
        test_set = PermutedMNIST(task_id=task_id, train=False)
        tasks.append({
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
            'test': DataLoader(test_set, batch_size=batch_size)
        })
    return tasks
