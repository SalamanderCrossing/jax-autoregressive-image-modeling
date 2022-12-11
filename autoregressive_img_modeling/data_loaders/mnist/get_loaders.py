from torchvision.datasets import MNIST
import torch
from torch.utils import data
import numpy as np

def image_to_numpy(img):
        img = np.array(img, dtype=np.int32)
        img = img[..., None]  # Make image [28, 28, 1]
        return img

    # We need to stack the batch elements
    def numpy_collate(batch):
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple, list)):
            transposed = zip(*batch)
            return [numpy_collate(samples) for samples in transposed]
        else:
            return np.array(batch)

def get_loaders(train_batch_size:int, test_batch_size:int):
    # Transformations applied on each image => bring them into a numpy array
    # Note that we keep them in the range 0-255 (integers)

    DATASET_PATH = './data'
    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = MNIST(
        root=DATASET_PATH, train=True, transform=image_to_numpy, download=True
    )
    train_set, val_set = data.random_split(
        train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(42)
    )
    # Loading the test set
    test_set = MNIST(
        root=DATASET_PATH, train=False, transform=image_to_numpy, download=True
    )

    # We define a set of data loaders that we can use for various purposes
    train_loader = data.DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=numpy_collate,
        num_workers=8,
        persistent_workers=True,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        collate_fn=numpy_collate,
    )
    test_loader = data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        collate_fn=numpy_collate,
    )

    return train_loader, val_loader, test_loader
