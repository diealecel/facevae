from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def get_data_loader(data_loc, batch_size):
    train_dataset = ImageFolder(
        root = data_loc,
        transform = ToTensor()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        num_workers = 0,
        shuffle = True
    )

    return train_loader
