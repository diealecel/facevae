from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path


def get_data_loader(data_loc, batch_size, shuffle = True):
    dataset = ImageFolder(
        root = data_loc,
        transform = ToTensor()
    )

    dataset = DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = 0,
        shuffle = shuffle
    )

    return dataset


def get_data_loader_with_paths(data_loc, batch_size, shuffle = True):
    dataset = ImageFolderWithPaths(
        root = data_loc,
        transform = ToTensor()
    )

    dataset = DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = 0,
        shuffle = shuffle
    )

    return dataset
