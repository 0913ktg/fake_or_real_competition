from src.data.imgdataset import SampleDataset
from torch.utils.data import Subset, DataLoader

import torch


def create_datasets():
    dataset = SampleDataset()

    # train, val, test split
    len_data = len(dataset)
    print("data length: ", len_data)
    train_separator = int(len_data * 0.9)
    val_separator = train_separator + int(len_data * 0.05)

    train_dataset = Subset(dataset, range(0, train_separator))
    val_dataset = Subset(dataset, range(train_separator, val_separator))
    test_dataset = Subset(dataset, range(val_separator, len_data))

    return train_dataset, val_dataset, test_dataset


def initiate_loaders(train_dataset, val_dataset, test_dataset, env):
    train_dataloader = DataLoader(
        train_dataset, batch_size=env.batch_size, shuffle=False, num_workers=2
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=int(env.batch_size / 8), shuffle=False
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def instantiate_model(config):
    MODEL_DICT = {}
    return MODEL_DICT[config["model"]](config)


def test():
    pass
