import torch
import numpy as np

from utils.utils import read_dump_util
from torch.utils.data import Dataset, DataLoader, random_split

# turn dumps on disk into single stacked tensor 
def tensorize_dumps(dumps:list, log_non_negs: bool = False):
    data = []
    for dump in dumps:
        _, dump_dict = read_dump_util(dump=dump)

        # rd(dump)
        if log_non_negs:
            rho = np.log10(dump_dict['rho'])
            ug = np.log10(dump_dict['ug'])
        else:
            rho = dump_dict['rho']
            ug = dump_dict['ug']
        uu = dump_dict['uu']
        B = dump_dict['B']

        rho_tensor = torch.tensor(rho).squeeze(2).unsqueeze(0)
        ug_tensor = torch.tensor(ug).squeeze(2).unsqueeze(0)
        uu_tensor = torch.tensor(uu[1:4]).squeeze(3)
        B_tensor = torch.tensor(B[1:4]).squeeze(3)
        
        data_tensor = torch.cat((rho_tensor, ug_tensor, uu_tensor, B_tensor), dim=0)
        data.append(data_tensor.unsqueeze(0))

    data = torch.cat(data, dim=0)
    return data

# tensorize dumps 
def tensorize_dumps_sc(dumps:list):

# turn list of latent tensors into single tensor for training
def tensorize_latents(latents:list[torch.Tensor]):
    data = torch.cat(latents, dim=0)
    return data

# Dataset for predicting the next frame
class PredictionDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)-1

    def __getitem__(self, idx):
        data = self.dataset[idx]
        label = self.dataset[idx+1]
        return data, label

# 
def make_prediction_dataloaders(data: torch.Tensor, batch_size: int = 8):
    loaded_data = PredictionDataset(data)
    train_size = int(0.7 * len(loaded_data))
    val_size = int(0.15 * len(loaded_data))
    test_size = len(loaded_data) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(loaded_data, [train_size, val_size, test_size])

    # create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# dataset for autoencoders
class EncodingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)-1

    def __getitem__(self, idx):
        data = self.dataset[idx]
        label = self.dataset[idx]
        return data, label

# 
def make_encoding_dataloaders(data, batch_size: int = 8):
    loaded_data = EncodingDataset(data)
    train_size = int(0.7 * len(loaded_data))
    val_size = len(loaded_data) - train_size
    train_dataset, val_dataset = random_split(loaded_data, [train_size, val_size])

    # create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
