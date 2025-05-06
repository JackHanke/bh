from torch.utils.data import Dataset, DataLoader, random_split

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
def make_prediction_dataloaders(batch_size: int = 8):
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
