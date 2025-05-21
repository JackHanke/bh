import numpy as np

# returns random selection of dump incides
def custom_batcher(
        batch_size: int, 
        num_dumps: int, 
        split: float = 0.8, 
        seed: int = None
    ):
    # randomize what data is trained on
    if seed is not None: np.random.seed(seed)
    # randomize data
    indexes = np.arange(num_dumps)
    # TODO if only training on some portion of dumps, use line below:
    # indexes = np.arange(start = start_dump, end=end_dump)
    
    np.random.shuffle(indexes)
    # get split
    split_idx = round(num_dumps*(split))
    # split indexes and return
    train_indexes = indexes[:split_idx]
    validation_indexes = indexes[split_idx:]
    return train_indexes, validation_indexes

# turn relevant global variables into single tensor
# dim for reduced data is (batch_size=1, channels=8, depth=224, width=48, height=96)
def tensorize_globals(rho: np.array, ug: np.array, uu: np.array, B: np.array):
    rho_tensor = torch.tensor(rho)[0].unsqueeze(0)
    ug_tensor = torch.tensor(ug)[0].unsqueeze(0)
    uu_tensor = torch.tensor(uu[1:4]).squeeze(1)
    B_tensor = torch.tensor(B[1:4]).squeeze(1)
    # tensorize
    data_tensor = torch.cat((rho_tensor, ug_tensor, uu_tensor, B_tensor), dim=0)
    return data_tensor.unsqueeze(0)


