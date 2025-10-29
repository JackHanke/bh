import numpy as np
import h5py

# 
def custom_batcher(
        batch_size: int, 
        num_dumps: int, 
        split: float = 0.8, 
        seed: int = None,
        start: int = None,
        end: int = None,
    ):
    # randomize what data is trained on
    if seed is not None: np.random.seed(seed)
    # randomize data
    if start is None and end is None:
        indexes = np.arange(num_dumps) # 0 to num_dumps
    else:
        # if only training on some portion of dumps, use line below:
        indexes = np.arange(start=start, stop=end) # start to end
    
    np.random.shuffle(indexes)
    # get split
    split_idx = round(len(indexes)*(split))
    # split indexes and return
    train_indexes = indexes[:split_idx]
    validation_indexes = indexes[split_idx:]
    return train_indexes, validation_indexes

# 
def construct_batch(batch_indexes: list[int], data_path: str):
    # 
    with h5py.File(data_path, "r") as f:
        # get the axis the specific dump index is stored at on disk
        
        batch = []
        for idx in batch_indexes:

            # get data
            batch.append(f['data'][idx][0])
        
    return np.stack(batch, axis=0)


