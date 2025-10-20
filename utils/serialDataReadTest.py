import matplotlib.pyplot as plt
from time import time
from multiprocessing import Pool

# 
idxs = np.arange(0,4000)
np.random.shuffle(idxs)

batch_size = 10

with h5py.File(DATA_PATH, "r") as f:
    total_serial_time = 0
    start = time()
    batch = []
    for idx in idxs[:batch_size]:
        start_read = time()
        var = f['data'][idx]
        batch.append(var)
        read_time = time()-start_read
        total_serial_time += read_time
        print(f'Im fetched in {read_time:.4f}s')

    batch = np.stack(batch, axis=0)
    print(f'Random batch of {batch_size} serially built in: {time()-start:.4f}s')
