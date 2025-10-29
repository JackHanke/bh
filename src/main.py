
from batching import custom_batcher


if __name__ == '__main__':
    # get indexes for training data
    train_idxs, valid_idxs = custom_batcher(
        batch_size=batch_size,
        num_dumps=num_dumps,
        split = 0.8,
        seed=1,
        start=start_dump,
        end=end_dump,
    )

