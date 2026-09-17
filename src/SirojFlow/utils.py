import numpy as np

class DataLoader:
    """ 
        ```
        x: input, shape (samples, features) 
        y: label
        batch_size: put an integer value to create batches of that size
        shuffle=True : shuffles the dataset/ batches each epochs
        drop_last=True : drops incomplete batch if any
        ```
        
        ### Usage:

        ```loader = DataLoader(x, y, batch_size=32, shuffle=True, drop_last=False)
        for x_batch, y_batch in loader:
            ...
        ```
    """
 
    def __init__(self, x, y, batch_size=None, shuffle=False, drop_last=False):
        x = np.asarray(x)
        y = np.asarray(y)
 
        if x.ndim == 1:
            x = x.reshape(-1, 1)
 
        if y.ndim == 1:
            y = y.reshape(-1, 1)
 
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Number of sample mismatch: {x.shape[0]} vs {y.shape[0]}")
 
        n_samples = x.shape[0]
 
        if batch_size is None:
            batch_size = n_samples

        else:
            if type(batch_size) != int:
                raise TypeError(f"Batch size must be an integer")
            if batch_size <= 0 or batch_size > n_samples:
                raise ValueError(f"Batch size {batch_size} must be between 1 and {n_samples}")

        if type(shuffle) != bool or type(drop_last) != bool:
            raise TypeError("This much be a boolean 'True' or 'False'")
        self.x = x.copy()
        self.y = y.copy()
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
 
    def __len__(self):
        if self.drop_last:
            return self.n_samples // self.batch_size
        return int(np.ceil(self.n_samples / self.batch_size))
 
    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(self.n_samples)
        else:
            indices = np.arange(self.n_samples)
 
        for i in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
 
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
 
            yield self.x[batch_idx], self.y[batch_idx]
 
    def __call__(self):
        """Returns all batches as a list."""
        return list(self) 