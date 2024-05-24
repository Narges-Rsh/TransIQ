import pytorch_lightning as pl
import h5py
import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split, StackDataset


class SimpleHDF5DataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, batch_size, seed: int = 42, n_rx=1, n_workers:int = 8):
        super().__init__()
        self.dataset_path = os.path.expanduser(dataset_path)
        self.batch_size = batch_size
        self.frame_size = 1024
        self.transforms = []
        self.seed = seed
        self.n_rx=n_rx
        self.n_workers = n_workers
        
       
        self.classes = ['bpsk', 'qpsk', '8psk', 'dqpsk', 'msk', '16qam', '64qam', '256qam']
        self.ds_train, self.ds_val, self.ds_test = [], [], []

    def prepare_data(self): 
        pass

    def setup(self, stage: str = None):
        if not len(self.ds_train) or not len(self.ds_val) or not len(self.ds_test):
            print('Preprocessing Data...')
            with h5py.File(self.dataset_path, "r") as f:
                x = torch.from_numpy(f['x'][()][:,:self.n_rx])
                y = torch.from_numpy(f['y'][()]).to(torch.long)
                snr = torch.from_numpy(f['P_rx'][()][:,:self.n_rx])
            
            
            snr = snr.flatten(1)
            ds_full = StackDataset(x=x, y=y, snr=snr)
            print(f"Dataset size: {len(ds_full)}")

            self.ds_train, self.ds_val, self.ds_test = random_split(ds_full, [0.6, 0.2, 0.2], generator = torch.Generator().manual_seed(self.seed))

    def train_dataloader(self) -> DataLoader:
        return self._data_loader(self.ds_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.ds_val, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        return self._data_loader(self.ds_test, shuffle=False)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.n_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
            generator=torch.Generator().manual_seed(self.seed)
        )
