import pytorch_lightning as pl
import pickle
import numpy as np
import torch
from torch.utils.data import StackDataset, Dataset, DataLoader, random_split
from torch.utils.data import Dataset, DataLoader, random_split

class RML2016DataModule(pl.LightningDataModule):
    def __init__(self, data_file: str, batch_size: int, is_10b: bool = False, seed: int = 42, n_workers:int = 8):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        self.seed = seed
        self.frame_size = 128
        self.n_rx = 1
        self.n_workers = n_workers

        if is_10b:
            self.classes = ['QPSK', 'PAM4', 'AM-DSB', 'GFSK', 'QAM64', 'QAM16', '8PSK', 'WBFM', 'BPSK', 'CPFSK']
        else:
            self.classes = ['QPSK', 'PAM4', 'AM-DSB', 'GFSK', 'QAM64', 'AM-SSB', 'QAM16', '8PSK', 'WBFM', 'BPSK', 'CPFSK']
            
        self.snrs = [-20, -18, -16, -14, -12, -10,  -8,  -6,  -4,  -2,   0,   2,   4,
            6,   8,  10,  12,  14,  16,  18]

    def prepare_data(self):            
        pass

    def setup(self, stage: str = None):
        print('Preprocessing Data...')
        f = pickle.load(open(self.data_file, 'rb'), encoding='latin')

        x = []
        y = []
        snr = []

        for i, c in enumerate(self.classes):
            for s in self.snrs:
                x1 = f[(c, s)]
                x.append(torch.from_numpy(x1))
                y.append(torch.ones(len(x1))*i)
                snr.append(torch.ones(len(x1))*s)

        x = torch.cat(x).mT.contiguous()
        y = torch.cat(y).type(torch.long)
        snr = torch.cat(snr)[:,None]

        x = torch.view_as_complex(x).unsqueeze(1)

        # print("Normalizing...")
        # Per-frame normalize to -1.0:1.0
        # TODO: try 0:1 scaling
        new_min, new_max = -1.0, 1.0
        x_max = torch.amax(torch.abs(x), axis=(1,2), keepdims=True) # farthest value from 0 in each frame
        scale = ((new_max - new_min) / (x_max*2))
        x *= scale

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
