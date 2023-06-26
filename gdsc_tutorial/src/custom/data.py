import os
from glob import glob

import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import torchaudio
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import one_hot
from sklearn.model_selection import train_test_split

from torch_audiomentations import AddColoredNoise, ApplyImpulseResponse

from .utils import Compose, OneOf, NoiseInjection, GaussianNoise, PinkNoise


def collate_fn(batch):
    return {
      'wave': torch.stack([x['wave'] for x in batch]),
      'labels': torch.stack([x['labels'] for x in batch])
    }


class AudioDataset(Dataset):
    def __init__(self, df, max_time: int = 15, mode: str = 'train', n_classes: int = 66):
        self.df = df
        self.filepaths = df["path"].values
        self.labels = torch.zeros((df.shape[0], n_classes))
        self.weights = None
        self.n_classes = n_classes
        self.max_time = max_time
        self.mode = mode
        self.setup()
                        
        self.wave_transforms = Compose(
                [OneOf(
                    [NoiseInjection(p=1, max_noise_level=0.04),
                     GaussianNoise(p=1, min_snr=5, max_snr=20),
                     PinkNoise(p=1, min_snr=5, max_snr=20)],
                    p=0.2),
                ]
            )

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idxs):
        wave, sample_rate = torchaudio.load(f'{self.filepaths[idxs]}')
        wave = wave[0] # only one channel
        start = 0
        max_time = int(self.max_time*sample_rate)
        if wave.shape[0] <= max_time:
            pad = max_time - wave.shape[0]
            wave = torch.from_numpy(np.pad(wave, (0, pad)))
        else:
            if self.mode == 'train' or self.mode == 'test':
                start = np.random.randint(0, wave.shape[0] - max_time)
            
        wave = wave[start:start+max_time]
        
        if self.mode == 'train':
            wave = self.wave_transforms(wave, sample_rate)
        
        sample = {'wave': wave, 
                  'labels': self.labels[idxs],
                 }
        return sample

    def setup(self):
        if self.mode == 'train' or self.mode == 'val':
            self.labels = one_hot(torch.tensor(self.df['label'].values), 
                                  num_classes=self.n_classes).float()
            self.weights = 1/self.labels.sum(axis=0)


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", 
                 batch_size: int = 32, 
                 max_time: int = 15, 
                 n_workers: int = 4,
                 include_val: bool = False):
        super().__init__()
        self.df_metadata = pd.read_csv(f'{data_dir}/metadata.csv')
        self.batch_size = batch_size
        self.max_time = max_time
        self.n_workers = n_workers
        self.include_val = include_val

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.include_val:
                self.train = AudioDataset(self.df_metadata, mode='train', max_time=self.max_time)
            else:
                self.train = AudioDataset( self.df_metadata[self.df_metadata['subset']=='train'], mode='train', max_time=self.max_time)

            self.val = AudioDataset(self.df_metadata[self.df_metadata['subset']=='validation'], mode='val', max_time=self.max_time)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, pin_memory=True, 
                          num_workers=self.n_workers, persistent_workers=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, pin_memory=True, 
                          num_workers=self.n_workers, persistent_workers=True, collate_fn=collate_fn)
