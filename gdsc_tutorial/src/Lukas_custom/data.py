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
    def __init__(self, df, cfg, mode: str = 'train'):
        self.df = df
        self.cfg = cfg
        self.filepaths = df["path"].values
        self.labels = torch.zeros((df.shape[0], cfg.n_classes))
        self.weights = None
        self.mode = mode
        self.setup()
                        
        self.wave_transforms = Compose(
                [OneOf(
                    [NoiseInjection(p=1, max_noise_level=cfg.max_noise),
                     GaussianNoise(p=1, min_snr=cfg.min_snr, max_snr=cfg.max_snr),
                     PinkNoise(p=1, min_snr=cfg.min_snr, max_snr=cfg.max_snr)],
                    p=cfg.noise_prob),
                ]
            )

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idxs):
        wave, sample_rate = torchaudio.load(self.filepaths[idxs])

        wave = wave[0] # only one channel
        start = 0
        max_time = int(self.cfg.wav_crop_len*sample_rate)
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
                                  num_classes=self.cfg.n_classes).float()

class DataModule(pl.LightningDataModule):
    def __init__(self, 
                 cfg):
        super().__init__()
        self.cfg = cfg
        self.df_metadata = pd.read_csv(f'{cfg.data_path}/metadata.csv')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.cfg.include_val:
                self.train = AudioDataset(self.df_metadata, mode='train', cfg=self.cfg)
            else:
                self.train = AudioDataset( self.df_metadata[self.df_metadata['subset']=='train'], mode='train', cfg=self.cfg)

            self.val = AudioDataset(self.df_metadata[self.df_metadata['subset']=='validation'], mode='val', cfg=self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True, num_workers=self.cfg.num_workers, persistent_workers=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, shuffle=False, pin_memory=True, num_workers=self.cfg.num_workers, persistent_workers=True, collate_fn=collate_fn)
