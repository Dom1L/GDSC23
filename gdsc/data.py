import os
from glob import glob

import numpy as np
from tqdm import tqdm
import torch
import torchaudio
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class AudioDataset(Dataset):
    def __init__(self, filepaths):
        self.audio_files = filepaths
        self.wave_samples = None
        self.labels = None
        self.n_classes = 8
        self.max_time = 8000
        self.setup()

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idxs):
        # we only have one sample rn, so no indexing
        return self.wave_samples, self.labels

    def setup(self):
        labels = torch.zeros((self.n_classes))
        wave, sample_rate = torchaudio.load(self.audio_files[0])
        # Some pseudo labels for prototyping
        labels[0] += 1
        self.wave_samples = wave
        self.labels = labels


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.audio_files = np.array(glob(rf'{self.data_dir}\*\*.ogg'))
        self.batch_size = batch_size
        self.dims = None
        self.n_classes = None

    def setup(self, stage=None, seed: int = 42, subsample: int = 10):
        all_ids = np.arange(len(self.audio_files))
        if subsample:
            np.random.seed(seed)
            np.random.shuffle(all_ids)
            all_ids = all_ids[:subsample]
        train_ids, self.test_ids = train_test_split(all_ids, train_size=0.8, test_size=0.2)
        self.train_ids, self.val_ids = train_test_split(train_ids, train_size=0.9, test_size=0.1, random_state=seed)

        if stage == 'fit' or stage is None:
            self.train = AudioDataset(self.audio_files[self.train_ids])
            self.val = AudioDataset(self.audio_files[self.val_ids])

        if stage == 'test' or stage is None:
            self.test = AudioDataset(self.audio_files[self.test_ids])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass