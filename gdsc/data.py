import os
from glob import glob

import torchaudio
import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.audio_files = glob(rf'{self.data_dir}\*\*.ogg')
        self.batch_size = batch_size

    def setup(self, stage: str):
        wave_sample = []
        for filename in tqdm(self.audio_files):
            wave_sample.append(torchaudio.load(filename))
        mel_spec = self.preprocess(wave_sample)
        # self.train, self.val, self.test = random_split(mnist_full, [55000, 5000])

    def preprocess(self, wave_sample):
        n_fft = 1024
        win_length = None
        hop_length = 512
        n_mels = 128

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="htk",
        )
        mel_spec = []
        for wave in tqdm(wave_sample):
            mel_spec.append(mel_spectrogram(wave))
        return mel_spec

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass