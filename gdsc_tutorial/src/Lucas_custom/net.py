import torch
import timm
import torch.nn as nn
import torchaudio as ta
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from .utils import NormalizeMelSpec, min_max_norm


class SimpleCNN(nn.Module):
    def __init__(self, cfg, init_backbone=True):
        super(SimpleCNN, self).__init__()

        self.cfg = cfg
        self.n_classes = cfg.n_classes

        self.mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.window_size,
            hop_length=cfg.hop_length,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            n_mels=cfg.n_mels,
            power=cfg.power,
            normalized=cfg.mel_normalized,
        )

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=cfg.top_db)
        self.wav2img = torch.nn.Sequential(self.mel_spec, 
                                           self.amplitude_to_db)
        if init_backbone:
            self.backbone = timm.create_model(
                cfg.backbone,
                pretrained=cfg.pretrained,
                num_classes=cfg.n_classes,
                in_chans=cfg.in_chans,
            )

    def forward(self, x):
        # (bs, channel, time)
        x = x[:, None, :] # one channel for CNN input
        x = self.wav2img(x)  # (bs, channel, mel, time)
        
        if self.cfg.minmax_norm:
            x = min_max_norm(x, min_val=self.cfg.min, max_val=self.cfg.max) 
        
        logits = self.backbone(x)
        return logits