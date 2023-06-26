import torch
import timm
import torch.nn as nn
import torchaudio as ta
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.parameter import Parameter



class SimpleCNN(nn.Module):
    def __init__(self, cfg):
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
            normalized=True,
        )

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=cfg.top_db)
        self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=cfg.n_classes,
            in_chans=cfg.in_chans,
        )

    def forward(self, x):
        x = self.wav2img(x)  # (bs, mel, time)
        
#         if self.training:
#             x = self.spectra_transforms(x)    
        
        x = x[:, None, :, :] # one channel for CNN input
        logits = self.backbone(x)
        return logits