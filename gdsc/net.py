import torch
import timm
import torch.nn as nn
import torchaudio as ta
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.parameter import Parameter

class SampleNet(nn.Module):
    def __init__(self, cfg):
        super(SampleNet, self).__init__()

        self.cfg = cfg
        self.n_classes = cfg.n_classes

        self.mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.window_size,
            win_length=cfg.window_size,
            hop_length=cfg.hop_size,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            pad=0,
            n_mels=cfg.mel_bins,
            power=cfg.power,
            normalized=False,
        )

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=cfg.top_db)
        self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)
        self.global_pool = GeM()
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="",
            in_chans=cfg.in_chans,
        )

        if "efficientnet" in cfg.backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.feature_info[-1]["num_chs"]

        self.head = nn.Linear(backbone_out, self.n_classes)

    def forward(self, x):
        x = self.wav2img(x)  # (bs, mel, time)
        x = self.backbone(x)
        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        logits = self.head(x.swapaxes(1, -1))

        return logits


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    # Generalized mean: https://arxiv.org/abs/1711.02512
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (self.__class__.__name__+ "(p="+ "{:.4f}".format(self.p.data.tolist()[0])+ ", eps="+ str(self.eps)+ ")")
# n_fft = 1024
# win_length = None
# hop_length = 512
# n_mels = 128
# mel_transform = torchaudio.transforms.MelSpectrogram(
#     sample_rate=sample_rate,
#     n_fft=n_fft,
#     win_length=win_length,
#     hop_length=hop_length,
#     center=True,
#     pad=0,
#     pad_mode="reflect",
#     power=2.0,
#     norm="slaney",
#     n_mels=n_mels,
#     mel_scale="htk",
# )
# mel_specs = []
# for wave in tqdm(wave_samples):
#     spec = mel_transform(wave)
#     mel_spectro = torch.zeros((n_mels, self.time_samp)).type(torch.FloatTensor)
#     mel_spectro[:, :spec.shape[2]] = spec[0]
#     mel_specs.append(mel_spectro)
# return torch.stack(mel_specs, dim=0)
