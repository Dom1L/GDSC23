import random
import json
import math

import torch
import torch.nn as nn
import numpy as np
import colorednoise as cn
from tqdm import tqdm


def get_max_amplitude_window_index(path: str,
                                   waveform = None,
                                   samplerate = None,
                                   window_length_sec = 5,
                                   scan_param = 50, 
                                   verbose = True):
    '''
    Returns index of waveform that starts the window of length window_length_sec*samplerate, with the highest summed amplitude.
    only scans at certain scan intervals, to speed up the calculation

    Args:
        path (str): path to data, as in torchaudio.load
        window_length_sec: window length to calculate sum over absolute amplitudes
        scan_param: samplerate should be divisible by scan_param
        verbose (bool): to print return index in seconds

    Returns:
        max_index (int): start index of window with max amplitudes 
    '''
    
    if waveform is None:
        waveform, samplerate =  torchaudio.load(path)
    
    print('type equals:', type(waveform))
    print('size equals:', waveform.shape)
    #print(waveform[0].shape)
    waveform_length = waveform[0].numpy().shape[0]
    window_length = math.floor(window_length_sec * samplerate)
    
    if window_length >= waveform_length:
        return 0
    
    #divide available waveform length by scan_param, to construct scan array
    scan_length = math.floor((waveform_length-window_length)/scan_param)
    
    max_sum = 0
    max_index = 0
    
    #in every scan interval: calculate sum over window and save max
    for x in range(scan_length):
        tmp = np.sum(abs(waveform[0].numpy()[x*scan_param:x*scan_param+window_length]))
        if tmp > max_sum:
            max_sum = tmp
            max_index = x*scan_param
    
    if verbose:
        print('window starts at:', max_index/samplerate, 'seconds')
    return max_index


def get_min_max(cfg, dm, model):
    dm = dm(cfg=cfg)
    model = model(cfg, init_backbone=False)
    dm.setup()
    dm.train.mode = 'val'
    total_max = -1e3
    total_min = 1e3
    print('Gather min max statistics:')
    for batch in tqdm(dm.train_dataloader()):
        spec = model.wav2img(batch['wave'][:, None, :])
        if spec.max() > total_max:
            total_max = spec.max()
        if spec.min() < total_min:
            total_min = spec.min()
    return total_min, total_max


def min_max_norm(x, min_val=-39.4655, max_val=53.6203):
    return (x - min_val) / (max_val - min_val)


def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


def get_state_dict(sd_fp):
    sd = torch.load(sd_fp, map_location="cpu")['state_dict']
    sd = {k.replace("model.", ""):v for k,v in sd.items()}
    sd.pop("loss_fn.weight")
    return sd

        
class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, sr):
        for trns in self.transforms:
            y = trns(y, sr)
        return y


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr):
        if self.always_apply:
            return self.apply(y, sr=sr)
        else:
            if np.random.rand() < self.p:
                return self.apply(y, sr=sr)
            else:
                return y

    def apply(self, y: np.ndarray, **params):
        raise NotImplementedError


class OneOf(Compose):
    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms)
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, y: np.ndarray, sr):
        data = y
        if self.transforms_ps and (random.random() < self.p):
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            data = t(y, sr)
        return data

    
class NoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)

    def apply(self, y: np.ndarray, **params):
        noise_level = torch.distributions.Uniform(*self.noise_level).sample([1])

        noise = torch.randn(len(y))
        augmented = y + noise * noise_level
        return augmented


class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = torch.distributions.Uniform(self.min_snr, self.max_snr).sample([1])
        a_signal = torch.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = torch.randn(len(y))
        a_white = torch.sqrt(white_noise ** 2).max()
        augmented = y + white_noise * 1 / a_white * a_noise
        return augmented


class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = torch.distributions.Uniform(self.min_snr, self.max_snr).sample([1])
        a_signal = torch.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = torch.from_numpy(cn.powerlaw_psd_gaussian(1, len(y)))
        a_pink = torch.sqrt(pink_noise ** 2).max()
        augmented = y + (pink_noise * 1 / a_pink * a_noise).type(y.dtype)
        return augmented
   

class Mixup(nn.Module):
    def __init__(self, mix_beta):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight

        
class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        norm_min, norm_max = Xstd.min(-1)[0].min(-1)[0], Xstd.max(-1)[0].max(-1)[0]
        fix_ind = (norm_max - norm_min) > self.eps * torch.ones_like(
            (norm_max - norm_min)
        )
        V = torch.zeros_like(Xstd)
        if fix_ind.sum():
            V_fix = Xstd[fix_ind]
            norm_max_fix = norm_max[fix_ind, None, None]
            norm_min_fix = norm_min[fix_ind, None, None]
            V_fix = torch.max(
                torch.min(V_fix, norm_max_fix),
                norm_min_fix,
            )
            # print(V_fix.shape, norm_min_fix.shape, norm_max_fix.shape)
            V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
            V[fix_ind] = V_fix
        return V
