import torch
import torch.nn as nn
import scipy.signal
import numpy as np

# Pre-emphasis filter class (for use as a layer)
class PreEmphasis(nn.Module):
    def __init__(self, alpha=0.97):
        super(PreEmphasis, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.cat((x[:, :1], x[:, 1:] - self.alpha * x[:, :-1]), dim=-1)

# Mel-spectrogram computation class without torchaudio
class MelSpectrogramNoTorchAudio(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                 f_min=20, f_max=7600, n_mels=80):
        super(MelSpectrogramNoTorchAudio, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels

        # Pre-emphasis filter
        self.pre_emphasis = PreEmphasis()

        # Mel filterbank creation
        self.mel_filters = self.mel_filterbank(n_fft, n_mels, sample_rate, f_min, f_max)

    def mel_filterbank(self, n_fft, n_mels, sample_rate, f_min, f_max):
        # Mel filterbank creation
        mel_points = np.linspace(self.hz_to_mel(f_min), self.hz_to_mel(f_max), n_mels + 2)
        hz_points = self.mel_to_hz(mel_points)

        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        filters = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(1, n_mels + 1):
            filters[i - 1, bin_points[i - 1]:bin_points[i]] = \
                (np.linspace(0, 1, bin_points[i] - bin_points[i - 1]))
            filters[i - 1, bin_points[i]:bin_points[i + 1]] = \
                (np.linspace(1, 0, bin_points[i + 1] - bin_points[i]))

        return torch.tensor(filters, dtype=torch.float32)

    def hz_to_mel(self, hz):
        return 1127 * np.log(1 + hz / 700)

    def mel_to_hz(self, mel):
        return 700 * (np.exp(mel / 1127) - 1)

    def forward(self, x):
        # Apply pre-emphasis
        x = self.pre_emphasis(x)

        # Perform STFT
        stft_result = torch.stft(x, self.n_fft, self.hop_length, self.win_length, return_complex=True)
        magnitude = torch.abs(stft_result)  # Magnitude of STFT

        # Apply Mel filterbank to the magnitude
        mel_spec = torch.matmul(self.mel_filters, magnitude)

        return mel_spec

# Example usage in a Sequential model

model = nn.Sequential(
    MelSpectrogramNoTorchAudio(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                               f_min=20, f_max=7600, n_mels=80)
)

# Example input: random audio signal (batch_size=1, audio_length=16000)
audio = torch.randn(1, 16000)  # Example input with 1 sample of length 16000

# Compute Mel-spectrogram by passing audio through the model
mel_spectrogram = model(audio)
print(mel_spectrogram.shape)  # Output shape: (n_mels, num_frames)
