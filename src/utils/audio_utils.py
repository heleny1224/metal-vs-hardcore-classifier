"""
Audio utilities for loading and processing audio files
"""
import math
import random
import torch
import torchaudio
from torchaudio import transforms


class AudioUtil:
    """Utility class for audio processing"""
    
    @staticmethod
    def open(audio_file):
        """Load audio file"""
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)
    
    @staticmethod
    def rechannel(aud, new_channel):
        """Change number of channels"""
        sig, sr = aud
        if sig.shape[0] == new_channel:
            return aud
        if new_channel == 1:
            resig = sig[:1, :]
        else:
            resig = torch.cat([sig, sig])
        return (resig, sr)
    
    @staticmethod
    def resample(aud, newsr):
        """Resample audio"""
        sig, sr = aud
        if sr == newsr:
            return aud
        
        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if num_channels > 1:
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])
        return (resig, newsr)
    
    @staticmethod
    def pad_trunc(aud, max_ms):
        """Pad or truncate audio to fixed length"""
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if sig_len > max_len:
            # Truncate the signal to the given length
            sig = sig[:,:max_len]
        elif sig_len < max_len:
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            
            sig = torch.cat((pad_begin, sig, pad_end), 1)
            
        return (sig, sr)
    
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        """Create Mel spectrogram"""
        sig, sr = aud
        top_db = 80
        
        spec = transforms.MelSpectrogram(
            sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        
        return spec