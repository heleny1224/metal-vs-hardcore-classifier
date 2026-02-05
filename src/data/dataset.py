"""
PyTorch Dataset for audio classification
"""
import torch
from torch.utils.data import Dataset
from src.utils.audio_utils import AudioUtil


class AudioDataset(Dataset):
    """Dataset for audio classification"""
    
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.class_names = sorted(dataframe['Genre'].unique())
        self.class_to_index = {
            class_name: i for i, class_name in enumerate(self.class_names)
        }
        self.file_list = [
            (row['Path'], self.class_to_index[row['Genre']]) 
            for index, row in dataframe.iterrows()
        ]
        self.transform = transform
        
        # Audio parameters
        self.sr = 44100
        self.duration = 5500  # ms
        self.channel = 2
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        audio_file, class_id = self.file_list[idx]
        
        # Process audio
        aud = AudioUtil.open(audio_file)
        resample = AudioUtil.resample(aud, self.sr)
        rechannel = AudioUtil.rechannel(resample, self.channel)
        equal_dur = AudioUtil.pad_trunc(rechannel, self.duration)
        melspectrogram = AudioUtil.spectro_gram(equal_dur)

        return melspectrogram, class_id
    
    def get_class_names(self):
        return self.class_names


def create_data_loader(dataframe, batch_size=16, shuffle=True):
    """Create DataLoader from dataframe"""
    audio_dataset = AudioDataset(dataframe)
    data_loader = torch.utils.data.DataLoader(
        audio_dataset, batch_size=batch_size, shuffle=shuffle
    )
    return data_loader