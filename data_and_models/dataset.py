from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torchaudio.transforms as T
import torchaudio
import torch
import pandas as pd
import os

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
   
def max_lengths():
    metadata = pd.read_csv(f'./data_and_models/data/train/metadata/expanded_metadata.csv')
    if os.path.isfile(f'./data_and_models/data/train/metadata/max_length.txt') == False:
        with open(f'./data_and_models/data/train/metadata/max_length.txt', 'w', encoding='utf-8') as logfile:
            lengths = []
            for i in range(len(metadata)):
                if metadata.loc[i, 'length'] == 'long':
                    file_path = f"./data_and_models/data/train/audio/transformed/{metadata.loc[i, 'folder']}/{metadata.loc[i, 'chapter num']:03d}.wav"
                    signal, sample_rate = torchaudio.load(file_path)
                    lengths.append(signal.size(1))
            logfile.write(str(max(lengths)))
        return int(max(lengths))
    else:
        with open(f'./data_and_models/data/train/metadata/max_length.txt', 'r') as logfile:
            for num in logfile.readlines():
                return int(num)
   
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ReciterDataset(Dataset):
    def __init__(self, transformation):
        segments_metadata_file = './data_and_models/data/train/metadata/segments_metadata.csv'
        self.segments_path = './data_and_models/data/train/audio/transformed/segments'
        self.meta = pd.read_csv(segments_metadata_file, encoding='utf-16')
        self.reciter = {}
        # creating the dictionaries
        for key, val in zip(self.meta['reciter id'], self.meta['folder']):
            self.reciter[key] = val.replace('-', ' ')
        self.reciter_count = len(set(self.reciter))
        self.transformation = transformation
        
    def __len__(self):
        # number of samples
        return len(self.meta)
    
    def __getitem__(self, index):
        reciter_num = self.meta.loc[index, 'reciter id']
        file = f"{self.segments_path}/{self.meta.loc[index, 'folder']}/{self.meta.loc[index, 'file path']}"
        signal, sample_rate = torchaudio.load(file)
        signal = self._right_pad(signal)
        signal = self.transformation(signal)
        return signal, reciter_num

    def _right_pad(self, signal):
        base_len = 661504
        if signal.size(1) < base_len or signal.size(1) > base_len:
            signal = F.pad(signal, pad=(0, base_len - signal.size(1)), mode='constant')
        return signal
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ReciterTestDataset(Dataset):
    def __init__(self, transformation):
        metadata_file = './data_and_models/data/test/metadata/metadata.csv'
        segments_metadata_file = './data_and_models/data/train/metadata/segments_metadata.csv'
        self.meta = pd.read_csv(metadata_file, encoding='utf-8')
        self.segments_meta = pd.read_csv(segments_metadata_file, encoding='utf-16')
        self.reciter = {}
        # creating the dictionaries
        for key, val in zip(self.meta['reciter id'], self.meta['folder']):
            self.reciter[key] = val.replace('-', ' ')
        self.transformation = transformation
        
    def __len__(self):
        # number of samples
        return len(self.meta)
    
    def __getitem__(self, index):
        reciter_num = self.meta.loc[index, 'reciter id']
        file = f"./data_and_models/data/test/audio/{self.meta.loc[index, 'folder']}/{self.meta.loc[index, 'file path']}"
        filename = self.meta.loc[index, 'file path']
        chapter = self.meta.loc[index, 'chapter name']
        info = self.meta.loc[index, 'info']
        signal, sample_rate = torchaudio.load(file)
        # check num of audio channels
        if signal.shape[0] >= 2:
            signal = torch.mean(signal, dim=0).unsqueeze(0)
        resampler = T.Resample(sample_rate, 44100, lowpass_filter_width=128, rolloff=0.99, dtype=signal.dtype)
        signal = resampler(signal)
        signal = self._right_pad(signal)
        signal = self.transformation(signal)
        return signal, reciter_num, chapter, filename, info

    def _right_pad(self, signal):
        base_len = 661504
        if signal.size(1) < base_len or signal.size(1) > base_len:
            signal = F.pad(signal, pad=(0, base_len - signal.size(1)), mode='constant')
        return signal
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ChapterDataset(Dataset):
    def __init__(self, transformation):
        metadata_file = './data_and_models/data/train/metadata/segments_metadata.csv'
        self.segments_path = './data_and_models/data/train/audio/transformed/segments'
        self.meta = pd.read_csv(metadata_file, encoding='utf-16')
        self.chapter = {}
        # creating chapter dict
        for key, val in zip(self.meta['chapter num'], self.meta['chapter name']):
            self.chapter[key-1] = val
        self.chapter_count = len(set(list(self.chapter)))
        self.transformation = transformation
        
    def __len__(self):
        # number of samples
        return len(self.meta)
    
    def __getitem__(self, index):
        chapter_num = self.meta.loc[index, 'chapter num']
        file = f"{self.segments_path}/{self.meta.loc[index, 'folder']}/{self.meta.loc[index, 'file path']}"
        signal, sample_rate = torchaudio.load(file)
        signal = self._right_pad(signal)
        signal = self.transformation(signal)
        return signal, chapter_num

    def _right_pad(self, signal):
        base_len = 661504
        if signal.size(1) < base_len or signal.size(1) > base_len:
            signal = F.pad(signal, pad=(0, base_len - signal.size(1)), mode='constant')
        return signal
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class CustomTest(Dataset):
    def __init__(self, transformation, file_path):
        self.segments_path = './data_and_models/data/train/audio/transformed/segments'
        self.file_path = file_path
        self.transformation = transformation

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        signal, sample_rate = torchaudio.load(self.file_path)
        if signal.shape[0] >= 2: # num_channels (mono 1, stereo 2)
            signal = torch.mean(signal, dim=0).unsqueeze(0)
        signal = self._right_pad(signal)
        signal = self.transformation(signal)
        return signal

    def _right_pad(self, signal):
        base_len = 661504
        if signal.size(1) < base_len or signal.size(1) > base_len:
            signal = F.pad(signal, pad=(0, base_len - signal.size(1)), mode='constant')
        return signal
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

mel_spectrogram_rec = T.MelSpectrogram(sample_rate=44100,
                                       n_fft=512,
                                       hop_length=512,
                                       n_mels=32)
mel_spectrogram_ch = T.MelSpectrogram(sample_rate=44100,
                                      n_fft=512,
                                       hop_length=512,
                                       n_mels=32)

train_data_rec = ReciterDataset(mel_spectrogram_rec)
test_data_rec = ReciterTestDataset(mel_spectrogram_rec)
train_dataloader_rec = DataLoader(train_data_rec, batch_size=900, shuffle=True)
test_dataloader_rec = DataLoader(test_data_rec, batch_size=1, shuffle=False)

train_data_ch = ChapterDataset(mel_spectrogram_ch)
train_dataloader_ch = DataLoader(train_data_ch, batch_size=900, shuffle=True)
