import torch
from torchaudio import functional as AF
from torch.nn import functional as F
import torchaudio
import argparse

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('-i', '--input', type=str, required=True, help='Audio file input.')
parser.add_argument('-o', '--output', type=str, required=True, help='Audio file output (mp3 or wav).')
parser.add_argument('-w', '--weight', type=str, required= True, choices=('high', 'low'), help='Weight / intensity of the noise.')
args = parser.parse_args()

if args.weight == None:
    print('-w (--weight) missing!')
    exit()

signal, sample_rate = torchaudio.load(args.input)
signal_noise, sample_rate_noise = torchaudio.load('./data_and_models/data/noise.mp3')

# check num of audio channels
if signal.shape[0] >= 2:
    signal = torch.mean(signal, dim=0).unsqueeze(0)
# padding
if signal_noise.size(1) < signal.size(1) or signal_noise.size(1) > signal.size(1):
    signal = F.pad(signal, pad=(0, signal_noise.size(1) - signal.size(1)), mode='constant')

if args.weight == 'low':
    snr_dbs = torch.tensor([5])
elif args.weight == 'high':
    snr_dbs = torch.tensor([-5])
distorted_signal = AF.add_noise(signal, signal_noise, snr_dbs)

if (args.output).endswith('.mp3'):
    final_format = 'mp3'
elif (args.output).endswith('.wav'):
    final_format = 'wav'
else:
    print('Please use mp3 or wav as output format!')

torchaudio.save(args.output, src=distorted_signal, sample_rate=sample_rate, format=final_format)
