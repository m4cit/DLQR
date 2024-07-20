# Deep Learning Quran Recognition
DLQR is an experimental project for Qur'an audio recognition via Deep-Learning (PyTorch). A total of 34,403 audio samples were used to train a custom Convolutional Neural Network. The goal is to predict the reciter and / or the chapter. The chapter prediction is not fully thought out yet and currently unavailable.


## Gallery
### Icon
<img src='https://raw.githubusercontent.com/m4cit/Deep-Learning-Quran-Recognition/gallery/icon.png' height="120">


## Requirements
1. Install Python **3.10** or newer.
2. Install [PyTorch](https://pytorch.org/get-started/locally/) (I used 2.2.2 + cu121)
3. Install the required packages by running `pip install -r requirements.txt` in your shell of choice. Make sure you are in the project directory.


## Usage
**Example 1:**
>```
>python DLQR.py --predict -t reciter -i .\path\to_some\file.mp3 -dev cpu
>```
or
>```
>python DLQR.py --predict --target reciter --input .\path\to_some\file.mp3 --device cuda
>```
\
\
**Example 2:**
>```
>python DLQR.py --train -m cnn_reciter -dev cuda
>```
or
>```
>python DLQR.py --train --model cnn_reciter --device cpu
>```

You can predict with the included pre-trained models (currently one model), and re-train if needed. Delete the existing model to train from scratch.


## Data
The samples were manually obtained through the following websites:

[Quran Central](https://qurancentral.com/)

[Quran Player MP3](https://www.quranplayermp3.com/)

Web scraping scripts didn't really work...


## Preprocessing
For the preprocessing I wrote two PowerShell scripts. Both utilize [ffmpeg](https://www.ffmpeg.org/).

This could have been achieved with PyTorch itself, but I always wanted to write a PowerShell script :)

### trim_audio.ps1:
Trims the beginning of the original audio files (start time and details are specified in the correction.csv file, located in data_and_models\data\). Most files from the cited websites contain a portion in the beginning, which isn't from the reciters themselves.

### resample_segment_audio_files.ps1:
As the name suggests, this script is resampling the original audio files and splits them into 15 second chunks (.wav format to avoid re-encoding and further quality loss).


## Performance
<img src='https://raw.githubusercontent.com/m4cit/Deep-Learning-Quran-Recognition/gallery/demo.png' width="900">

The *Demo* image above suggests that 50% of the unseen data (during training) in the test set is being recognized / predicted correctly, and that the accuracy between different reciters is not consistent.

About 64% of the seen data is being predicted correctly. Adding low and high intensity noise to one of the samples made no difference (15, 16). As mentioned, most files contain a portion in the beginning, which seems to affect results (6, 7). These portions were removed before training.


## Used Libraries / Open Source Projects
* [PyTorch](https://pytorch.org/) and its dependencies
* [tqdm](https://tqdm.github.io/)

