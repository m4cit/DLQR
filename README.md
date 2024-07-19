# Deep Learning Quran Recognition
DLQR is an experimental project for Qur'an recognition via Deep-Learning. A total of 34,403 audio samples were used to train a custom Convolutional Neural Network. The goal is to predict the reciter through audio inputs. There is also a model for chapter prediction, but not trained or fully thought out yet.

## Gallery
### Icon
<img src='https://raw.githubusercontent.com/m4cit/Deep-Learning-Quran-Recognition/gallery/icon.png' height="120">


### Demo
<img src='https://raw.githubusercontent.com/m4cit/Deep-Learning-Quran-Recognition/gallery/demo.png' width="900">


## Requirements
1. Install Python **3.10** or newer.
2. Install [PyTorch](https://pytorch.org/get-started/locally/) (I used 2.2.2 + cu121)
3. Install the required packages by running `pip install -r requirements.txt` in your shell of choice. Make sure you are in the project directory.


## Usage
**Example 1:**
>```
>python --predict -t reciter -i .\path\to_some\file.mp3 -dev cpu
>```
or
>```
>python --predict --target reciter --input .\path\to_some\file.mp3 --device cuda
>```
\
\
**Example 2:**
>```
>python --train -m cnn_reciter -dev cuda
>```
or
>```
>python --train --model cnn_reciter --device cpu
>```

You can predict with the included pre-trained models (currently one model), and re-train if needed. Delete the existing model to train from scratch.


## Performance
The performance is currently not great (check out the demo screenshot).


## Data
The samples were manually obtained through the following websites:
[Quran Central](https://qurancentral.com/)

[Quran Player MP3](https://www.quranplayermp3.com/)

Web scraping scripts didn't really work...


## Preprocessing
For the preprocessing I wrote two PowerShell scripts: *trim_audio.ps1* and *resample_segment_audio_files.ps1*. Both utilize [ffmpeg](https://www.ffmpeg.org/).


### trim_audio.ps1
Trims the beginning of the original audio files (start time and file is specified in the correction.csv file located in data_and_models\data\

### resample_segment_audio_files.ps1:
As the name suggests, this script is resampling the original audio files and splits them into 15 second chunks (.wav format to avoid re-encoding and further quality loss).


## Augmentation
I categorized the slang words as:
* \<pex> personal expressions
  * _dude, one and only, bro_
* \<n> singular nouns
  * _shit_
* \<npl> plural nouns
  * _crybabies_
* \<shnpl> shortened plural nouns
  * _ppl_
* \<mwn> multiword nouns
  * _certified vaccine freak_
* \<mwexn> multiword nominal expressions
  * _a good one_
* \<en> exaggerated nouns
  * _guysssss_
* \<eex> (exaggerated) expressions
  * _hahaha, aaaaaah, lmao_
* \<adj> adjectives
  * _retarded_
* \<eadj> exaggerated adjectives
  * _weirdddddd_
* \<sha> shortened adjectives
  * _on_
* \<shmex> shortened (multiword) expressions
  * _tbh, imo_
* \<v> infinitive verb
  * _trigger_

(not all tags are available due to the small dataset)


## Source of the data
Most of the phrases come from archive.org's [Twitter Stream of June 6th](https://archive.org/details/archiveteam-twitter-stream-2021-06).


## Recognition of Open Source use
* PyTorch
* scikit-learn
* customtkinter
* pandas
* numpy
* tqdm

