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
You can predict with the included pre-trained models (currently one model), and re-train if needed. Delete the existing model to train from scratch.



## Performance


* **NeuralNet_2l_lin:** Neural Network with 2 linear layers
* **NeuralNet_4l_relu_lin:** Neural Network with 4 linear layers and 3 ReLU layers

The best **F<sub>1</sub> score is ~71.4% with model NeuralNet_2l_lin**

**Note:** Score on the test set with the best parameters within 100 epochs of training, with the original training data.


## Issues
- The training dataset is still too small, resulting in overfitting (after augmentation).
- Reproducibility is an issue with regard to training.


## Preprocessing
The preprocessing script removes the slang tags, brackets, hyphens, and converts everything to lowercase.


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

