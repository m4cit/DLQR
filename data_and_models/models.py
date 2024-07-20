import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from .dataset import train_data_rec, train_dataloader_rec, train_data_ch, train_dataloader_ch
from .dataset import test_dataloader_rec
from .dataset import CustomTest, color, mel_spectrogram_ch, mel_spectrogram_rec
import os

device_options = ('cpu', 'cuda', 'ipu', 'xpu', 'mkldnn', 'opengl', 'opencl', 'ideep', 'hip', 've', 'fpga', 'ort', 'xla', 'lazy', 'vulkan', 'mps', 'meta', 'hpu', 'mtia')



# //////////////////////////////////////////////////////////////////////////
reciter_inputs = {'in_channels' : 1, 'out_channels' : 16, 'kernel_size' : 3,
                        'stride' : 1, 'max_pool_kernel_size' : 2, 'classes' : train_data_rec.reciter_count}
    
class Reciter_CNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(reciter_inputs['in_channels'], # mono audio, therefore 1
                      reciter_inputs['out_channels'],
                      reciter_inputs['kernel_size'],
                      reciter_inputs['stride']),
            nn.ReLU(),
            nn.MaxPool2d(reciter_inputs['max_pool_kernel_size'])
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(reciter_inputs['out_channels'],
                      (reciter_inputs['out_channels'])*2,
                      reciter_inputs['kernel_size'],
                      reciter_inputs['stride']),
            nn.ReLU(),
            nn.MaxPool2d(reciter_inputs['max_pool_kernel_size']),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d((reciter_inputs['out_channels'])*2,
                      (reciter_inputs['out_channels'])*4,
                      reciter_inputs['kernel_size'],
                      reciter_inputs['stride']),
            nn.ReLU(),
            nn.MaxPool2d(reciter_inputs['max_pool_kernel_size']),
        )
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(20352, reciter_inputs['classes'])
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear_layer(x)
        return x
# //////////////////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////////////////
chapter_inputs = {'in_channels' : 1, 'out_channels' : 16, 'kernel_size' : 3,
                        'stride' : 1, 'max_pool_kernel_size' : 2, 'classes' : train_data_ch.chapter_count}

class Chapter_CNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(chapter_inputs['in_channels'], # mono audio, therefore 1
                      chapter_inputs['out_channels'],
                      chapter_inputs['kernel_size'],
                      chapter_inputs['stride']),
            nn.ReLU(),
            nn.MaxPool2d(chapter_inputs['max_pool_kernel_size'])
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(chapter_inputs['out_channels'],
                      (chapter_inputs['out_channels'])*2,
                      chapter_inputs['kernel_size'],
                      chapter_inputs['stride']),
            nn.ReLU(),
            nn.MaxPool2d(chapter_inputs['max_pool_kernel_size']),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d((chapter_inputs['out_channels'])*2,
                      (chapter_inputs['out_channels'])*4,
                      chapter_inputs['kernel_size'],
                      chapter_inputs['stride']),
            nn.ReLU(),
            nn.MaxPool2d(chapter_inputs['max_pool_kernel_size']),
        )
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(20352, chapter_inputs['classes'])
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear_layer(x)
        return x
# //////////////////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////////////////

def train_model(target_type:str=('reciter', 'chapter'), device:str=device_options):
    if target_type == 'reciter':
        model = Reciter_CNN_Model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=2)
        train_len = len(train_dataloader_rec)
        train_dataloader = train_dataloader_rec
    elif target_type == 'chapter':
        model = Chapter_CNN_Model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=Chapter_CNN_Model().parameters(), lr=0.1)
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=2)
        train_len = len(train_dataloader_ch)
        train_dataloader = train_dataloader_ch
        
    # load existing model to avoid training entirely new model
    if os.path.isfile(f'./data_and_models/saved_models/{model.__class__.__name__}.pth'):
        checkpoint = torch.load(f'./data_and_models/saved_models/{model.__class__.__name__}.pth')
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        model = model.to(device)
    
    model.train()
    epochs = 90
    print(f'Training {model.__class__.__name__}...')
    for epoch in tqdm(range(epochs), desc='Epochs Total'):
        train_loss = 0
        batch_progress = tqdm(total=train_len, desc='Batch Count')
        for  signal, target in train_dataloader:
            if target_type == 'chapter':
                target = target-1
            # 1. forward pass
            target_pred = model(signal.to(device))
            # 2. calculate loss (per batch)
            loss = loss_fn(target_pred, target.to(torch.long).to(device))
            train_loss += loss
            # 3. optimizer zero grad
            optimizer.zero_grad()
            # 4. loss backward
            loss.backward()
            # 5. optimizer step
            optimizer.step()
            batch_progress.update()
        scheduler.step()
        batch_progress.close()
        # divide total train loss by length of train loader
        train_loss = train_loss / len(train_dataloader)
        print(f'\nAverage loss: {train_loss}\n')
        
        # save after every epoch just in case
        state = {'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'loss': loss}
        torch.save(state, f'./data_and_models/saved_models/{model.__class__.__name__}.pth')
        if train_loss <= 0.0009:
            print(f'{color.GREEN}Stopping at Epoch {epoch + 1}{color.END}\n')
            break
# //////////////////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////////////////
def predict(target_type, device, input_file):
    with torch.inference_mode():
        if target_type == 'reciter':
            model = Reciter_CNN_Model()
            mel_spectrogram = mel_spectrogram_rec
        elif target_type == 'chapter':
            model = Chapter_CNN_Model()
            mel_spectrogram = mel_spectrogram_ch
            
        if os.path.isfile(f'./data_and_models/saved_models/{model.__class__.__name__}.pth'):   
            # loading trained model
            checkpoint = torch.load(f'./data_and_models/saved_models/{model.__class__.__name__}.pth')
            model.load_state_dict(checkpoint['state_dict'])
            loaded_model = model.to(device)
            loaded_model.eval()
            
            if input_file != None:
                custom_test_data = CustomTest(mel_spectrogram, input_file)
                custom_test_dataloader = DataLoader(custom_test_data, batch_size=1)
            
                for signal in custom_test_dataloader:
                    loaded_pred = loaded_model(signal.to(device))
                    if target_type == 'reciter':
                        print(f'\n\nReciter prediction: {color.BLUE}{train_data_rec.reciter[loaded_pred.argmax().item()]}{color.END}\n')
                    elif target_type == 'chapter':
                        print(f'\n\nChapter prediction: {color.BLUE}{train_data_ch.chapter[(loaded_pred.argmax().item())+1]}{color.END}\n')
            else:
                print(f'\n{color.RED}No input file specified!{color.END}')
        else:
            print(f'\n{color.RED}Trained model does not exist!{color.END}')
# //////////////////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////////////////
def demo(target_type):
    with torch.inference_mode():
        if target_type == 'reciter':
            model = Reciter_CNN_Model()
        elif target_type == 'chapter':
            model = Chapter_CNN_Model()
            
        if os.path.isfile(f'./data_and_models/saved_models/{model.__class__.__name__}.pth'):   
            # loading trained model
            checkpoint = torch.load(f'./data_and_models/saved_models/{model.__class__.__name__}.pth')
            model.load_state_dict(checkpoint['state_dict'])
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            loaded_model = model.to(device)
            loaded_model.eval()
            if target_type == 'reciter':
                i = 1
                for signal, label, chapter, filename, info in test_dataloader_rec:
                    loaded_pred = loaded_model(signal.to(device))
                    actual_label = train_data_rec.reciter[label.item()]
                    prediction  = train_data_rec.reciter[loaded_pred.argmax().item()]
                    
                    if prediction == actual_label:
                        prediction = f'{color.BLUE}{train_data_rec.reciter[loaded_pred.argmax().item()]}{color.END}'
                        actual_label = f'{color.BLUE}{train_data_rec.reciter[label.item()]}{color.END}'
                    else:
                        prediction = f'{color.RED}{train_data_rec.reciter[loaded_pred.argmax().item()]}{color.END}'
                        actual_label = f'{color.RED}{train_data_rec.reciter[label.item()]}{color.END}'
                        
                    print(f'{i}.\n{filename[0]}')
                    print(chapter[0])
                    print(f'Reciter Prediction: {prediction}')
                    print(f'Actual: {actual_label}')
                    print(f'{info[0]}\n')
                    i += 1
                    
            elif target_type == 'chapter':
                print(f'\n\nChapter Prediction: {color.BLUE}{train_data_ch.chapter[(loaded_pred.argmax().item())+1]}{color.END}\n')
# //////////////////////////////////////////////////////////////////////////