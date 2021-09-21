import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import os

#restrict to 30 cores
torch.set_num_threads(30)

CHECKPOINT_DIR = 'torch/'
CHECKPOINTS = CHECKPOINT_DIR+'model.pt'

def chunk(frame,cap):
    pass
    
class Variable_Dataset(Dataset):
    def __init__(self, sequence_file, coords_file, transform=None):
       
        include_fields = ['hours','avg_employees_7days','day_of_week','Lemployees']
        self.frame = pd.read_csv(sequence_file,usecols=include_fields).dropna()
        self.coords = pd.read_csv(coords_file).dropna()
        self.coords = self.coords[self.coords['last_index']-self.coords['first_index']>1]

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start = int(self.coords.iloc[idx]['first_index'])
        end = int(self.coords.iloc[idx]['last_index'])
        series = torch.Tensor(np.array(self.frame[start:end+1]['hours'].astype('float32')))
        if(len(series)<2):
            series = torch.Tensor(np.array([-1,-1,-1]))
        series = series.view(len(series),1)
        return series[:-1], series[1:]

def pad_collate(batch):
  (xx, yy) = zip(*batch)
  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]
  #print(f"x_lens: {x_lens}")

  xx_pad = pad_sequence(xx, batch_first=True, padding_value=-1)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=-1)

  return xx_pad, yy_pad, x_lens, y_lens

train_set = Variable_Dataset("/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/VariableLengths/train10_sample_sequences.csv",
                             "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/VariableLengths/train10_sample_coords.csv")

train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=pad_collate)

class RNN(nn.Module):
    def __init__(self, hidden_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        # The LSTM takes in a single number (dim=1), and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(1, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x_padded,x_lens):
        x_packed = pack_padded_sequence(x_padded, x_lens, batch_first=True, enforce_sorted=False)
        output_packed, _ = self.lstm(x_packed)
        output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)
        output = self.output(output_padded)
        return output

model = RNN(64)
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

if not os.path.isdir(CHECKPOINT_DIR): 
    os.makedirs(CHECKPOINT_DIR)
elif os.path.isfile(CHECKPOINTS):
    checkpoint = torch.load(CHECKPOINTS)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

for epoch in range(1):
    for (idx,(x_padded, y_padded, x_lens, y_lens)) in enumerate(train_dataloader):
        model.zero_grad()

        predictions_padded = model(x_padded,x_lens)
        loss = loss_function(predictions_padded,y_padded)
        print(f"Loss: {loss}  idx: {idx} of ~{500000}")
        if idx%100==0:
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, CHECKPOINTS)
            print(f"Saved {idx} batches of training")
        loss.backward()
        optimizer.step()

eval_dataloader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=pad_collate)
total_loss = 0
total_days = 0
for (idx,(x_padded, y_padded, x_lens, y_lens)) in enumerate(eval_dataloader):
    predictions = model(x_padded,x_lens)
    loss = loss_function(predictions,y_padded)
    days = len(x_padded)
    total_loss += loss*days
    total_days += days
    if idx%1000 == 0:
        print(f"Current MSE: {total_loss/total_days}")
        #if len(x_padded[0])<=10:
        #            print(f"Sample output: {predictions}")
        #            print(f"True output: {y_padded}")    

print(f"Final MSE: {total_loss/total_days}")
