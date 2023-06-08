import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'.\temperature.csv' , header = 0)
print(df)

scaler = MinMaxScaler(feature_range = (-1, 1))

class temp(Dataset):
    def __init__(self, data):
        self.df = data
        self.org_data = self.df.mean_temp.to_numpy()
        self.normalize_data = np.copy(self.org_data)
        self.normalize_data = self.normalize_data.reshape(-1, 1)
        self.normalize_data = scaler.fit_transform(self.normalize_data)
        self.normalize_data = self.normalize_data.reshape(-1)
        self.sample_len = 12

    def __len__(self):
        if len(self.org_data) > self.sample_len:
            return len(self.org_data) - self.sample_len
        
        else:
            return 0

    def __getitem__(self, index):
        target = self.normalize_data[self.sample_len + index]
        target = np.array(target).astype(np.float32)

        input = self.normalize_data[index : (index + self.sample_len)]
        input = input.reshape(-1, 1)

        input = torch.from_numpy(input).float()
        target = torch.from_numpy(target).float()

        return input, target
    
data = temp(df)

print(data[0][1])

train_len = int(len(data)*0.7)
test_len = len(data) - train_len

generator = torch.Generator().manual_seed(0)
train_data, test_data = random_split(data, [train_len, test_len], generator)

train_loader = DataLoader(train_data, shuffle = True, batch_size = 32)
test_loader = DataLoader(test_data, shuffle = False, batch_size = 32)

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        # input size:輸入維度
        # hidden_size:隱藏層的特徵維度
        # hidden_size :LSTM隱藏層的層數
        # dropout : 每一層過後丟棄特定比例神經元
        self.input_size = 1
        self.hidden_size = 500
        self.num_layers = 3
        self.dropout = 0.1

        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size  = self.hidden_size, num_layers = self.num_layers, dropout = self.dropout, batch_first = True)
        self.linear = nn.Linear(500, 1)#設定全連接層

    def forward(self, x):
        h_0 = torch.zeros([self.num_layers, x.shape[0], self.hidden_size], device = x.device)
        c_0 = torch.zeros([self.num_layers, x.shape[0], self.hidden_size ], device = x.device)
        
        out, _ = self.lstm(x, (h_0.detach(), c_0.detach()))# x:新資料輸入 h0:上個隱藏層狀態 c0:上個細胞狀態
        out = self.linear(out[:, -1, :])#接收LSTM單元的輸出，並讓最後一層輸出

        return out

model = LSTM()
print(model)

device = torch.device('cuda')
model = model.to(device)

loss_f = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr = 1e-3)

def train():
    # train_loss = 0
    model.train()

    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        pred = model(data)
        pred = pred.view(-1)

        loss = loss_f(pred, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item()

def test():
    model.eval()

    for _, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        pred = model(data)
        pred = pred.view(-1)
        loss = loss_f(pred, target)

    return loss.item()

def pred(data):
    model.eval()

    with torch.no_grad():
        pred = model(data)
        return pred
    
train_losses = []
test_losses = []
for i in range(10):
    train_loss = train()
    test_loss = test()

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print("epoch:{}, train_loss:{:0.6f}, test_loss:{:0.6f}".format(i, train_loss, test_loss))

pred_temps = []
for i in range(len(data)):
    nor_temp, taget = data[i]
    temps = nor_temp
    temps = nor_temp.view(1, 12, 1)
    temps = temps.to(device)
    pred_temp = pred(temps)
    pred_temp = pred_temp.detach().cpu().numpy()

    real_temp = scaler.inverse_transform(pred_temp.reshape(-1, 1))
    pred_temps.append(real_temp.item())

month = range(0, df.month.size)
mean_temp = df.mean_temp

plt.figure(1)
plt.plot(train_losses, label = 'train_loss')
plt.plot(test_losses, label = 'test_loss')
plt.legend()

plt.figure(2)
plt.plot(month, mean_temp, label = "org_data", color = 'b')
plt.plot(month[12:], pred_temps, label = "pred_data", color = 'r')
plt.legend()

plt.show()





