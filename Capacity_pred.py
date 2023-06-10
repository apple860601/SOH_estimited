import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn.functional as func
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px

scaler = MinMaxScaler(feature_range = (-2, 2))
data_len = 3 #要輸入的時序訊號長度
pio.renderers.default = "browser"
plotly_config = dict({"scrollZoom": True,'modeBarButtonsToAdd':[
                                        'drawline',
                                        'drawopenpath',
                                        'drawclosedpath',
                                        'drawcircle',
                                        'drawrect',
                                        'eraseshape'
                                       ]})
features_list = ["CCCT","CVCT","V37_419","V38_419","V37_41","V38_41","cap"]
f_len = len(features_list)
pred_battery = ["B0036"]


class temp(Dataset):
    def __init__(self, battery_list):
        featurelist=[]
        capacitylist=[]
        for battery in battery_list:
            featurelist.append(f"features\charge\{battery}\{battery}_Features.csv")
            capacitylist.append(f"features\charge\{battery}\{battery}_capacity.csv")
        
        self.feadf = pd.concat((pd.read_csv(f) for f in featurelist), ignore_index=True)
        self.capdf = pd.concat((pd.read_csv(f) for f in capacitylist), ignore_index=True)
        self.feadf["cap"] = self.capdf["Capacity"]
        self.feadf = pd.DataFrame(scaler.fit_transform(self.feadf),
                                columns=self.feadf.keys())
        self.features = self.feadf.loc[:,features_list].to_numpy()
        # print(self.features)
        self.capacity = self.capdf["Capacity"].to_numpy()
        self.sample_len = data_len

    def __len__(self):
        if len(self.capdf) > self.sample_len:
            return len(self.capdf) - self.sample_len
        
        else:
            return 0

    def __getitem__(self, index):
        target = self.capacity[self.sample_len + index]
        target = np.array(target).astype(np.float32)

        input = self.features[index : (index + self.sample_len)]
        input = torch.from_numpy(input).float()   
        # print(input)
        input.reshape(-1, f_len)
        target = torch.from_numpy(target).float()

        return input, target

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        self.input_size = f_len # input size : 輸入維度
        self.hidden_size = 400 # hidden_size : 隱藏層的特徵維度
        self.num_layers = 8 # hidden_size : LSTM隱藏層的層數
        self.dropout = 0.1 # dropout : 每一層過後丟棄特定比例神經元

        self.gru = nn.GRU(input_size = self.input_size, hidden_size  = self.hidden_size, bidirectional=True,
                            num_layers = self.num_layers, dropout = self.dropout, batch_first = True)
        for m in self.modules():
            if type(m) in [nn.GRU, nn.GRU, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0.1)

        self.fc1 = nn.Linear(self.hidden_size*2, 600)#設定全連接層
        self.fc2 = nn.Linear(600, 600)#設定全連接層
        self.fc3 = nn.Linear(600, 600)#設定全連接層
        self.fc4 = nn.Linear(600, 1)#設定全連接層


    def forward(self, x):
        # print(x.shape)
        h_0 = torch.zeros([self.num_layers*2, x.shape[0], self.hidden_size], device = x.device)
        c_0 = torch.zeros([self.num_layers*2, x.shape[0], self.hidden_size], device = x.device)

        out, _ = self.gru(x,h_0)# x:新資料輸入 h0:上個隱藏層狀態 c0:上個細胞狀態
        # print(out.shape)
        out = self.fc1(func.tanh(out[:, -1, :]))#接收LSTM單元的輸出，並讓最後一層輸出
        out = self.fc2(func.tanh(out))
        out = self.fc3(func.tanh(out))
        out = self.fc4(func.tanh(out))
        # out3 = self.fc3(out2)
        # out1 = func.tanh(self.fc1(out))
        # out2 = func.tanh(self.fc2(out1))
        # out3 = self.fc3(out2)#接收LSTM單元的輸出，並讓最後一層輸出

        return out

pred_data = temp(pred_battery)
data_len = 3 #要輸入的時序訊號長度

device = torch.device('cuda')
model = GRU()
model.load_state_dict(torch.load("GRU_model.pth"))
model = model.to(device)
def pred(data):
    model.eval()

    with torch.no_grad():
        pred = model(data)
        return pred
    
result = []
targets = []
for i in range(len(pred_data)):
    nor_temp, target = pred_data[i]
    targets.append(target)
    temps = nor_temp
    temps = nor_temp.view(1, data_len, f_len)
    temps = temps.to(device)
    pred_temp = pred(temps)
    pred_temp = pred_temp.detach().cpu().numpy().squeeze()

    # real_temp = scaler.inverse_transform(pred_temp.reshape(-1, 1))
    result.append(pred_temp)

df = pd.DataFrame(columns=["target","result"])
df["target"] = targets
df["result"] = result
fig1 = px.line(df)
fig1.update_layout(
    dragmode='drawopenpath',
    newshape_line_color='cyan',
    title_text=f'{pred_battery}',
    xaxis_title="cycle", 
    yaxis_title="capacity/Ahr"
)
# val = pd.DataFrame(pred_trains)
# fig.add_scatter(val,x=val.index,y=["0"])
# fig1.add_scatter(y=result,labels="result")

fig1.show(config=plotly_config)