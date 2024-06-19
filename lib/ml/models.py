import torch
from torch import nn
import torch.nn.functional as F

class ModelWrapper():
    def __init__(self, lengths):
        self.model = LSTMNet(lengths)
        self.lengths = lengths

    def predict(self, X, xres):
        with torch.no_grad():
            return self.model(X, xres)
    
    def __call__(self, X, xres):
        return self.predict(X, xres)
    
    def load(self, weights_file):
        self.model.load_state_dict(torch.load(weights_file))

class LSTMNet(nn.Module):
    def __init__(self, lengths):
        super().__init__()
        self.lengths = lengths
        self.s = [sum(lengths[:i]) for i,l in enumerate(lengths)]

        self.h = 32
        h = self.h
        nl = 1
        p_drop = 0.0
        mlp_h = 20
        self.lstm1 = nn.LSTM(2, h, nl, batch_first=True, bidirectional=True, dropout=p_drop)
        self.lstm2 = nn.LSTM(2, h, nl, batch_first=True, bidirectional=True, dropout=p_drop)
        self.lstm3 = nn.LSTM(2, h, nl, batch_first=True, bidirectional=True, dropout=p_drop)
        self.lstm4 = nn.LSTM(2, h, nl, batch_first=True, bidirectional=True, dropout=p_drop)
        self.ln1 = nn.LayerNorm(h*8, elementwise_affine=False)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(h*8+20, mlp_h)
        self.ln2 = nn.LayerNorm(mlp_h, elementwise_affine=False)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(mlp_h, 2)
        
    def forward(self, x, xres):
        x1 = self._sort(x[:,:,self.s[0]:self.s[1]].permute(0,2,1))
        x2 = self._sort(x[:,:,self.s[1]:self.s[2]].permute(0,2,1))
        x3 = self._sort(x[:,:,self.s[2]:self.s[3]].permute(0,2,1))
        x4 = self._sort(x[:,:,self.s[3]:         ].permute(0,2,1))

        # h = self.lstm1(x1)[1][0]  # num_layers * num_directions, batch, hidden_size
        # h = h.permute(1,0,2)      # batch, num_layers * num_directions, hidden_size
        # h = h[:,-2:,:]            # batch, 2, hidden_size (final hidden state of each direction for last layer)
        # x = h.flatten(1)          # batch, 2*hidden_size
        x1 = self.lstm1(x1)[1][0].permute(1,0,2)[:,-2:,:].flatten(1)
        x2 = self.lstm2(x2)[1][0].permute(1,0,2)[:,-2:,:].flatten(1)
        x3 = self.lstm3(x3)[1][0].permute(1,0,2)[:,-2:,:].flatten(1)
        x4 = self.lstm4(x4)[1][0].permute(1,0,2)[:,-2:,:].flatten(1)
        x = torch.cat([x1,x2,x3,x4], dim=1)
        x = self.ln1(x)
        x = torch.cat([x, xres], dim=1)
        x = self.dropout1(x)
        x = self.fc1(F.relu(x))
        x = self.ln2(x)
        x = self.dropout2(x)
        x = self.fc2(F.relu(x))
        return x
    
    def _sort(self, x):
        # idxs = x.sum(dim=2).sort()[1].unsqueeze(-1).expand(-1,-1,2)
        # x = x.gather(1, idxs)
        return x