from torch import nn
import torch
from torch.nn import functional as F

def get_offset(L):
    if L % 2 == 0:
        return L // 2 - 1
    return L - L // 2 - 1
    
class SelfAttention(nn.Module):
    def __init__(self, d, nhead):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, nhead, batch_first=True)
    def forward(self, x):
        return self.mha(x, x, x)[0]

class Block(nn.Module):
    def __init__(self, d, nhead, n=None, dropout=0.0):
        super().__init__()
        n = n or d*2
        self.mha = SelfAttention(d, nhead)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ffwd = nn.Sequential(
            nn.Linear(d, n),
            nn.GELU(),
            nn.Linear(n, d),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, x_lens, winsizes, device, n_classes=20):
        super().__init__()
        self.x_lens = x_lens
        self.winsizes = winsizes
        self.n_medoids = sum(x_lens[1:])
        self.af_input_size = sum([w*2 for w in winsizes])
        self.input_size = sum([l*w*2 for l,w in zip(x_lens, winsizes)])
        self.device = device

        self.d = 32
        nheads = 4

        # self.embs = nn.ModuleList([nn.Linear(w*2, self.d, bias=False) for w in winsizes[1:]])
        n_clusters = 2 # FOR WIN 7 ONLY
        self.pos_emb = nn.Embedding(winsizes[-1], self.d) # positional embedding for 7 (biggest window size) positions
        self.emb = nn.Linear(n_classes + 2*n_clusters, self.d)
        
        dropout = 0.15
        nlinear = self.d
        self.blocks = nn.Sequential(
            Block(self.d, nheads, nlinear, dropout),
            # Block(self.d, nheads, nlinear, dropout),
            # Block(self.d, nheads, nlinear, dropout),
            # Block(self.d, nheads, nlinear, dropout)
        )

        self.ln_f = nn.LayerNorm(self.d)
        self.out = nn.Linear(self.d, 2)

    def emb_mask(self, x_medoids):
        xs = [[] for _ in range(x_medoids[0].shape[0])] # batch size
        for i,x in enumerate(x_medoids[1:]):
            mask = torch.any(x, dim=2)
            for j,xi in enumerate(x):
                xs[j].append(self.embs[i](xi[mask[j]]))
        return torch.nested.nested_tensor([torch.cat(xi, dim=0) for xi in xs])

    def forward(self, x_medoids, x_res):
        # x = torch.cat([self.embs[i](x[:,:1]) for i,x in enumerate(x_medoids[1:])], dim=1) # one cluster for each window
        # x = torch.cat([self.embs[i](x) for i,x in enumerate(x_medoids[1:])], dim=1)
        # TODO remove examples in batch with all zeros in x_medoids (no clusters)
        N, C, L = x_medoids.shape
        L = L // 2
        x_res = nn.functional.one_hot(x_res, num_classes=20)
        x_medoids = x_medoids.transpose(-2,-1).view(N, 2, L, C).transpose(-3,-2).flatten(-2,-1)
        x = torch.cat([x_res, x_medoids], dim=-1)

        pos = (torch.arange(L) + get_offset(self.winsizes[-1]) - get_offset(L)).to(self.device)
        pos = self.pos_emb(pos)
        
        x = self.emb(x) + pos
        
        x = self.blocks(x)
        x = self.out(x)
        return x
    
    def get_optimizer(model):
        decay_params = [p for n,p in model.named_parameters() if p.dim() >= 2] # linear and attention layers
        no_decay_params = [p for n,p in model.named_parameters() if p.dim() < 2] # layernorm and bias layers
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': 1e-5},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], betas=(0.9, 0.999), lr=1e-4)
        return optimizer
    
class AngleMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, reduce='mean'):
        def diff(x1, x2):
            d = torch.abs(x1 - x2)
            d = torch.minimum(d, 360-d)
            return d
        if reduce=='sum':
            return torch.sum(diff(x, y)**2)
        return torch.mean(diff(x, y)**2)