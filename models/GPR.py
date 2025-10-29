import torch
import torch.nn.functional as F
from torch_geometric.nn import GPRConv
from torch.nn import Linear

class GPRGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, alpha=0.1):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.prop = GPRConv(K, alpha, dropout=0.5)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return x
