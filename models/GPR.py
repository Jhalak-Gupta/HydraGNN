import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm  

class GPRNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, alpha=0.1, init='PPR'):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.K = K
        self.alpha = alpha

        if init == 'PPR':
            weights = [(1 - alpha) ** k for k in range(K)] + [alpha]
        else:
            weights = [1.0 / (K + 1)] * (K + 1)
        self.prop_weights = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float32))

    def forward(self, x, edge_index, num_nodes=None):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        norm_edge_index, edge_weight = gcn_norm(edge_index, num_nodes=num_nodes, add_self_loops=True)

        out = self.prop_weights[0] * x
        h = x
        for k in range(1, self.K + 1):
            row, col = norm_edge_index
            m = h[col] * edge_weight.view(-1, 1)
            h_new = torch.zeros_like(h)
            h_new = h_new.index_add(0, row, m)
            h = h_new
            out = out + self.prop_weights[k] * h
        return out
