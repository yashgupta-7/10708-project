import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, GATConv, LabelPropagation
import torch.nn.functional as F
from alp import AdaptiveLabelPropagation
import torch_geometric

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.features = None
    def forward(self, x: Tensor, edge_index: Tensor, edge_weight:Tensor=None) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index, edge_weight).relu()
        self.features = x
        x = self.conv2(x, edge_index, edge_weight)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)
        self.features = None
        # self.s = adj.clone()
        # self.alpha = 0.99

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        self.features = x
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # xtx = x @ x.t()
        # temp_S = torch.where(xtx > torch.quantile(xtx, 0.98), torch.tensor(1.), torch.tensor(0.))
        # self.s = self.alpha * self.s + (1 - self.alpha) * temp_S
        # w = self.s * xtx
        return x

class LP(torch.nn.Module):
    def __init__(self, num_layers, alpha):
        super().__init__()
        self.prop = LabelPropagation(num_layers, alpha)
    
    def train(self, dataset):
        self.data = dataset[0]
        self.out = self.prop(self.data.y, self.data.edge_index, self.data.train_mask)
        return self.out
    
    def test(self):
        pred = self.out.argmax(dim=-1)
        accs = []
        for mask in [self.data.train_mask, self.data.val_mask, self.data.test_mask]:
            accs.append(int((pred[mask] == self.data.y[mask]).sum()) / int(mask.sum()))
        return accs

class AdaptiveLP(torch.nn.Module):
    def __init__(self, num_layers, yshape, edge_dim):
        super().__init__()
        self.k = num_layers
        self.yshape = yshape
        self.edge_weight = torch.nn.Parameter(torch.ones(edge_dim))
        self.weight1 = torch.nn.Linear(yshape, yshape, bias=True)
        self.weight2 = torch.nn.Linear(yshape, 1, bias=False)
    
    def dense_lp(self, data):
        labels = torch.nn.functional.one_hot(data.y.type(torch.long)).type(torch.float)
        labels[~data.train_mask] = 0
        matrix = torch_geometric.utils.to_dense_adj(data.edge_index, edge_attr=self.edge_weight.sigmoid(), max_num_nodes=data.num_nodes)
        matrix = matrix.squeeze(0)
        selfloop = torch.diag(torch.ones(matrix.shape[0]))
        matrix += selfloop
        self.ys = [labels]
        for _ in range(self.k):
          y = torch.matmul(matrix, labels)
          self.ys.append(y)
          labels = y
        return torch.nn.functional.normalize(labels, dim=1)
    
    def forward(self, data):
        dense_lp = self.dense_lp(data)
        # return dense_lp
        self.ys = torch.stack(self.ys).permute(0, 2, 1)
        q = self.weight1(self.ys)
        q = torch.relu(q)
        alpha = self.weight2(q)
        alpha = torch.nn.Softmax(dim=-1)(alpha.view(self.ys.shape[1], -1))
        alpha = alpha.view(self.ys.shape[1], -1, 1).float()
        b = torch.sum(alpha * self.ys.permute(1, 0, 2), dim=1).permute(1, 0)
        return torch.nn.functional.normalize(b, dim=1)
    
        