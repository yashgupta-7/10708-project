import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, GATConv, LabelPropagation
import torch.nn.functional as F
from alp import AdaptiveLabelPropagation

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)
        # self.s = adj.clone()
        # self.alpha = 0.99

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
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

class ALP(torch.nn.Module):
    def __init__(self, num_layers, alpha, yshape):
        super().__init__()
        self.prop = AdaptiveLabelPropagation(num_layers, alpha)
        self.yshape = yshape
        self.weight1 = torch.nn.Linear(yshape, yshape, bias=True)
        self.weight2 = torch.nn.Linear(yshape, 1, bias=False)
    
    def lpa(self, dataset):
        self.data = dataset[0]
        
        sub_train_mask = self.data.train_mask.clone()
        # sub_train_mask[torch.randperm(sub_train_mask.shape[0])[:int(sub_train_mask.shape[0] * 0.7)]] = False
        # print(sub_train_mask.sum(), self.data.train_mask.sum())
        
        self.outs = self.prop(self.data.y, self.data.edge_index, sub_train_mask)
        # print(self.outs[0].shape)
        self.outs = torch.stack(self.outs).permute(0, 2, 1)
        # print(self.outs.shape)
        return self.outs
    
    def forward(self):
        q = self.weight1(self.outs)
        q = torch.relu(q)
        alpha = self.weight2(q)
        alpha = torch.nn.Softmax(dim=-1)(alpha.view(self.outs.shape[1], -1))
        alpha = alpha.view(self.outs.shape[1], -1, 1).float()
        b = torch.sum(alpha * self.outs.permute(1, 0, 2), dim=1).permute(1, 0)
        # b = torch.nn.Sof .tmax(dim=-1)(b)
        return b
    
    def predict(self):
        pred = self.forward().argmax(dim=-1)
        return pred
    
    # def loss(self):
    #     pred = self.forward()[self.data.train_mask]
    #     y = self.data.y[self.data.train_mask]
    #     # y = torch.nn.functional.one_hot(y).float()
    #     # print(y.shape)
    #     # print(pred.sum(axis=1), y.sum(axis=1))
    #     return torch.nn.CrossEntropyLoss()(pred, y)
        
    def test(self):
        pred = self.predict()
        accs = []
        for mask in [self.data.train_mask, self.data.val_mask, self.data.test_mask]:
            accs.append(int((pred[mask] == self.data.y[mask]).sum()) / int(mask.sum()))
        return accs