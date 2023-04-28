import torch.nn.functional as F
import torch
from torch_geometric.utils import get_laplacian
import numpy as np

def edgeindex2adj(edge_index, num_nodes):
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1
    return adj

def adj2edgeindex(adj):
    edge_index = torch.nonzero(adj).t()
    return edge_index

def laplacian(adj, typ = 'normailzed'):
    if typ == 'normalized':
        # invert only the non-zero entries
        D = torch.diag(torch.sum(adj, dim=0) ** (-1/2))
        L = torch.eye(adj.shape[0]) - torch.mm(torch.mm(D, adj), D)
    elif typ == 'rw':
        D = torch.diag(torch.sum(adj, dim=0) ** (-1))
        L = torch.eye(adj.shape[0]) - torch.mm(D, adj)
    else:
        D = torch.diag(torch.sum(adj, dim=0))
        L = D - adj
    return L


def smooth_label_loss(data, out, adj, alpha=0.9):
    adj = edgeindex2adj(data.edge_index, data.x.shape[0])
    lap = laplacian(adj, typ='adf')
    # pred_z = out
    # data.edge_index = adj2edgeindex(adj)
    loss_1 = torch.trace(torch.mm(torch.mm(out.t(), lap), out)) / out.shape[0]
    loss_2 = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    # loss_3 = -torch.norm(adj)**2
    return alpha * loss_1 + loss_2 

def train(model, data, optimizer, scheduler=None, loss='cross_entropy', alpha=0.9): # train for one epoch
    model.train()
    optimizer.zero_grad()
    adj = edgeindex2adj(data.edge_index, data.x.shape[0])
    out = model(data.x, data.edge_index)
    if loss == 'smooth_label':
        loss = smooth_label_loss(data, out, adj, alpha=alpha)
    elif loss == 'cross_entropy':
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    else:
        raise ValueError('invalid loss function')
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss

def test(model, data): # get accuracy on train, val, and test sets
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


def non_smooth_label_metric(dataset, preds, k=1):
    data = dataset[0]
    non_label_smooth_node_counts = 0
    total_nodes = 0
    non_label_smooth_nodes = []
    n = data.x.shape[0]
    adj = edgeindex2adj(data.edge_index, n)
    counts = torch.zeros((n, dataset.num_classes))
    for edge in data.edge_index.T:
        counts[edge[0]][data.y[edge[1]]] +=1
    for i in range(n):
        if data.val_mask[i]:
            if preds[i]!=data.y[i]:
                if counts[i][data.y[i]] == adj[i].sum() and adj[i].sum() > 1:
                    non_label_smooth_node_counts +=1
                    non_label_smooth_nodes.append(i)
            total_nodes +=1
    return non_label_smooth_node_counts/total_nodes

def label_difference(model, dataset):
    model.eval()
    data = dataset[0]
    pred = model(data.x, data.edge_index).argmax(dim=-1)
    # label diff = 1/V * sum_{v in V} |z_v - z_c|
    # z_c is readout(intermediate feature) of nodes with class c
    # z_v is readout(intermediate feature) of node v
    # V is the set of nodes with class c

    C = dataset.num_classes
    label_diff = np.zeros(C)
    features = model.features
    for c in range(C):
        num_nodes = data.y[data.y==c].shape[0]
        if num_nodes == 0:
            return 0
        zc = features[data.y==c].mean(dim=0)
        zv = features[data.y==c]
        label_diff[c] = (torch.norm(zv-zc, dim=1).sum()/num_nodes).item()
    return label_diff