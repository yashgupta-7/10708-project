import torch.nn.functional as F
import torch
from torch_geometric.utils import get_laplacian

def edgeindex2adj(edge_index, num_nodes):
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1
    return adj

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


def smooth_label_loss(data, out, alpha=0.9):
    adj = edgeindex2adj(data.edge_index, data.x.shape[0])
    lap = laplacian(adj, typ='normailzed')
    pred_z = out
    # print(pred_z.shape, lap.shape)
    # print(torch.mm(pred_z.t(), lap).shape)
    # print("laplacian has nan: ", torch.isnan(lap).any())
    # print(pred_z[:5])
    # print(torch.mm(pred_z.t(), lap)[:5, :5])
    loss_1 = torch.trace(torch.mm(torch.mm(pred_z.t(), lap), pred_z))
    loss_2 = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    return 0.000005 * loss_1 + loss_2

def train(model, data, optimizer, loss='cross_entropy'): # train for one epoch
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    if loss == 'smooth_label':
        loss = smooth_label_loss(data, out)
    elif loss == 'cross_entropy':
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    else:
        raise ValueError('invalid loss function')
    loss.backward()
    optimizer.step()
    return loss

def test(model, data): # get accuracy on train, val, and test sets
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

def modify_adj(adj, features):
    def feat_dist(feat1, feat2):
        return torch.norm(feat1 - feat2, dim=-1)
    def graph_dist(adj, i, j):
        _, lap = get_laplacian(data.edge_index, normalization=None) # 'sym', 'rw'
        return lap[i, j]