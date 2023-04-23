import torch.nn.functional as F
import torch
from torch_geometric.utils import get_laplacian

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
    pred_z = out
    # data.edge_index = adj2edgeindex(adj)
    loss_1 = torch.trace(torch.mm(torch.mm(pred_z.t(), lap), pred_z))
    loss_2 = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss_3 = -torch.norm(adj)**2
    return 0.00005 * loss_1 + loss_2 + 0.0 * loss_3

def train(model, data, optimizer, loss='cross_entropy'): # train for one epoch
    model.train()
    optimizer.zero_grad()
    adj = edgeindex2adj(data.edge_index, data.x.shape[0])
    out = model(data.x, data.edge_index)
    if loss == 'smooth_label':
        loss = smooth_label_loss(data, out, adj)
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