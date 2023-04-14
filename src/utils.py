import torch.nn.functional as F
import torch
from torch_geometric.utils import get_laplacian

def smooth_label_loss(data, out, alpha=0.9):
    _, lap = get_laplacian(data.edge_index, normalization=None) # 'sym', 'rw'
    pred_z = out
    print(pred_z.shape, lap.shape)
    loss_1 = torch.trace(torch.mm(torch.mm(pred_z, lap), pred_z.t()))[data.train_mask]
    loss_2 = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    return loss_1 + alpha * loss_2

def train(model, data, optimizer, loss='cross_entropy'): # train for one epoch
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    if loss == 'smooth_label':
        loss = smooth_label_loss(data, out)
    if loss == 'cross_entropy':
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

