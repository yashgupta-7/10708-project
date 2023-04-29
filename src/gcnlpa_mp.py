import torch
import random
import numpy as np
import argparse

import torch
from torch import Tensor
from torch_geometric.logging import init_wandb, log
from torch_geometric.datasets import Planetoid
from utils import train, test, non_smooth_label_metric
from models import GCN, GAT, LP
import torch.nn.functional as F
import multiprocessing as mp
from models import AdaptiveLP
import copy

citeseer = Planetoid(root='.', name='Citeseer')
cora = Planetoid(root='.', name='Cora')
pubmed = Planetoid(root='.', name='Pubmed')
torch.use_deterministic_algorithms(True)

def run_analysis(dataset_type='cora', model_type = 'GCN', k = 5, lpa = True):
    seeds = [1234]
    lr = 0.02
    epochs = 200
    if dataset_type == 'cora':
        dataset = copy.deepcopy(cora)
    elif dataset_type == 'citeseer':
        dataset = copy.deepcopy(citeseer)
    elif dataset_type == 'pubmed':
        dataset = copy.deepcopy(pubmed)
    else :
        print("Dataset not found")
        return
    if model_type == 'GCN':
        model = GCN(dataset.num_features, 16, dataset.num_classes)
    elif model_type == 'GAT':
        model = GAT(dataset.num_features, 8, dataset.num_classes, heads=8)

    torch.manual_seed(0)
    data = dataset[0]
    for c in data.y.unique():
        idx = ((data.y == c) & data.train_mask).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))]
        idx = idx[k:]
        data.train_mask[idx] = False
    lp = AdaptiveLP(num_layers=8, yshape=dataset[0].y.shape[0], edge_dim=dataset.edge_index.shape[1])
    av_val_acc = av_test_acc = 0
    state_dict_model = model.state_dict().copy()
    state_dict_lp = lp.state_dict().copy()

    for seed in seeds:
        print("RUNNING FOR SEED =", seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        model.load_state_dict(state_dict_model)
        lp.load_state_dict(state_dict_lp)
        
        optimizer = torch.optim.Adam(list(model.parameters()) + list(lp.parameters()), lr=lr, weight_decay=5e-4)

        best_val_acc = final_test_acc = 0
        if lpa:
            for epoch in range(1, 200):
                model.train()
                lp.train()
                optimizer.zero_grad()
                
                out_model = model(data.x, data.edge_index)
                out_lp = lp(data)
                
                loss_model = F.cross_entropy(out_model[data.train_mask], data.y[data.train_mask])
                # loss_lp = (out_lp[data.train_mask] - data.y[data.train_mask]).pow(2).mean()
                loss_lp = F.cross_entropy(out_lp[data.train_mask], data.y[data.train_mask])
                ##########################################
                # sample some nodes from the unlabelled set
                unlab_mask = ~data.train_mask & ~data.val_mask & ~data.test_mask
                unlab_idx = unlab_mask.nonzero(as_tuple=False).view(-1)
                sample_unlab_idx = unlab_idx[torch.rand(unlab_idx.shape[0]) < 0.005]
                sample_unlab_mask = torch.zeros(unlab_mask.shape[0], dtype=torch.bool)
                sample_unlab_mask[sample_unlab_idx] = True
                
                # loss_unsup = F.cross_entropy(out_model[sample_unlab_mask], out_lp[sample_unlab_mask].argmax(dim=1))
                loss_unsup = F.cross_entropy(out_lp[sample_unlab_mask], out_model[sample_unlab_mask].argmax(dim=1))
                ##########################################
                # print(loss_model, loss_lp, loss_unsup)
                loss = loss_model + 2 * loss_lp + loss_unsup
                loss.backward()
                optimizer.step()
                
                train_acc, val_acc, tmp_test_acc = test(model, data)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                if epoch % 25 == 0:
                    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-2)
        for epoch in range(1, 200):
            loss = train(model, data, optimizer, loss='cross_entropy')
            train_acc, val_acc, tmp_test_acc = test(model, data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            if epoch % 25 == 0:
                log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
        
        print(f'Best Val Acc: {best_val_acc:.4f}', f'Test Acc: {test_acc:.4f}')
        av_val_acc += best_val_acc
        av_test_acc += test_acc
        
    print(f'Average Val Acc / Average Test Acc: {av_val_acc / len(seeds):.4f} / {av_test_acc / len(seeds):.4f}')
    preds = model(data.x, data.edge_index).argmax(dim=1)
    # save model
    if lpa:
        torch.save(model.state_dict(), f'./models/{dataset.name}_{model_type}_LPA_{k}.pt')
    else:
        torch.save(model.state_dict(), f'./models/{dataset.name}_{model_type}_{k}.pt')
    metric = non_smooth_label_metric(dataset, preds)
    return av_val_acc / len(seeds), av_test_acc / len(seeds), metric


if __name__=='__main__':
    pool = mp.Pool(8)
    results = pool.starmap(run_analysis, [('citeseer', 'GAT', k_i, True) for k_i in [1,2,5,10,20]])
    pool.close()
    pool.join()
    print(results)
    file = open('results.txt', 'a')
    for line in results:
        file.write(str(line) + '\n')
        file.write(f"{line[0]:.4f} / {line[1]:.4f}\n")
        file.write(str(line[2]) + '\n')
    file.close()

    for i in results:
        print(f"{i[0]:.4f} / {i[1]:.4f}")
    for i in results:
        print(i[2])

    # print(run_analysis(cora, 'GCN', 5))