from model import GAT
from utils import DGraphFin
from utils.evaluator import Evaluator

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.loader import NeighborSampler

from tqdm import tqdm
import gc

def train(epoch, train_loader, model, data, train_idx, optimizer, device):
    model.train()

    pbar = tqdm(total=train_idx.size(0), ncols=80)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(data.x[n_id], adjs)
        loss = F.nll_loss(out, data.y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        pbar.update(batch_size)

        gc.collect()

    pbar.close()
    loss = total_loss / len(train_loader)

    return loss

@torch.no_grad()
def test(layer_loader, model, data, split_idx, evaluator, device):
    # data.y is labels of shape (N, ) 
    model.eval()

    out = model.inference(data.x, layer_loader, device)

    y_pred = out.exp()  # (N,num_classes)   

    losses, eval_results = dict(), dict()

    for key in ['train', 'valid']:
        node_id = split_idx[key]
        node_id = node_id.to(device)

        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
        eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]

    return eval_results, losses, y_pred

if __name__ == '__main__':
    path = './datasets/'  # 数据保存路径
    save_dir = './results/'  # 模型保存路径
    dataset_name = 'DGraph'
    model_name = 'GAT'
    eval_metric = 'auc'
    epochs = 200
    nlabels = 2
    log_steps = 10

    params = {
        'lr': 0.003,
        'hidden_channels': 128,
        'dropout': 0.0,
        'l2': 5e-6,
        'layer_heads': [4, 1]
    }

    # Set Device
    device = 0
    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Handle Data
    data = DGraphFin(
        root=path,
        name=dataset_name,
        transform=T.ToSparseTensor()
    )[0]
    data.adj_t = data.adj_t.to_symmetric()
    data.x = (data.x - data.x.mean(0)) / data.x.std(0)
    data.y = data.y.squeeze(1)

    split_idx = {
        'train': data.train_mask,
        'valid': data.valid_mask,
        'test': data.test_mask
    }

    # To Device
    data = data.to(device)
    train_idx = split_idx['train'].to(device)

    train_loader = NeighborSampler(
        data.adj_t,
        node_idx=train_idx,
        sizes=[10, 5],
        batch_size=1024,
        shuffle=True,
        num_workers=8
    )

    layer_loader = NeighborSampler(
        data.adj_t,
        node_idx=None,
        sizes=[-1],
        batch_size=1024,
        shuffle=False,
        num_workers=8
    )

    model_para = params.copy()
    model_para.pop('lr')
    model_para.pop('l2')

    # 需要改名字
    model = GAT(
        in_channels=data.x.size(-1),
        out_channels=nlabels,
        **model_para
    ).to(device)
    model.reset()

    print(f'Model {model_name} initialized')

    evaluator = Evaluator(eval_metric)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['l2']
    )

    min_valid_loss = 1e8

    for epoch in range(1, epochs + 1):
        loss = train(
            epoch, train_loader, model, data, train_idx, optimizer, device
        )

        eval_results, losses, out = test(
            layer_loader, model, data, split_idx, evaluator, device
        )

        train_eval, valid_eval = eval_results['train'], eval_results['valid']

        valid_loss = losses['valid']

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), save_dir + '/model.pt')  # 将表现最好的模型保存
            print('model saved')
            torch.save(out, save_dir + '/preds.pt')
            print('prediction saved')

        if epoch % log_steps == 0:
            print(
                f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                f'Train: {100 * train_eval:.3f}%, '
                f'Valid: {100 * valid_eval:.3f}% '
            )

        eval_results, losses, out = None, None, None
        train_eval, valid_eval = None, None
        valid_loss = None

        gc.collect()  # 垃圾回收
