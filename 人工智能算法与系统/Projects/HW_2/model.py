import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from tqdm import tqdm

class GAT(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, dropout, 
        layer_heads=[]
    ):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()

        self.convs.append(
            GATv2Conv(
                in_channels, 
                hidden_channels, 
                heads=layer_heads[0], 
                concat=True
            )
        )

        self.convs.append(
            GATv2Conv(
                hidden_channels * layer_heads[0], 
                out_channels, 
                heads=layer_heads[1], 
                concat=False
            )
        )

        self.dropout = dropout

    def reset(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != 2 - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x.log_softmax(dim=-1)

    def inference(self, x_all, layer_loader, device):

        pbar = tqdm(total=x_all.size(0) * 2, ncols=80)
        pbar.set_description('Evaluating')
        
        for i in range(2):
            xs = []
            for batch_size, n_id, adj in layer_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != 2 - 1:
                    x = F.relu(x)
                xs.append(x)

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all.log_softmax(dim=-1)
