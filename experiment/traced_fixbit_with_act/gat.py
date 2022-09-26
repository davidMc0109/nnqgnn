import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from mqbench.prepare_by_platform import BackendType
from utils.prepare_by_platform import gnn_prepare_by_platform
from mqbench.utils.state import enable_quantization, enable_calibration

bit_width = 2
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
data.is_directed()


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()

        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


my_qconfig = {
    'extra_qconfig_dict':{
        'w_observer': 'MinMaxObserver',     'w_fakequantize': 'LearnableFakeQuantize',
        'a_observer': 'EMAMinMaxObserver',  'a_fakequantize': 'LearnableFakeQuantize',
        'w_qscheme': {
            'bit': bit_width, 'symmetry': True, 'per_channel': False, 'pot_scale': True
        },
        'a_qscheme': {
            'bit': bit_width, 'symmetry': True, 'per_channel': False, 'pot_scale': True
        }
    }
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, dataset.num_classes).to(device)
model = gnn_prepare_by_platform(model, BackendType.Academic, my_qconfig).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    if epoch == 200:
        hook = 0
        pass
    elif epoch == 0:
        enable_calibration(model)
    elif epoch == 5:
        enable_quantization(model)
    train(data)
    train_acc, val_acc, test_acc = test(data)
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')
