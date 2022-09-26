import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import torch.fx
from mqbench.prepare_by_platform import BackendType
from utils.prepare_by_platform import gnn_prepare_by_platform
from mqbench.utils.state import enable_calibration, enable_quantization

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
dataset = TUDataset(path, name='MUTAG').shuffle()

train_dataset = dataset[len(dataset) // 10:]
test_dataset = dataset[:len(dataset) // 10]

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

bit_width = -1


class Net(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels):
        super(Net, self).__init__()

        self.conv1 = GINConv(
            Sequential(Linear(in_channels, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv4 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv5 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

my_qconfig = {
    'extra_qconfig_dict':{
        # 'w_observer': 'ClipStdObserver',
        # 'a_observer': 'ClipStdObserver',
        # 'w_fakequantize': 'DSQFakeQuantize',
        # 'a_fakequantize': 'DSQFakeQuantize',
        'w_observer': 'MinMaxObserver',
        'a_observer': 'EMAMinMaxObserver',
        'w_fakequantize': 'LearnableFakeQuantize',
        'a_fakequantize': 'LearnableFakeQuantize',
        'w_qscheme': {
            'bit': bit_width,
            'symmetry': True,
            'per_channel': False,
            'pot_scale': True
        },
        'a_qscheme': {
            'bit': bit_width,
            'symmetry': True,
            'per_channel': False,
            'pot_scale': True
        }
    }
}
torch.fx.wrap(global_add_pool)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, 32, dataset.num_classes).to(device)
if bit_width>0:
    model = gnn_prepare_by_platform(model, BackendType.Academic, my_qconfig).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        total_correct += int((out.argmax(-1) == data.y).sum())
    return total_correct / len(loader.dataset)


for epoch in range(1, 101):
    if bit_width>0:
        if epoch == 1:
            enable_calibration(model)
        elif epoch == 5:
            enable_quantization(model)
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f} '
          f'Test Acc: {test_acc:.4f}')
