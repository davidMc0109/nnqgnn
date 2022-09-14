import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torchvision.models import resnet18
from traced_all_fixbit.prepare_by_platform import gnn_prepare_by_platform
from mqbench.prepare_by_platform import BackendType, prepare_by_platform

model = resnet18(True)
model_quant = resnet18(True)

my_config = {
    'extra_qconfig_dict':{
        'w_observer': 'ClipStdObserver',
        'a_observer': 'ClipStdObserver',
        'w_fakequantize': 'DSQFakeQuantize',
        'a_fakequantize': 'DSQFakeQuantize',
        'w_qscheme': {
            'bit': 4,
            'symmetry': False,
            'per_channel': True,
            'pot_scale': True
        },
        'a_qscheme': {
            'bit': 4,
            'symmetry': False,
            'per_channel': True,
            'pot_scale': True
        }
    }
}

# model_quant = prepare_by_platform(model_quant, BackendType.Academic, my_config)
#
# hook = 0
#
# x = torch.randn(1, 3, 224, 224)
# y = model_quant(x)

hook = 1

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                             normalize=not True)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                             normalize=not True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

hook = 2

import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
if True:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)

model_gnn = Net()
y = model_gnn(data)


hook = 2.5

model_gnn_quant = Net()
from torch_geometric.nn.fx import Transformer
model_gnn_quant = Transformer(model_gnn_quant).transform()
model_gnn_quant = gnn_prepare_by_platform(model_gnn_quant, BackendType.Academic, my_config)

hook = 3
y = model_gnn_quant(data)

