import copy
from typing import Any, Dict

import torch
import torch_geometric.nn.dense
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from lp_academic_quantizer import LPAcademicQuantizer
from mqbench.utils.registry import register_model_quantizer
from torch.fx import Tracer
from torch_geometric.nn import MessagePassing

__all__ = ['gnn_prepare_by_platform']


def gnn_prepare_by_platform(
        model: torch.nn.Module,
        deploy_backend: BackendType,
        prepare_custom_config_dict: Dict[str, Any] = {},
        custom_tracer: Tracer = None):
    if deploy_backend == BackendType.Academic:
        register_model_quantizer(deploy_backend)(LPAcademicQuantizer)

    class GNNCustomedTracer(Tracer):
        def __init__(self, *args, **kwargs):
            super(GNNCustomedTracer, self).__init__(*args, **kwargs)

        def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
            # if the module is a MessagePassing, do not tracing in
            if isinstance(m, MessagePassing):
                return True
            return super(GNNCustomedTracer, self).is_leaf_module(m, module_qualified_name)

    if not ('extra_quantizer_dict' in prepare_custom_config_dict):
        prepare_custom_config_dict['extra_quantizer_dict'] = {}
    if not ('additional_module_type' in prepare_custom_config_dict['extra_quantizer_dict']):
        prepare_custom_config_dict['extra_quantizer_dict']['additional_module_type'] = tuple()
    prepare_custom_config_dict['extra_quantizer_dict']['additional_module_type'] += (MessagePassing,)

    tracer = GNNCustomedTracer()
    if custom_tracer is not None:
        tracer = custom_tracer
    graph_model = prepare_by_platform(model, deploy_backend, prepare_custom_config_dict, tracer)
    hook = 0

    # replace fp to int
    from torch.quantization import swap_module
    from mqbench.utils.registry import DEFAULT_MODEL_QUANTIZER
    from mqbench.fuser_method_mappings import fuse_custom_config_dict
    from torch.ao.quantization.quantization_mappings import get_default_static_quant_module_mappings
    from mqbench.utils.logger import logger
    from mqbench.fake_quantize.quantize_base import QuantizeBase
    extra_fuse_dict = prepare_custom_config_dict.get('extra_fuse_dict', {})
    extra_fuse_dict.update(fuse_custom_config_dict)
    extra_quantizer_dict = prepare_custom_config_dict.get('extra_quantizer_dict', {})
    # extra_quantizer_dict['additional_module_type'] = (MessagePassing,)
    QUANTIZER_CLASS = DEFAULT_MODEL_QUANTIZER[deploy_backend]

    class ModifiedQuantized(QUANTIZER_CLASS):
        def __init__(self, *args, **kwargs):
            super(ModifiedQuantized, self).__init__(*args, **kwargs)

        def _convert(self, module, mapping=None, inplace=False, scope=''):
            if mapping is None:
                mapping = get_default_static_quant_module_mappings()
            module = swap_module(module, mapping, {})
            super(ModifiedQuantized, self)._convert(module, mapping, inplace, scope)
            return module

    class NonableFakeActQuantizeWrapper(torch.nn.Module):
        def __init__(self, fake_act_quantize):
            super(NonableFakeActQuantizeWrapper, self).__init__()
            self.fake_act_quantize = fake_act_quantize

        def forward(self, X):
            if X is None:
                return X
            elif not X.dtype == torch.float32:
                return X
            else:
                return self.fake_act_quantize(X)

    quantizer = ModifiedQuantized(extra_quantizer_dict, extra_fuse_dict)

    graph = graph_model.graph
    modules = dict(graph_model.named_modules())
    for node in graph.nodes:
        if node.op == 'call_module':
            module = modules[node.target]
            if isinstance(module, MessagePassing):
                # found a MessagePassing, replace its Linear
                logger.info("Found a MessagePassing @ %s, replace its Linear" % (node.target,))
                sub_modules = dict(module.named_modules(prefix=node.target))
                for k in sub_modules.keys():
                    v = sub_modules[k]
                    if isinstance(v, torch_geometric.nn.dense.Linear):
                        nn_v = torch.nn.Linear(v.in_channels, v.out_channels, not (v.bias is None), v.weight.device,
                                            v.weight.dtype)
                        nn_v.load_state_dict(v.state_dict())
                        nn_v.qconfig = v.qconfig
                        sub_modules[k] = nn_v
                        nnq_v = quantizer._qat_swap_modules(nn_v, {})
                        module._modules[k.split('.')[-1]] = nnq_v
                        logger.info("Linear replaced for MessagePassing %s" % (k,))
                    elif isinstance(v, torch.nn.Sequential):
                        prepared_v = prepare_by_platform(v, deploy_backend, prepare_custom_config_dict, custom_tracer)
                        module._modules[k.split('.')[-1]] = prepared_v
                        logger.info("Sequence Quantized for MessagePassing %s" % (k,))
            elif isinstance(module, QuantizeBase):
                module = NonableFakeActQuantizeWrapper(module)
            modules[node.target] = module
    return torch.fx.GraphModule(modules, graph)
