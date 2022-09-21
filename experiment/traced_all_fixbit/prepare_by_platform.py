import inspect
import types
from typing import Any, Dict

import torch
import torch_geometric.nn.dense.linear
from mqbench.fuser_method_mappings import fuse_custom_config_dict
from mqbench.utils.logger import logger
from mqbench.utils.registry import DEFAULT_MODEL_QUANTIZER
from torch.fx import Tracer
from torch.fx.graph_module import GraphModule
from torch.quantization.quantize_fx import _swap_ff_with_fxff

__all__ = ['prepare_by_platform', 'gnn_prepare_by_platform']

from mqbench.prepare_by_platform import BackendType, CustomedTracer
from mqbench.prepare_by_platform import get_qconfig_by_platform, duplicate_reused_nodes, prepare_constant_dict


def prepare_by_platform(
        model: torch.nn.Module,
        deploy_backend: BackendType,
        prepare_custom_config_dict: Dict[str, Any] = {},
        custom_tracer: Tracer = None):
    """
    Args:
        model (torch.nn.Module):
        deploy_backend (BackendType):

    >>> prepare_custom_config_dict : {
            extra_qconfig_dict : Dict, Find explanations in get_qconfig_by_platform,
            extra_quantizer_dict: Extra params for quantizer.
            preserve_attr: Dict, Specify attribute of model which should be preserved
                after prepare. Since symbolic_trace only store attributes which is
                in forward. If model.func1 and model.backbone.func2 should be preserved,
                {"": ["func1"], "backbone": ["func2"] } should work.
            Attr below is inherited from Pytorch.
            concrete_args: Specify input for model tracing.
            extra_fuse_dict: Specify extra fusing patterns and functions.
        }

    """
    model_mode = 'Training' if model.training else 'Eval'
    logger.info("Quantize model Scheme: {} Mode: {}".format(deploy_backend, model_mode))

    # Get Qconfig
    extra_qconfig_dict = prepare_custom_config_dict.get('extra_qconfig_dict', {})
    qconfig = get_qconfig_by_platform(deploy_backend, extra_qconfig_dict)

    _swap_ff_with_fxff(model)
    # Preserve attr.
    preserve_attr_dict = dict()
    if 'preserve_attr' in prepare_custom_config_dict:
        for submodule_name in prepare_custom_config_dict['preserve_attr']:
            cur_module = model
            if submodule_name != "":
                cur_module = getattr(model, submodule_name)
            preserve_attr_list = prepare_custom_config_dict['preserve_attr'][submodule_name]
            preserve_attr_dict[submodule_name] = {}
            for attr in preserve_attr_list:
                preserve_attr_dict[submodule_name][attr] = getattr(cur_module, attr)
    # Symbolic trace
    concrete_args = prepare_custom_config_dict.get('concrete_args', None)
    customed_leaf_module = prepare_custom_config_dict.get('leaf_module', [])
    tracer = CustomedTracer(customed_leaf_module=tuple(customed_leaf_module))
    if custom_tracer is not None:
        tracer = custom_tracer
    # todo: works tracing, but not check if correct
    #       It works, but things is, the pyg conv use pyg.nn.dense.Linear instead of torch.nn.Linear
    #       That why the fakequant is not attached.
    #       Now the weight should works if we changes that fake linear
    #       in the future, we need to automate the replacement
    #       also, we need to find a way to insert fake quantization for act.:q!
    if isinstance(model, GraphModule):
        graph = model.graph
    else:
        graph = tracer.trace(model, concrete_args)

    name = model.__class__.__name__ if isinstance(model, torch.nn.Module) else model.__name__
    modules = dict(model.named_modules())
    graph, duplicated_modules = duplicate_reused_nodes(graph, modules)
    constant_nodes = prepare_constant_dict(graph, model)
    modules.update(duplicated_modules)
    modules.update(constant_nodes)
    graph_module = GraphModule(modules, graph, name)
    # Model fusion.
    extra_fuse_dict = prepare_custom_config_dict.get('extra_fuse_dict', {})
    extra_fuse_dict.update(fuse_custom_config_dict)
    # Prepare
    import mqbench.custom_quantizer  # noqa: F401
    extra_quantizer_dict = prepare_custom_config_dict.get('extra_quantizer_dict', {})
    quantizer = DEFAULT_MODEL_QUANTIZER[deploy_backend](extra_quantizer_dict, extra_fuse_dict)
    prepared = quantizer.prepare(graph_module, qconfig)
    # Restore attr.
    if 'preserve_attr' in prepare_custom_config_dict:
        for submodule_name in prepare_custom_config_dict['preserve_attr']:
            cur_module = prepared
            _type = type(model)
            if submodule_name != "":
                cur_module = getattr(prepared, submodule_name)
                _type = type(getattr(model, submodule_name))
            preserve_attr_list = prepare_custom_config_dict['preserve_attr'][submodule_name]
            for attr_name in preserve_attr_list:
                logger.info("Preserve attr: {}.{}".format(submodule_name, attr_name))
                _attr = preserve_attr_dict[submodule_name][attr_name]
                if inspect.ismethod(_attr):
                    _attr = types.MethodType(getattr(_type, attr_name), cur_module)
                setattr(cur_module, attr_name, _attr)
    return prepared


def gnn_prepare_by_platform(
        model: torch.nn.Module,
        deploy_backend: BackendType,
        prepare_custom_config_dict: Dict[str, Any] = {},
        custom_tracer: Tracer = None):
    # TODO: the version in compatible version of torch_geometric is not working, use another Transformer from newer repo
    # TODO: the newer version still can not traced inside MessagePassing, need to implement our own.
    # from torch_geometric.nn.fx import Transformer
    from fx import Transformer
    # class Transformer:
    #     def __init__(self, model):
    #         super(Transformer, self).__init__()
    #         self.model = model
    #         raise NotImplementedError("Not Implement")
    #
    #     def transform(self):
    #         raise NotImplementedError("Not Implement")

    model = Transformer(model).transform()
    model = replace_linear_pyg2torch(model)

    return prepare_by_platform(model, deploy_backend, prepare_custom_config_dict, custom_tracer)


def replace_linear_pyg2torch(fx_model):
    assert isinstance(fx_model, torch.fx.GraphModule)
    graph = fx_model.graph
    modules = dict(fx_model.named_modules())
    from torch.fx.experimental.optimization import replace_node_module

    for node in graph.nodes:
        if node.op == 'call_module':
            old_module = modules[node.target]
            if isinstance(old_module, torch_geometric.nn.dense.linear.Linear):
                weight = old_module.weight.clone().detach()
                bias = old_module.bias.clone().detach()
                new_module = torch.nn.Linear(weight.shape[1], weight.shape[0], not (bias is None), weight.device, weight.dtype)
                new_module.weight.data = weight
                new_module.bias.data = bias
                replace_node_module(node, modules, new_module)
    return torch.fx.GraphModule(fx_model, graph)
