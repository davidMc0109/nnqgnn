import copy

from mqbench.custom_quantizer import AcademicQuantizer
from mqbench.utils import is_symmetric_quant
from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType
from torch.ao.quantization import propagate_qconfig_
from torch.fx import GraphModule

from mqbench.utils.logger import logger
from torch.quantization.fx.qconfig_utils import get_flattened_qconfig_dict


@register_model_quantizer(BackendType.Academic)
class LPAcademicQuantizer(AcademicQuantizer):
    def __init__(self, *args, **kwargs):
        super(LPAcademicQuantizer, self).__init__(*args, **kwargs)

    def _weight_quant(self, model: GraphModule, qconfig):
        logger.info("Replace module to qat module.")
        wqconfig_8bit = copy.deepcopy(qconfig)
        wq_symmetry = True if is_symmetric_quant(qconfig.weight.p.keywords['qscheme']) else False
        # wqconfig_8bit.weight.p.keywords['quant_min'] = -2 ** (8 - 1) if wq_symmetry else 0
        # wqconfig_8bit.weight.p.keywords['quant_max'] = 2 ** (8 - 1) - 1 if wq_symmetry else 2 ** 8 - 1
        for name, module in model.named_modules():
            if name in self.io_module.keys():
                logger.info("Set layer {} to 8 bit.".format(name))
                module.qconfig = wqconfig_8bit
        flattened_qconfig_dict = get_flattened_qconfig_dict({'': qconfig})
        propagate_qconfig_(model, flattened_qconfig_dict)
        self._qat_swap_modules(model, self.additional_qat_module_mapping)
        return model