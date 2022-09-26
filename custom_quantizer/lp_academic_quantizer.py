import copy

from mqbench.custom_quantizer import AcademicQuantizer
from mqbench.utils import is_symmetric_quant
from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType
from torch.ao.quantization import propagate_qconfig_
from torch.fx import GraphModule

from mqbench.utils.logger import logger
from torch.quantization.fx.qconfig_utils import get_flattened_qconfig_dict
import math


def replace_lp_academic_quantizer(enable=True):
    if enable:
        register_model_quantizer(BackendType.Academic)(LPAcademicQuantizer)
    else:
        register_model_quantizer(BackendType.Academic)(AcademicQuantizer)


@register_model_quantizer(BackendType.Academic)
class LPAcademicQuantizer(AcademicQuantizer):
    def __init__(self, *args, **kwargs):
        super(LPAcademicQuantizer, self).__init__(*args, **kwargs)

    def _weight_quant(self, model: GraphModule, qconfig):
        logger.info("Replace module to qat module.")
        wqconfig = copy.deepcopy(qconfig)
        for name, module in model.named_modules():
            if name in self.io_module.keys():
                logger.info(
                    "Set layer {} to {} bit.".format(name,
                                                     math.ceil(math.log2(
                                                         qconfig.weight.p.keywords['quant_max'] -
                                                         qconfig.weight.p.keywords['quant_min']))))
                module.qconfig = wqconfig
        flattened_qconfig_dict = get_flattened_qconfig_dict({'': qconfig})
        propagate_qconfig_(model, flattened_qconfig_dict)
        self._qat_swap_modules(model, self.additional_qat_module_mapping)
        return model
