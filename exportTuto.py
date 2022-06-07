from typing import Mapping, OrderedDict, List
from transformers import ElectraConfig, ElectraTokenizer, AutoModelForTokenClassification
from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import config, OnnxConfig, export
from pathlib import Path

class PIIELECTRAConfig(OnnxConfig):
    def __init__(self, config: PretrainedConfig, task: str = "default", patching_specs: List[config.PatchingSpec] = None, use_past: bool = False):
        super().__init__(config, "token-classification", patching_specs)

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("token_type_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
                ("labels", {0: "batch", 1: "sequence"}),
            ]
        )
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("logits", {0: "batch", 1: "sequence"}),
            ]
        )


                
if __name__ == '__main__':
    onnx_path = Path('monologg/koelectra-base-v3-discriminator')
    tokenizer = ElectraTokenizer.from_pretrained(onnx_path)
    base_model = AutoModelForTokenClassification.from_pretrained('model/')
    onnx_config = PIIELECTRAConfig(ElectraConfig.from_pretrained('model/'))
    out_path = Path("model/exported.onnx")
    onnx_inputs = export(tokenizer, base_model, onnx_config, onnx_config.default_onnx_opset, out_path)

