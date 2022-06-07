import torch.onnx
from typing import Mapping, OrderedDict, List
from transformers import ElectraConfig, ElectraTokenizer, AutoModelForTokenClassification
from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import config, OnnxConfig, export
from pathlib import Path
import onnx
import onnxruntime
import numpy as np



if __name__ == '__main__':
    onnx_model = onnx.load('model/exported.onnx')
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession('model/exported.onnx')

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # ONNX 런타임에서 계산된 결과값
    tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
    ort_inputs = tokenizer("제발 이상헌 퇴근 전에 되라", return_tensors="np")
    # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
    print(ort_inputs)
    result = ort_session.run(output_names=["logits"], input_feed=dict(ort_inputs))

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
