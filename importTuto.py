from transformers import ElectraTokenizer
import onnx
import onnxruntime

if __name__ == '__main__':
    onnx_model = onnx.load('model/exported.onnx')
    # 모델 구조 확인, 스키마의 유효성 체크
    onnx.checker.check_model(onnx_model)

    # 추론 세션 생성
    ort_session = onnxruntime.InferenceSession('model/exported.onnx')

    # ONNX 런타임에서 계산된 결과값
    tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
    ort_inputs = tokenizer(["특히 둘째 출산 후 몸무게가 63kg까지 치솟았다고 밝혔지만 체중 증가가 무색할 정도로 변함없는 비주얼을 자랑해 감탄을 안겼다.\n"], padding='max_length', max_length=100, return_tensors='np')
    input_feed = {"input_ids": ort_inputs['input_ids'],
                "attention_mask": ort_inputs['attention_mask'],
                "token_type_ids": ort_inputs['token_type_ids']
                }
    print(ort_inputs)
    # 모델 run
    result = ort_session.run(output_names=["logits"], input_feed=dict(ort_inputs))
    print(result)
