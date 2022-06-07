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
    ort_inputs = tokenizer("제발 이상헌 퇴근 전에 되라", return_tensors="np")
    print(ort_inputs)
    # 모델 run
    result = ort_session.run(output_names=["logits"], input_feed=dict(ort_inputs))
    print(result)
