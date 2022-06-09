from transformers import ElectraTokenizerFast, ElectraTokenizer
import time

from transformers import ElectraTokenizerFast, ElectraForTokenClassification
from transformers import TokenClassificationPipeline

element = "특히 둘째 출산 후 몸무게가 63kg까지 치솟았다고 밝혔지만 체중 증가가 무색할 정도로 변함없는 비주얼을 자랑해 감탄을 안겼다.\n"

def check_time_tokenizer(tokenizer, count):
    input = [element for _ in range(count)]
    start = time.time()
    tokenizer(input, padding='max_length', max_length=100)
    return round(time.time() - start, 6)

def check_time_pipeline(pipeline, count):
    input = [element for _ in range(count)]
    start = time.time()
    pipeline(input)
    return round(time.time() - start, 6)

def ten_power_of_n_loop(target, checker, start=0, stop=4):
    record_time = []
    for i in range(start, stop):
        record_time.append(checker(target, 1 * 10**i))
    return record_time
    
if __name__ == '__main__':
    # tokenizer = ElectraTokenizerFast.from_pretrained('monologg/koelectra-base-v3-discriminator')


    # pipeline test
    tokenizer = ElectraTokenizerFast.from_pretrained('monologg/koelectra-base-v3-discriminator')
    model = ElectraForTokenClassification.from_pretrained("model")
    pipeline = TokenClassificationPipeline(task="ner", model=model, tokenizer=tokenizer)
    input = [element for _ in range(100)]

    print(ten_power_of_n_loop(pipeline, check_time_pipeline, start=0, stop=3))