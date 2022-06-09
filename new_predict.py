from transformers import ElectraTokenizerFast, ElectraForTokenClassification
from transformers import TokenClassificationPipeline


tokenizer = ElectraTokenizerFast.from_pretrained('monologg/koelectra-base-v3-discriminator')
encoding = tokenizer("꽱 몸무게가 63kg다.\n")
print(encoding.is_fast)
print(encoding.tokens())
print(encoding.word_ids())
model = ElectraForTokenClassification.from_pretrained("model")
pipeline = TokenClassificationPipeline(task="ner", model=model, tokenizer=tokenizer)
result = pipeline("꽱 몸무게가 63kg다.\n")
for e in result:
    print(e)