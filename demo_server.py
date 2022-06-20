from flask import Flask, request, jsonify
from transformers import ElectraTokenizerFast, ElectraForTokenClassification, ElectraTokenizer
from transformers import TokenClassificationPipeline
import torch
import time
import os, psutil

process = psutil.Process(os.getpid())
count = 0
spent_sum = 0
start_memory = 0
app = Flask(__name__)
pipeline = None

def init_pipeline():
    global pipeline
    # init config
    tokenizer = ElectraTokenizerFast.from_pretrained('monologg/koelectra-base-v3-discriminator')
    model = ElectraForTokenClassification.from_pretrained("model")
    pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer, framework='pt')

@app.route('/pii_demo', methods=['POST'])
def pii_demo():
    global pipeline
    global spent_sum
    global count
    global start_memory
    
    with torch.no_grad():
        count += 1
        start = time.time()
        lines = request.get_json()["lines"]
        if pipeline == None:
            return "Server not ready"
    
        if not lines:
            return "Empty sentences requested"
        metas = pipeline(lines * 10, batch_size=32)
        spent = round(time.time() - start, 3)
        spent_sum += spent
        print("spent", spent)
        print("spent_avg", round(spent_sum / count, 2))
        memory = round(process.memory_info().rss / 1024 ** 2, 3)
        if count == 1:
            start_memory = memory
        print("memory usages:", memory)
        print("memory start, end, differencies:", start_memory, memory, round(memory - start_memory, 3))
        return {
            "memory": memory,
            "spent": spent
        }
    
if __name__ == '__main__':
    init_pipeline()
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True)