from flask import Flask, request, jsonify
from transformers import ElectraTokenizerFast, ElectraForTokenClassification
from transformers import TokenClassificationPipeline
import torch
import time
import os, psutil
process = psutil.Process(os.getpid())

app = Flask(__name__)
pipeline = None

def init_pipeline():
    global pipeline
    # init config
    tokenizer = ElectraTokenizerFast.from_pretrained('monologg/koelectra-base-v3-discriminator')
    model = ElectraForTokenClassification.from_pretrained("model")
    pipeline = TokenClassificationPipeline(task="ner", model=model, tokenizer=tokenizer)

@app.route('/pii_demo', methods=['POST'])
def pii_demo():
    global pipeline
    if pipeline == None:
        return "Server not ready"
    lines = request.get_json()["lines"]
    if not lines:
        return "Empty sentences requested"
    with torch.no_grad():
        # start = time.time()
        sentence_metas = pipeline(lines)
        # pipeline("dkkkkddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
        # print(type(sentence_metas))
        # print(time.time() - start, flush=True)

        # respo = [
        #     [
        #         {"label" : pii_metas["entity"],
        #         'start': pii_metas['start'], 
        #         'end': pii_metas['end']} 
        #         for pii_metas in sentence_meta] 
        #         for sentence_meta in sentence_metas
        #     ]
        # json_data = jsonify({"result" : respo})
        print(len(lines), process.memory_info().rss / 1024 ** 2)
        return "json_data"
    
if __name__ == '__main__':
    init_pipeline()
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=False)