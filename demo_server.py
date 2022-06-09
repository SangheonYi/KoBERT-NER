from flask import Flask, request, jsonify

from transformers import ElectraTokenizerFast, ElectraForTokenClassification
from transformers import TokenClassificationPipeline
import time
app = Flask(__name__)
pipeline = None

def init_pipeline():
    # init config
    tokenizer = ElectraTokenizerFast.from_pretrained('monologg/koelectra-base-v3-discriminator')
    model = ElectraForTokenClassification.from_pretrained("model")
    pipeline = TokenClassificationPipeline(task="ner", model=model, tokenizer=tokenizer)

    return pipeline

@app.route('/pii_demo', methods=['POST'])
def pii_demo():
    if pipeline == None:
        return "Server not ready"
    lines = request.get_json()["lines"]
    if not lines:
        return "Empty sentences requested"
    start = time.time()
    pii_metas = pipeline(lines)
    print(time.time() - start)
    print(pii_metas)
    json_data = jsonify({"result" : pii_metas})
    # for line, result in zip(lines, pii_metas):
    #     for meta in result:
    #         line = line.replace(meta["token"], f'[{meta["token"]}:{meta["label"]}]', meta["start"])
    # print(json_data.json)
    return json_data
    
if __name__ == '__main__':
    pipeline = init_pipeline()
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True)