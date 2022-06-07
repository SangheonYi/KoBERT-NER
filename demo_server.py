from flask import Flask, request, jsonify
from predict import *
import argparse

from utils import load_tokenizer, get_labels, init_logger

app = Flask(__name__)
server = None

def init_server():
    # init config
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    pred_config = parser.parse_args()

    # load model, tokenizer and args
    args = get_args(pred_config)
    device = get_device(pred_config)

    model = load_model(pred_config, args, device)
    label_lst = get_labels(args)
    logger.info(args)

    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    tokenizer = load_tokenizer(args)

    return {
        "pred_config" : pred_config, 
        "args" : args, 
        "device" : device,
        "model" : model,
        "label_lst" : label_lst,
        "pad_token_label_id" : pad_token_label_id,
        "tokenizer" : tokenizer,
    }

@app.route('/pii_demo', methods=['POST'])
def pii_demo():
    if server == None:
        return "Model config isn't loaded"
    lines = request.get_json()["lines"]
    if not lines:
        return "Empty strings requested"
    pii_metas = predict(lines, **server)
    for line, result in zip(lines, pii_metas):
        for meta in result:
            line = line.replace(meta["token"], f'[{meta["token"]}:{meta["label"]}]', meta["start"])
    json_data = jsonify({"result" : pii_metas})
    # print(json_data.json)
    return json_data
    
if __name__ == '__main__':
    server = init_server()
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True)