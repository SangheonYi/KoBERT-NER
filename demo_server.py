from crypt import methods
from flask import Flask
from flask import request
from predict import *
from utils import init_logger
app = Flask(__name__)

@app.route('/pii_demo', methods=['POST'])
def pii_demo():
    init_logger()
    parser = argparse.ArgumentParser()
    lines = request.get_json()["lines"]
    print(lines)
    parser.add_argument("--model_dir", default="model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    return predict(pred_config, lines)

if __name__ == '__main__':
    app.run(debug=True)