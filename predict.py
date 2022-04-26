import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForTokenClassification
from output_analysis import answer_diff

from utils import init_logger, load_tokenizer, get_labels

from kss import split_sentences

logger = logging.getLogger(__name__)

label_list = [
'UNK',
'O',
'SS_NAME-B',
'SS_NAME-I',
'SS_WEIGHT-B',
'SS_WEIGHT-I',
'SS_AGE-B',
'SS_AGE-I',
'AD_METRO-B',
'AD_METRO-I',
'AD_ADDRESS-B',
'AD_ADDRESS-I',
'AD_CITY-B',
'AD_CITY-I',
'AD_DETAIL-B',
'AD_DETAIL-I',
'ID_PHONE-B',
'ID_PHONE-I',
'ID_CARD-B',
'ID_CARD-I',
'ID_ACCOUNT-B',
'ID_ACCOUNT-I',
'ID_INUM-B',
'ID_INUM-I',
'SS_LENGTH-B',
'SS_LENGTH-I',
'SS_BIRTH-B',
'SS_BIRTH-I',
'SS_BRAND-B',
'SS_BRAND-I',
]

def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, 'training_args.bin'))


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)  # Config will be automatically loaded from model_dir
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def read_input_file(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            # sentences = split_sentences(line)
            # for sentence in sentences:
            stripped = line.strip()
            words = stripped.split()
            lines.append(words)

    return lines


def convert_input_file_to_tensor_dataset(lines,
                                         pred_config,
                                         args,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    all_input_tokens = []

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            # slot_label_mask.extend([0] + [pad_token_label_id] * (len(word_tokens) - 1))
            
            # use the real label id for all tokens of the word
            slot_label_mask.extend([0] * (len(word_tokens)))

        all_input_tokens.append(tokens)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)

        input_ids = input_ids + ([pad_token_id] * padding_length)

        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)


    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return dataset, all_input_tokens


def predict(pred_config):
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)

    model = load_model(pred_config, args, device)
    label_lst = get_labels(args)
    logger.info(args)

    # Convert input file to TensorDataset
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    tokenizer = load_tokenizer(args)
    lines = read_input_file(pred_config)
    dataset, all_input_tokens = convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    all_slot_label_mask = None
    preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            if args.model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    # for pred in preds:
    i = 0
    prob_matrix = []
    with open('res/logit_prob.py', 'w', newline='', encoding='UTF-8') as prob_file:
        for sentence_pred, tokens in zip(torch.softmax(input=torch.tensor(preds), dim=2).numpy(), all_input_tokens):
            prob_file.write(f"👩{tokens}\n")
            token_probs = []
            for i, token_pred in enumerate(sentence_pred):
                if 0 < i < len(tokens):
                    tmp = [[label_list[i], round(e * 100, 2)] for i, e in enumerate(token_pred)]
                    tmp.sort(key=lambda x: x[1],reverse=True)
                    prob_file.write(f"{tmp[:4]}\n")
                    token_probs.append(tmp[:4])
            prob_matrix.append(token_probs)
        i += 1

    preds = np.argmax(preds, axis=2)

    slot_label_map = {i: label for i, label in enumerate(label_lst)}
    preds_list = [[] for _ in range(preds.shape[0])] # [[]*batch 수] 
    
    slot_label_map = {i: label for i, label in enumerate(label_lst)}

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                preds_list[i].append(slot_label_map[preds[i][j]])

    ################# per token
    # Write to output file
    with open(pred_config.output_file, "w", encoding="utf-8") as f:
        output_tsv_form = []
        for words, preds in zip(all_input_tokens, preds_list):
            line = ""
            for word, pred in zip(words, preds):
                if pred == 'O':
                    line = line + word + " "
                else:
                    line = line + "[{}:{}] ".format(word, pred)
            output_tsv_form.append(f"{' '.join(words[:-1])}\t{' '.join(preds)}")
            f.write("{}\n".format(line.strip()))
        answer_diff(output_tsv_form, prob_matrix, "ALL")
  
    logger.info("Prediction Done!")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="sample_pred_in.txt", type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="sample_pred_out.txt", type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    predict(pred_config)
