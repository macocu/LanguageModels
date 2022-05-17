#!/usr/bin/env python

'''Fine-tune a language model for token classification (e.g. NER, POS)'''

import sys
import os
import argparse
import json
from simpletransformers.ner import NERModel


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", required=True, type=str,
                        help="Input file to learn from")
    parser.add_argument("-d", "--dev_file", type=str, required=True,
                        help="Separate dev set to evaluate on")
    parser.add_argument("-lt", "--lm_type", type=str, default="xlmroberta",
                        help="Simpletransformers LM type identifier")
    parser.add_argument("-li", "--lm_ident", type=str, default="xlm-roberta-large",
                        help="Language model identifier (default XLM-R OR location of the folder with the trained model")
    # Arguments for training a model
    parser.add_argument("-a", "--arg_dict", type=str,
                        help="Optional json dict with extra arguments (recommended). Otherwise use our default settings")
    args = parser.parse_args()
    return args


def get_train_args():
    '''Default training arguments, recommended to overwrite them with your own arguments
       They differ a bit from the simpletransformers default argument, so beware!'''
    return {
    "output_dir": "outputs/",
    "cache_dir": "cache_dir/",
    "fp16": True,
    "fp16_opt_level": "O1",
    "max_seq_length": 512,
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "eval_batch_size": 8,
    "num_train_epochs": 3,
    "weight_decay": 0,
    "learning_rate": 4e-5,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "logging_steps": 50,
    "save_steps": 2000,
    "overwrite_output_dir": True,
    "reprocess_input_data": False,
    "evaluate_during_training": False,
    "early_stopping_patience": 2,
    "process_count": 8,
    "n_gpu": 1,
    }


def load_json_dict(d):
    '''Funcion that loads json dictionaries'''
    with open(d, 'r') as in_f:
        dic = json.load(in_f)
    in_f.close()
    return dic


def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()
    # Load our own training arguments, or use the default ones
    train_args = get_train_args() if not args.arg_dict else load_json_dict(args.arg_dict)

    # It's necessary to get the labels first, to overwrite the default NER ones
    init_labels = [x.split()[-1].strip() for x in open(args.train_file, 'r', encoding="utf-8") if x.strip() and not x.strip().startswith("#")]
    init_dev_labels = [x.split()[-1].strip() for x in open(args.dev_file, 'r', encoding="utf-8") if x.strip() and not x.strip().startswith("#")]
    # Always load them in the same order, without set()
    labels = []
    for item in init_labels + init_dev_labels:
        if item not in labels:
            labels.append(item)
    print ("Labels:\n", labels)
    train_args["label_list"] = labels

    # Create the model
    model = NERModel(args.lm_type, args.lm_ident, labels=labels, use_cuda=True, args=train_args)

    # Model can train directly from input files
    model.train_model(args.train_file, args=train_args, labels=labels)
    result, model_outputs, predictions = model.eval_model(args.dev_file)

    # Print the metrics
    print (result)

if __name__ == '__main__':
    main()
