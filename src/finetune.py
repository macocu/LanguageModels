#!/usr/bin/env python

'''Fine-tune a language model for token classification (e.g. NER, POS)'''

import sys
import os
import argparse
import json
import random as python_random
import numpy as np
import torch
from simpletransformers.ner import NERModel


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", required=True, type=str,
                        help="Input file to learn from")
    parser.add_argument("-o", "--output_dir", required=True, type=str,
                        help="Output dir to write saved model and evaluations")
    parser.add_argument("-t", "--test_files", type=str, required=True, nargs="+",
                        help="Files to evaluate on - first one is used as dev set")
    parser.add_argument("-n", "--names", default=["dev", "test", "eval", "other"], nargs="+",
                        help="Names for dev/test files, just for printing, default will often work")
    parser.add_argument("-lt", "--lm_type", type=str, default="xlmroberta",
                        help="Simpletransformers LM type identifier")
    parser.add_argument("-li", "--lm_ident", type=str, default="xlm-roberta-large",
                        help="Language model identifier (specific LM identifier (default XLM-R) OR location of the folder with the trained model")
    parser.add_argument("-s", "--seed", type=int, default=2222,
                        help="Random seed that we use")
    # Arguments for training a model
    parser.add_argument("-a", "--arg_dict", type=str,
                        help="Optional json dict with extra arguments (recommended). Otherwise use our default settings")
    args = parser.parse_args()
    # Make reproducible as much as possible by setting the random seed everywhere
    # Not sure if it matters here, but do it either way
    np.random.seed(args.seed)
    python_random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return args


def get_train_args():
    '''Default training arguments, recommended to overwrite them with your own arguments
       They differ a bit from the simpletransformers default arguments, so beware!'''
    return {
    "cache_dir": "cache_dir/",
    "fp16": True,
    "fp16_opt_level": "O1",
    "max_seq_length": 512,
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "eval_batch_size": 8,
    "num_train_epochs": 10,
    "weight_decay": 0,
    "learning_rate": 4e-5,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "logging_steps": 25,
    "save_steps": 500000,
    "overwrite_output_dir": True,
    "reprocess_input_data": False,
    "evaluate_during_training": True,
    "evaluate_during_training_verbose": True,
    "evaluate_during_training_silent": False,
    "evaluate_each_epoch": True,
    "evaluate_during_training_steps": 500000,
    "early_stopping_consider_epochs": True,
    "use_early_stopping": True,
    "early_stopping_metric_minimize": False,
    "early_stopping_patience": 2,
    "early_stopping_metric": "f1_score",
    "save_model_every_epoch": False,
    "process_count": 8,
    "no_cache": True,
    "n_gpu": 1,
    "early_stopping_delta": 0.0005,
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
    # Load our own training arguments, or use the default ones (still specified in this file)
    train_args = get_train_args() if not args.arg_dict else load_json_dict(args.arg_dict)

    # It's necessary to get the labels first, to overwrite the default NER ones
    labels = []
    for in_file in [args.train_file] + args.test_files:
        cur_labels = [x.split()[-1].strip() for x in open(in_file, 'r', encoding="utf-8") if x.strip()]
        # Make sure we always add the labels in the same order
        for item in cur_labels:
            if item not in labels:
                labels.append(item)

    # Let training arguments know about the labels
    train_args["label_list"] = labels
    train_args["output_dir"] = args.output_dir

    # Let arguments know about the seed and use the seed in all places
    train_args["manual_seed"] = args.seed

    # Create the model - hardcoded to use GPU
    model = NERModel(args.lm_type, args.lm_ident, labels=labels, use_cuda=True, args=train_args)

    # Model can train directly from input files
    global_step, train_res = model.train_model(args.train_file, eval_data=args.test_files[0], args=train_args, verbose=True)

    # Print loss and f1-scores of dev set during training
    losses = train_res["eval_loss"]
    scores = train_res["f1_score"]
    print(f"Training eval losses: {losses}")
    print(f"Training f1-scores on dev: {scores}\n")

    # Evaluate for all specified test files and print results
    for idx, test_file in enumerate(args.test_files):
        result, model_outputs, predictions = model.eval_model(test_file)
        print (f"Result: {result}")
        for met in ["precision", "recall", "f1_score"]:
            val = round(result[met], 4)
            print (f"{met} for {args.names[idx]}: {val}")
        # Hard reset of result to prevent issues (probably not necessary)
        result = {}


if __name__ == '__main__':
    main()
