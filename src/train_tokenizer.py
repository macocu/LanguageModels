#!/usr/bin/env python
# -*- coding: utf8 -*-

'''Train a Byte-level BPE huggingface tokenizer'''

import sys
import argparse
from tokenizers import ByteLevelBPETokenizer
from tokenizers.decoders import ByteLevel


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_files", nargs="+", required=True, type=str,
                        help="Files to use when training the tokenizer")
    parser.add_argument("-o", "--out", required=True, type=str,
                        help="Folder + name where we save the tokenizer .e.g tok/roberta")
    parser.add_argument("-v", "--vocab_size", default=50000, type=int,
                        help="Size of the vocab - default 50k")
    parser.add_argument("-m", "--min_freq", default=2, type=int,
                        help="Minimal frequency a pair should have in order to be merged")
    parser.add_argument("-t", "--test", action="store_true",
                        help="Encode as a test the first 5 sentences of the first specified file")
    args = parser.parse_args()
    return args


def test(tokenizer, input_file):
    '''Test the newly trained tokenizer and print the output'''
    decoder = ByteLevel()
    for idx, line in enumerate(open(input_file, 'r')):
        # Only do 5 sentences
        if idx > 4:
            break
        sent = line.strip()

        # Tokenize the sentence with our fresh tokenizer
        out = tokenizer.encode(sent)

        # Usually, to get the output you can just do this
        #print (tokenizer.decode(out.ids))

        # But we want to see the actual pieces, so we have to do this:
        # Bit complicated because we have to go back to the original representation
        output = [decoder.decode(b) for b in out.tokens]
        print (sent + '\n' + " ".join(output) + '\n')


def main():
    '''Main function for training the tokenizer'''
    args = create_arg_parser()

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Set the special tokens: currently setup for RoBERTa models
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

    # Train the tokenizer
    tokenizer.train(files=args.input_files, vocab_size=args.vocab_size, min_frequency=args.min_freq,
                    special_tokens=special_tokens)

    # Save to the folder
    tokenizer.save_model(args.out)

    # Test if the output looks OK like this:
    if args.test:
        test(tokenizer, args.input_files[0])


if __name__ == '__main__':
    main()
